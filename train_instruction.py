import random
import tqdm
import gzip
import numpy as np
import json
import os
import logging
from datasets import Dataset as HFDataset

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入自定义的path_manager模块
from titans_pytorch.path_manager import (
  get_data_file_path,
  get_model_output_path,
  get_final_model_path,
  get_interrupted_model_path,
  get_logs_dir,
  ensure_dir_exists,
  get_checkpoint_dir
)

# 禁用PyTorch动态编译和JIT
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True  # 禁用动态编译错误
torch.jit.enable = False  # 禁用JIT编译

# 禁用动态优化
os.environ["PYTORCH_JIT"] = "0"
os.environ["TORCH_COMPILE"] = "0"
os.environ["PYTORCH_NO_CUDA_MEMORY_EFFICIENCY_WARNING"] = "1"

from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import (
  AutoModelForCausalLM,
  AutoTokenizer,
  TrainingArguments,
  Trainer,
  DataCollatorForLanguageModeling
)

# 全局变量，在main函数中初始化
model = None
tokenizer = None
generation_config = {
  "temperature": 0.6,       # 使用更高的温度以增加多样性
  "top_p": 0.9,            # 调整top-p sampling
  "top_k": 50,             # 增加token选择范围
  "repetition_penalty": 1.5, # 适度重复惩罚
  "no_repeat_ngram_size": 3, # 避免重复n-gram
  "num_beams": 1,          # 不使用beam search
  "do_sample": True,       # 启用采样
  "early_stopping": False   # 不提前停止生成
}

# 训练参数
MAX_LENGTH = 128  # 减少最大序列长度以节省内存
BATCH_SIZE = 2 if torch.cuda.is_available() else 1  # GPU可使用更大的批量
LEARNING_RATE = 2e-5  # 学习率
NUM_EPOCHS = 10   # 增加训练轮数以提高微调效果
GRADIENT_ACCUMULATION_STEPS = 4  # 梯度累积步数
SAVE_EVERY = 50      # 保存检查点频率
VALIDATE_EVERY = 25   # 验证频率
PRIME_LENGTH = 32     # 减小提示长度
GENERATE_LENGTH = 64  # 减小生成长度
SHOULD_GENERATE = True

# 模型参数 - 改为使用更小的Qwen3-0.6B
MODEL_NAME = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = get_model_output_path()  # 使用path_manager

# 确保checkpoint目录存在
ensure_dir_exists("checkpoint")

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# 显示设备信息
if torch.cuda.is_available():
  logger.info(f"CUDA device count: {torch.cuda.device_count()}")
  logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
  logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
  logger.info(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
  logger.info(f"Running on CPU - PyTorch Version: {torch.__version__}")

def load_model_and_tokenizer():
  """加载模型和tokenizer"""
  global model, tokenizer
  
  try:
    logger.info(f"Loading model: {MODEL_NAME}")
    
    # 模型配置
    model_kwargs = {
      "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
      "low_cpu_mem_usage": True
    }
    
    # 根据设备设置适当的加载参数
    if torch.cuda.is_available():
      logger.info("Loading model with CUDA optimizations")
      # 获取可用GPU内存
      free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
      # 为模型分配合适的内存限制，留出30%缓冲
      mem_limit = int(free_mem * 0.7 / 1024**3)
      logger.info(f"设置GPU内存限制: {mem_limit}GiB (可用内存的70%)")
      
      model_kwargs.update({
        "device_map": "auto",
        "max_memory": {0: f"{mem_limit}GiB"}
      })
    
    # 加载tokenizer优先，因为它内存占用小
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 确保tokenizer设置正确
    if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
      MODEL_NAME,
      **model_kwargs
    )
    
    # 设置生成配置
    logger.info("配置模型生成参数...")
    for param, value in generation_config.items():
      if hasattr(model.generation_config, param):
        setattr(model.generation_config, param, value)
        logger.info(f"设置生成参数 {param} = {value}")
    
    logger.info(f"Model loaded: {model.config}")
    
    # 确认模型是否在CUDA上
    logger.info(f"Model is on CUDA: {next(model.parameters()).is_cuda}")
    if torch.cuda.is_available():
      logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
      logger.info(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
  except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise
    
  return model, tokenizer

def load_instruction_dataset(data_file=None):
  """加载指令数据集"""
  if data_file is None:
    data_file = get_data_file_path("wiki3.jsonl")
    
  # 确保数据目录存在
  ensure_dir_exists(os.path.dirname(data_file))
  
  data_items = []
  
  # 加载数据
  logger.info(f"Loading data file: {data_file}")
  try:
    # 判断文件类型（json或jsonl）
    if data_file.endswith('.jsonl'):
      # JSONL格式处理 - 一行一个JSON对象
      with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
          if line.strip():  # 跳过空行
            try:
              item = json.loads(line)
              data_items.append(item)
            except json.JSONDecodeError:
              logger.warning(f"Skipping invalid JSONL line")
    else:
      # 常规JSON文件处理
      with open(data_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
      # 处理数据
      for item in raw_data:
        if 'instruction' in item and ('answer' in item or 'output' in item):
          data_items.append(item)
          
    logger.info(f"Successfully loaded {len(data_items)} instruction-answer pairs")
    
    if len(data_items) == 0:
      raise ValueError("No valid data items found")
      
  except Exception as e:
    logger.error(f"Failed to load data: {e}")
    # 创建一些示例数据用于测试
    logger.info("Creating sample data...")
    data_items = _create_sample_data()
  
  return data_items

def _create_sample_data():
  """创建样本数据用于测试"""
  sample_pairs = [
    {
      "instruction": "What is Project Zomboid?", 
      "answer": "Project Zomboid is a zombie survival game with an emphasis on realistic survival mechanics."
    },
    {
      "instruction": "How do I find food in Project Zomboid?", 
      "answer": "Look for food in refrigerators, cabinets, and grocery stores. You can also forage, fish, farm and trap animals."
    },
    {
      "instruction": "What weapons are good for beginners?", 
      "answer": "Baseball bats, kitchen knives, and hammers are good starting weapons. Avoid firearms until you have higher skills."
    },
    {
      "instruction": "How do I heal injuries?", 
      "answer": "Use bandages for cuts, splints for fractures, and painkillers for pain. Rest accelerates healing."
    },
    {
      "instruction": "What is the most dangerous zombie?", 
      "answer": "Sprinters are the most dangerous as they can catch up to you quickly. Avoid them if possible."
    }
  ]
  
  # 添加更多随机的指令-回答对
  for i in range(20):
    instruction = f"This is test instruction {i+1}. Please respond appropriately."
    answer = f"I understand your instruction {i+1}. Here is my response with some context about zombies and survival techniques."
    sample_pairs.append({"instruction": instruction, "answer": answer})
  
  logger.info(f"Created {len(sample_pairs)} sample data pairs")
  return sample_pairs

def load_system_prompt(prompt_file="data/pz_system_prompt_simple.txt"):
  """加载系统提示"""
  try:
    with open(prompt_file, 'r', encoding='utf-8') as f:
      return f.read().strip()
  except Exception as e:
    # 如果找不到简化版系统提示，尝试加载原始系统提示
    logger.warning(f"无法加载简化系统提示文件 {prompt_file}: {e}")
    try:
      with open("data/pz_system_prompt.txt", 'r', encoding='utf-8') as f:
        return f.read().strip()
    except Exception as e2:
      logger.warning(f"无法加载原始系统提示文件: {e2}")
      return "You are an AI assistant providing helpful information about Project Zomboid, a zombie survival game. Answer directly and concisely."

def prepare_dataset(data_items):
  """准备数据集 - 简化版本不使用系统提示"""
  processed_data = []
  
  for item in data_items:
    instruction = item.get("instruction", "")
    answer = item.get("output", "") or item.get("answer", "")
    input_text = item.get("input", "")
    
    if not answer:
      logger.warning(f"Skipping item with empty answer: {instruction}")
      continue
      
    # 清理回答，移除<think>...</think>标签和内容
    clean_answer = answer
    if "<think>" in answer and "</think>" in answer:
      think_parts = answer.split("</think>")
      if len(think_parts) > 1:
        clean_answer = think_parts[1].strip()
    
    if "<answer>" in clean_answer:
      clean_answer = clean_answer.replace("<answer>", "").replace("</answer>", "")
      
    # 不使用系统提示，直接构建问答对
    if input_text:
      formatted_text = f"{instruction}\n{input_text}\n\n{clean_answer}"
    else:
      formatted_text = f"{instruction}\n\n{clean_answer}"
    
    processed_data.append({
      "text": formatted_text
    })
  
  # 返回简化的数据集
  return HFDataset.from_list(processed_data)

def format_instruction_dataset(examples):
  """简化版本的数据格式化函数"""
  # 数据已经是文本格式，不需要进一步处理
  return {"text": examples["text"]}

def test_model_generation(model, tokenizer, prompt, max_new_tokens=150):
  """超简化版生成函数 - 直接传入提示生成"""
  logger.info(f"生成测试 - 提示: {prompt}")
  
  # 将提示转换为模型输入
  encoded_input = tokenizer.encode(prompt, return_tensors="pt")
  
  # 将输入移动到适当的设备
  if torch.cuda.is_available():
    encoded_input = encoded_input.to(device)
    model = model.to(device)
  
  # 生成参数
  gen_params = {
    "max_new_tokens": max_new_tokens,
    "temperature": 0.6,  # 更高的温度以避免复制
    "top_p": 0.92,
    "top_k": 50,
    "do_sample": True,
    "no_repeat_ngram_size": 3,
    "repetition_penalty": 1.3,
  }
  
  # 生成文本
  try:
    with torch.no_grad():
      output_ids = model.generate(
        input_ids=encoded_input, 
        **gen_params
      )
    
    # 解码生成的文本
    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # 移除原始提示部分
    if full_output.startswith(prompt):
      response = full_output[len(prompt):].strip()
    else:
      response = full_output.strip()
      
    # 简单清理系统提示相关内容
    response = response.replace("You are an assistant specialized in the video game Project Zomboid.", "").strip()
    response = response.replace("Project Zomboid is a zombie survival game", "").strip()
    response = response.replace("When answering questions:", "").strip()
    response = response.replace("- Provide only factual game information", "").strip()
    response = response.replace("- Focus on specific", "").strip()
    response = response.replace("- Keep responses direct", "").strip()
    response = response.replace("- Never make up game features", "").strip()
    response = response.replace("Your goal is to help players", "").strip()
    
    # 确保不是空响应
    if not response or len(response.split()) < 3:
      response = "生成失败。尝试使用不同的提示或重新训练模型。"
      
  except Exception as e:
    logger.error(f"生成过程错误: {e}")
    response = f"生成错误: {str(e)}"
  
  # 记录结果
  logger.info(f"提示: {prompt}")
  logger.info("="*50)
  logger.info(f"生成回答: {response}")
  logger.info("="*50)
  
  return response

def clean_generated_text(text):
  """清理生成的文本，移除特殊标签、标记和格式问题"""
  if not text or len(text.strip()) == 0:
    return ""
  
  # 记录原始文本长度，用于调试
  original_length = len(text)
  
  try:
    # 步骤1: 尝试提取ChatML格式中的助手回复
    if "<|im_start|>assistant" in text:
      parts = text.split("<|im_start|>assistant")
      if len(parts) > 1:
        assistant_text = parts[-1]  # 取最后一个
        if "<|im_end|>" in assistant_text:
          assistant_text = assistant_text.split("<|im_end|>")[0]
        text = assistant_text.strip()
    
    # 步骤2: 移除<think>标签及其内容
    import re
    # 移除所有<think>...</think>内容
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # 如果有未闭合的<think>标签，去除它后面的所有内容
    if "<think>" in text:
      text = text.split("<think>")[0].strip()
    
    # 步骤3: 清除所有常见的XML/HTML标签
    tags_to_remove = [
      " ", " ", "<system>", "</system>", "<human>", "</human>", 
      "<assistant>", "</assistant>", "<output>", "</output>", "<o>", "</o>",
      "<input>", "</input>", "<i>", "</i>", "<think>", "</think>",
      "<answer>", "</answer>", "<|im_start|>", "<|im_end|>"
    ]
    
    for tag in tags_to_remove:
      text = text.replace(tag, "")
    
    # 步骤4: 移除markdown格式
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # 移除粗体 **text**
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # 移除斜体 *text*
    
    # 步骤5: 移除系统提示相关内容（通过关键词检测）
    # 这些是常见的指令/提示文本的开头部分，需要被移除
    system_prompt_patterns = [
      r"You (must|are|should) answer questions with (accurate|exact|specific).*?information from (the game|Project Zomboid)\.",
      r"Your responses should(\s|\:).*",
      r"Remember\: Project Zomboid is a zombie survival game.*",
      r".*?1\.\s+Be direct and concise.*",
      r".*?2\.\s+Include exact game mechanics.*",
      r".*?3\.\s+Name specific in-game.*",
      r".*?4\.\s+Avoid general statements.*",
      r".*?5\.\s+Never use markdown.*",
      r".*?DO NOT invent game features.*",
      r"You must answer questions with.*",
      r"Project Zombology.*",  # 明显错误的名称
      r"Project Zombody.*",    # 明显错误的名称
      r"Project ZOMBoid.*",    # 错误的大小写
      r"Players must manage.*?食物 \(food\).*"  # 混合语言内容
    ]
    
    # 对每个模式应用替换
    for pattern in system_prompt_patterns:
      text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
    
    # 步骤6: 清理列表格式和空白
    # 规范化列表
    text = re.sub(r'(\d+\.\s+)', r'\n\1', text)   # 确保数字列表项前有换行
    text = re.sub(r'(\-\s+)', r'\n\1', text)      # 确保破折号列表项前有换行
    
    # 检测并修复截断句子
    if text.strip().endswith(('to', 'the', 'and', 'or', 'of', 'in', 'as', 'for', 'with', 'action')):
      text = text.strip() + "..."  # 添加省略号表示句子被截断
    
    # 移除多余空白行和空格
    lines = [line.strip() for line in text.split('\n')]
    lines = [line for line in lines if line]  # 移除空行
    
    if lines:  # 确保有内容
      text = '\n'.join(lines)
      text = re.sub(r'\s+', ' ', text).strip()  # 规范化空白字符
    
    # 步骤7: 最终清理 - 移除多余内容
    # 移除可能的指令性开头
    text = re.sub(r'^(Here\'s|Here is|Let me|I will|The answer is|To answer your question|In Project Zomboid,?)\s+', '', text, flags=re.IGNORECASE)
    
    # 移除重复的段落（如果某一段内容重复出现）
    segments = text.split('. ')
    if len(segments) > 3:
      unique_segments = []
      for segment in segments:
        if segment and segment.strip() and segment.strip() not in unique_segments:
          unique_segments.append(segment.strip())
      text = '. '.join(unique_segments)
      if not text.endswith('.'):
        text += '.'
  
  except Exception as e:
    logger.warning(f"文本清理过程中出错: {e}")
    # 如果清理过程出错，则返回原始文本的简单清理版本
    text = re.sub(r'<.*?>', '', text)  # 移除所有XML标签
  
  # 如果清理后文本为空但原文不为空，返回警告信息
  if not text.strip() and original_length > 0:
    return "无法提取有效回答内容，请尝试更改问题表述。"
  
  # 最终检查 - 如果结果文本非常短，但原始文本很长，可能是清理过度
  if len(text.strip()) < 20 and original_length > 200:
    # 尝试最简单的清理方式
    simple_clean = re.sub(r'<.*?>', '', text)
    if len(simple_clean.strip()) > len(text.strip()):
      return simple_clean.strip()
  
  return text.strip()

# 创建自定义Trainer类来处理损失计算
class CausalLMTrainer(Trainer):
  def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    """
    修改计算损失的方法确保正确计算语言模型损失
    """
    if "labels" not in inputs:
      # 如果没有标签，我们将输入IDs作为标签
      inputs["labels"] = inputs["input_ids"].clone()
      # 将填充标记设置为-100，避免在计算损失时考虑这些标记
      if self.tokenizer.pad_token_id is not None:
        labels = inputs["labels"].detach().clone()
        pad_mask = labels == self.tokenizer.pad_token_id
        labels[pad_mask] = -100
        inputs["labels"] = labels
    
    # 打印设备信息以确认是否在CUDA上
    if hasattr(self, "_print_device_info") and self._print_device_info:
      logger.info(f"Compute loss - Input IDs device: {inputs['input_ids'].device}")
      logger.info(f"Model device: {next(model.parameters()).device}")
      self._print_device_info = False
    
    # 计算模型输出和损失
    outputs = model(**inputs)
    loss = outputs.loss
    
    # 处理潜在的NaN损失
    if torch.isnan(loss).any() or torch.isinf(loss).any():
      logger.warning(f"检测到NaN或Inf损失: {loss}，将其替换为大数值")
      loss = torch.where(torch.isnan(loss) | torch.isinf(loss), torch.tensor(100.0, device=loss.device), loss)
    
    return (loss, outputs) if return_outputs else loss
  
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._print_device_info = True  # 第一次计算损失时打印设备信息

def optimize_model_for_training(model, use_fp32=True):
  """优化模型配置以便进行训练"""
  if model is None:
    return None
    
  # 针对Qwen3模型的优化
  if "qwen" in model.config.model_type.lower():
    logger.info("Applying Qwen-specific optimizations")
    
    # 减少注意力头数量以节省内存
    if hasattr(model.config, "num_attention_heads") and model.config.num_attention_heads > 8:
      logger.info(f"Original attention heads: {model.config.num_attention_heads}")
      # Qwen3的注意力头是内部参数，不应修改
    
    # 确保模型处于正确的数据类型
    if use_fp32 and next(model.parameters()).dtype == torch.float16:
      logger.info("Converting model from fp16 to fp32 for training stability")
      model = model.to(torch.float32)
    
    # 确保滑动窗口关闭以提高训练效率
    if hasattr(model.config, "use_sliding_window"):
      original_value = model.config.use_sliding_window
      model.config.use_sliding_window = False
      logger.info(f"Disabled sliding window for training (was: {original_value})")
    
    # 禁用KV缓存以节省内存
    if hasattr(model.config, "use_cache"):
      original_value = model.config.use_cache
      model.config.use_cache = False
      logger.info(f"Disabled KV cache for training (was: {original_value})")
      
  return model

def main():
  """主函数"""
  global model, tokenizer
  
  # 加载模型和tokenizer
  model, tokenizer = load_model_and_tokenizer()
  
  # 为训练优化模型
  model = optimize_model_for_training(model, use_fp32=True)
  
  # 加载数据
  data_items = load_instruction_dataset()
  
  # 为了节省内存，只使用部分数据
  if len(data_items) > 100:
    logger.info(f"Using only 100 examples out of {len(data_items)} to save memory")
    # 随机选择100个样本
    random.seed(42)
    data_items = random.sample(data_items, 100)
  
  # 准备数据集
  dataset = prepare_dataset(data_items)
  logger.info(f"Prepared dataset with {len(dataset)} examples")
  
  # 分割数据集
  train_size = int(0.9 * len(dataset))
  val_size = len(dataset) - train_size
  
  if val_size == 0:
    # 数据太少，只用于训练
    train_dataset = dataset
    val_dataset = dataset
    logger.warning("Warning: Too few data, validation set is the same as training set")
  else:
    # 随机分割
    dataset = dataset.shuffle(seed=42)
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, len(dataset)))
    
    logger.info(f"Training set size: {len(train_dataset)} examples")
    logger.info(f"Validation set size: {len(val_dataset)} examples")
  
  # 格式化数据集
  formatted_train_dataset = train_dataset.map(
    lambda examples: format_instruction_dataset(examples),
    batched=True,
    remove_columns=train_dataset.column_names
  )
  
  formatted_val_dataset = val_dataset.map(
    lambda examples: format_instruction_dataset(examples),
    batched=True,
    remove_columns=val_dataset.column_names
  )
  
  # Tokenize数据集
  def tokenize_function(examples):
    """对文本进行标记，并准备用于训练的数据格式"""
    # 获取文本
    texts = examples["text"]
    result = tokenizer(
      texts,
      truncation=True,
      max_length=MAX_LENGTH,
      padding="max_length",
      return_attention_mask=True
    )
    
    # 添加标签 - 语言模型训练中，标签与输入相同
    result["labels"] = result["input_ids"].copy()
    
    return result
  
  tokenized_train_dataset = formatted_train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
  )
  
  tokenized_val_dataset = formatted_val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
  )
  
  # 打印数据集示例
  logger.info("Dataset example:")
  try:
    sample_item = tokenized_train_dataset[0]
    logger.info(f"Example 1:")
    logger.info(f"Input IDs type: {type(sample_item['input_ids'])}")
    logger.info(f"Input IDs first few tokens: {sample_item['input_ids'][:10]}")
    logger.info(f"Attention mask first few tokens: {sample_item['attention_mask'][:10]}")
    logger.info(f"Decoded text start: {tokenizer.decode(sample_item['input_ids'][:50])}...")
  except Exception as e:
    logger.error(f"Error inspecting dataset: {e}")
    # 打印原始数据集的内容
    logger.info(f"Raw dataset first item: {formatted_train_dataset[0]}")
    
    # 尝试修复数据集
    logger.info("Attempting to fix dataset format...")
    # 确保数据集格式正确
    from transformers import DefaultDataCollator
    data_collator = DefaultDataCollator()
  
  # 创建Data Collator - 使用特别为因果语言模型设计的数据整理器
  from transformers import DataCollatorForLanguageModeling
  
  # 使用适合因果语言模型的数据整理器
  data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # 不使用掩码语言模型（MLM），而是使用因果语言模型（CLM）
  )
  
  # 设置训练参数 - 关闭fp16导致的梯度问题
  use_fp16 = False
  
  training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_dir=get_logs_dir(),  # 使用path_manager
    logging_steps=5,  # 更频繁地记录日志
    save_strategy="epoch",
    save_total_limit=1,  # 只保留最近的1个检查点
    # CUDA设置 - 关闭fp16避免梯度问题
    fp16=use_fp16,
    bf16=False,
    # GPU设置
    no_cuda=False,  # 允许使用CUDA
    dataloader_num_workers=0,  # 不使用多进程以节省内存
    auto_find_batch_size=True,  # 自动调整批量大小以避免OOM
    # 使用合适的优化器
    optim="adamw_torch",  # 使用PyTorch的AdamW优化器
    # 其他设置
    remove_unused_columns=True,  # 移除不使用的列以节省内存
    group_by_length=True,  # 按序列长度分组以减少填充
    report_to="none",  # 不报告给任何平台
    # 避免GPU断言错误的设置
    ddp_find_unused_parameters=False,
    use_legacy_prediction_loop=True,
    label_smoothing_factor=0.0,
    # 禁用梯度缩放以避免FP16梯度问题
    gradient_checkpointing=True,  # 启用梯度检查点以节省内存
    max_grad_norm=1.0,  # 设置梯度裁剪以避免爆炸梯度
  )
  
  # 使用自定义Trainer进行训练
  trainer = CausalLMTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
  )
  
  # 开始训练
  logger.info("Starting training...")
  try:
    # 确保模型在正确的设备上
    if torch.cuda.is_available() and not next(model.parameters()).is_cuda:
      logger.info("Moving model to CUDA manually...")
      model = model.to(device)
      logger.info(f"After moving, model is on CUDA: {next(model.parameters()).is_cuda}")
    
    # 打印训练配置信息
    logger.info(f"Training on device: {device}")
    logger.info(f"FP16 enabled: {training_args.fp16}")
    logger.info(f"Model dtype: {next(model.parameters()).dtype}")
    logger.info(f"Training batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    
    # 开始训练
    trainer.train()
    
    # 保存最终模型
    logger.info("Saving model...")
    final_model_path = get_final_model_path()
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info(f"Training completed! Model saved to {final_model_path}")
    
    # 测试最终模型
    if SHOULD_GENERATE:
      logger.info("测试最终模型生成能力...")
      
      # 确保模型处于评估模式
      model.eval()
      
      # 使用直接的测试提示
      test_prompts = [
        "What weapons should I use against zombies in Project Zomboid?",
        "How do I siphon gas from cars in Project Zomboid?",
        "What are the symptoms of zombie infection in Project Zomboid?",
        "How do I build a base in Project Zomboid?",
        "How do I increase my carpentry skill in Project Zomboid?"
      ]
      
      # 使用合适的生成长度
      test_generate_tokens = 512
      
      # 逐个测试生成
      all_results = []
      for i, prompt in enumerate(test_prompts):
        logger.info(f"测试生成 {i+1}/{len(test_prompts)}")
        
        # 在每次生成前清理GPU内存
        if torch.cuda.is_available():
          torch.cuda.empty_cache()
          
        # 生成回答
        try:
          generated_text = test_model_generation(model, tokenizer, prompt, test_generate_tokens)
          all_results.append({"prompt": prompt, "response": generated_text})
        except Exception as e:
          logger.error(f"生成失败: {e}")
          all_results.append({"prompt": prompt, "response": f"Error: {str(e)}"})
      
      # 保存生成结果 - 移到循环外部，只保存一次所有结果
      try:
        result_path = os.path.join(get_checkpoint_dir(), "generation_results.json")
        with open(result_path, "w", encoding="utf-8") as f:
          json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"生成结果已保存到 {result_path}")
      except Exception as e:
        logger.error(f"保存生成结果失败: {e}")
        
  except KeyboardInterrupt:
    logger.info("Training interrupted by user")
    # 保存中断时的模型
    interrupted_model_path = get_interrupted_model_path()
    trainer.save_model(interrupted_model_path)
    tokenizer.save_pretrained(interrupted_model_path)
    logger.info(f"Saved interrupted model checkpoint to {interrupted_model_path}")
    
  except Exception as e:
    logger.error(f"Training error: {e}")
    import traceback
    traceback.print_exc()

if __name__ == "__main__":
  main() 