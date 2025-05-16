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
  "temperature": 0.7,
  "top_p": 0.95,
  "top_k": 50,
  "presence_penalty": 1.1,
}

# 训练参数
MAX_LENGTH = 128  # 减少最大序列长度以节省内存
BATCH_SIZE = 2 if torch.cuda.is_available() else 1  # GPU可使用更大的批量
LEARNING_RATE = 5e-5  # 学习率
NUM_EPOCHS = 1    # 减少训练轮数以快速完成
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
      model_kwargs.update({
        "device_map": "auto",
        "max_memory": {0: "7GiB"}  # 限制GPU内存使用
      })
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
      MODEL_NAME,
      **model_kwargs
    )
    
    # 使用标准方式加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Qwen3已经有pad_token，不需要额外设置
    if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token
    
    # 设置生成配置
    model.generation_config.temperature = generation_config["temperature"]
    model.generation_config.top_p = generation_config["top_p"] 
    model.generation_config.top_k = generation_config["top_k"]
    model.generation_config.presence_penalty = generation_config["presence_penalty"]
    
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

def load_system_prompt(prompt_file="data/pz_system_prompt.txt"):
  """加载系统提示"""
  try:
    with open(prompt_file, 'r', encoding='utf-8') as f:
      return f.read().strip()
  except Exception as e:
    logger.warning(f"无法加载系统提示文件 {prompt_file}: {e}")
    return "You are an AI assistant providing helpful information about Project Zomboid, a zombie survival game."

def prepare_dataset(data_items):
  """准备数据集"""
  processed_data = []
  
  # 加载系统提示
  system_prompt = load_system_prompt()
  logger.info(f"Loaded system prompt ({len(system_prompt)} chars)")
  
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
      
    # 创建消息格式 - 适用于Qwen3，包含系统提示
    messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": instruction if not input_text else f"{instruction}\n\n{input_text}"},
      {"role": "assistant", "content": clean_answer}
    ]
    
    processed_data.append({
      "messages": messages
    })
  
  # 使用正确的方法创建数据集
  return HFDataset.from_list(processed_data)

def format_instruction_dataset(examples, tokenizer):
  """将数据集格式化为模型输入格式"""
  formatted_texts = []
  
  for messages in examples["messages"]:
    try:
      formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
      )
      formatted_texts.append(formatted_text)
    except Exception as e:
      logger.error(f"Error formatting message: {e}")
      logger.error(f"Message content: {messages}")
      # 使用简单格式作为fallback
      user_content = messages[0]["content"] if len(messages) > 0 else ""
      assistant_content = messages[1]["content"] if len(messages) > 1 else ""
      formatted_text = f"<|user|>\n{user_content}<|endoftext|>\n<|assistant|>\n{assistant_content}<|endoftext|>"
      formatted_texts.append(formatted_text)
  
  return {"text": formatted_texts}

def test_model_generation(model, tokenizer, prompt, max_new_tokens=100):
  """测试模型生成能力"""
  logger.info(f"Testing generation for prompt: {prompt}")
  
  # 加载系统提示
  system_prompt = load_system_prompt()
  
  # 为Qwen3模型准备输入格式，包含系统提示
  messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt}
  ]
  
  try:
    formatted_prompt = tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True
    )
  except Exception as e:
    logger.error(f"Error applying chat template: {e}")
    # Fallback to simple format
    formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
  
  inputs = tokenizer(formatted_prompt, return_tensors="pt")
  
  # 如果使用CUDA，把input_ids和attention_mask移到GPU上
  if torch.cuda.is_available():
    inputs = {k: v.to(device) for k, v in inputs.items()}
  
  # 生成文本 - 使用更安全的设置避免断言错误
  try:
    with torch.no_grad():
      # 修复：使用max_new_tokens而不是max_length
      generation_kwargs = {
        "max_new_tokens": max_new_tokens,  # 使用max_new_tokens而不是max_length
        "num_return_sequences": 1,
        "do_sample": True,  # 确保启用采样
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
      }
      
      # 不要在这里包含attention_mask
      generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        **generation_kwargs
      )
      
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # 添加：清理生成的文本，处理<think>标签
    generated_text = clean_generated_text(generated_text)
    
  except Exception as e:
    logger.error(f"Error in generation: {e}")
    generated_text = "Error generating text with GPU. Falling back to simple output."
    # 尝试CPU回退
    if torch.cuda.is_available():
      try:
        logger.info("Attempting CPU fallback for generation...")
        model_cpu = model.to('cpu')
        inputs_cpu = {k: v.to('cpu') for k, v in inputs.items()}
        
        with torch.no_grad():
          generated_ids = model_cpu.generate(
            input_ids=inputs_cpu["input_ids"],
            max_new_tokens=max_new_tokens,  # 同样使用max_new_tokens
            num_return_sequences=1,
            do_sample=True,  # 保持一致的采样设置
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
          )
        
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # 清理CPU回退生成的文本
        generated_text = clean_generated_text(generated_text)
        
        # 把模型移回GPU
        model.to(device)
      except Exception as inner_e:
        logger.error(f"CPU fallback also failed: {inner_e}")
  
  # 输出生成的文本
  logger.info(f"指令: {prompt}")
  logger.info("="*50)
  logger.info(f"生成回答: {generated_text}")
  logger.info("="*50)
  return generated_text

def clean_generated_text(text):
  """清理生成的文本，移除特殊标签和不必要的内容"""
  # 处理思考标签：移除<think>...</think>及其内容
  while "<think>" in text and "</think>" in text:
    think_start = text.find("<think>")
    think_end = text.find("</think>", think_start) + len("</think>")
    if think_start != -1 and think_end != -1:
      text = text[:think_start] + text[think_end:]
    else:
      break  # 防止无限循环
  
  # 清理其他标签
  tags_to_remove = [
    "<s>", "</s>", "<system>", "</system>", "<human>", "</human>", 
    "<assistant>", "</assistant>", "<output>", "</output>", "<o>", "</o>",
    "<input>", "</input>", "<i>", "</i>"
  ]
  
  for tag in tags_to_remove:
    text = text.replace(tag, "")
  
  # 处理ChatML格式的特殊情况
  if "<|im_start|>" in text:
    # 尝试提取助手回复部分
    parts = text.split("<|im_start|>assistant")
    if len(parts) > 1:
      # 获取助手部分并清理
      assistant_part = parts[1]
      if "<|im_end|>" in assistant_part:
        assistant_part = assistant_part.split("<|im_end|>")[0]
      text = assistant_part.strip()
    
  # 处理多余的空白字符
  text = " ".join(text.split())
  
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
      # 可选地设置填充标记为-100
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
    
    outputs = model(**inputs)
    loss = outputs.loss
    
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
    lambda examples: format_instruction_dataset(examples, tokenizer),
    batched=True,
    remove_columns=train_dataset.column_names
  )
  
  formatted_val_dataset = val_dataset.map(
    lambda examples: format_instruction_dataset(examples, tokenizer),
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
      logger.info("Testing final model...")
      
      # 恢复模型生成设置
      logger.info("Restoring model generation settings...")
      
      # 恢复模型生成设置
      if hasattr(model.config, "use_cache"):
        original_use_cache = model.config.use_cache
        model.config.use_cache = True
        logger.info(f"Re-enabled KV cache for generation (was: {original_use_cache})")
        
      # 确保模型处于评估模式
      model.eval()
      logger.info("Model set to evaluation mode")
      
      # 记录当前模型配置
      logger.info(f"Generation config: temperature={generation_config['temperature']}, top_p={generation_config['top_p']}")
      
      # 测试提示
      test_prompts = [
        "What critical survival action should the agent take immediately upon hearing an approaching helicopter while outdoors?",
        "How can the agent gain advance warning of the helicopter event during Days 6-9, including the specific tool and frequency to monitor?",
        "What specific steps must the agent follow to obtain gasoline from a gas station pump after the main power grid has shut off?",
        "What type of location is a house heavily barricaded with wooden planks from the outside likely to be, and what tools are needed to remove the external barricades?",
        "What potential delayed consequence might occur after the agent sustains a scratch from a zombie, and if the \"Sick\" moodle appears afterward, what is the most likely cause and outcome?"
      ]
      
      # 每个生成的令牌数量
      test_generate_tokens = 256  # 增加生成长度来获得更完整的回答
      
      # 逐个测试生成
      all_results = []
      for i, prompt in enumerate(test_prompts):
        logger.info(f"Testing generation {i+1}/{len(test_prompts)}")
        try:
          # 在每次生成前确保GPU内存清理
          if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
          # 进行文本生成
          generated_text = test_model_generation(model, tokenizer, prompt, test_generate_tokens)
          all_results.append({"prompt": prompt, "response": generated_text})
          
        except Exception as e:
          logger.error(f"Error generating response for prompt {i+1}: {e}")
          all_results.append({"prompt": prompt, "response": f"Error: {str(e)}"})
          
      # 保存生成结果
      try:
        result_path = os.path.join(get_checkpoint_dir(), "generation_results.json")
        with open(result_path, "w", encoding="utf-8") as f:
          json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Generation results saved to {result_path}")
      except Exception as e:
        logger.error(f"Failed to save generation results: {e}")
        
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