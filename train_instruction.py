import random
import tqdm
import gzip
import numpy as np
import json
import os

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

try:
    from adam_atan2_pytorch import AdoptAtan2
except ImportError:
    print("AdoptAtan2 not found, using Adam optimizer instead")
    from torch.optim import Adam as AdoptAtan2

from titans_pytorch import (
    MemoryAsContextTransformer,
    MemoryMLP,
    MemoryAttention
)

# constants - 训练参数
NUM_BATCHES = 2       # 训练批次数量
BATCH_SIZE = 2000         # 批量大小
GRADIENT_ACCUMULATE_EVERY = 2  # 梯度累积
LEARNING_RATE = 8e-5     # 学习率
VALIDATE_EVERY = 50      # 验证频率
GENERATE_EVERY = 100     # 生成频率
SAVE_EVERY = 200         # 保存频率
MAX_INSTRUCTION_LEN = 64   # 指令最大长度
MAX_ANSWER_LEN = 128     # 回答最大长度
GENERATE_LENGTH = 128     # 生成长度
SHOULD_GENERATE = True

# 增强模型设置 - 提高模型记忆能力
DIM = 512        # 增加模型维度到512
DIM_HEAD = 64    # 保持每个头的维度
HEADS = 8        # 增加注意力头数量到8
DEPTH = 6        # 增加深度到6层
MEMORY_DEPTH = 3 # 增加内存深度到3
WINDOW_SIZE = 64 # 增加窗口大小到64
MEM_SEGMENT_LEN = 32  # 增加内存段长度到32
MEM_BATCH_SIZE = 64  # 增加内存批量大小到64

# 神经记忆设置 - 增强记忆能力
NUM_PERSIST_MEM = 12  # 增加持久内存数量到12
NUM_LONGTERM_MEM = 16 # 增加长期内存数量到16
NEURAL_MEM_LAYERS = (1, 2, 3, 4)  # 增加使用神经记忆的层数
NEURAL_MEM_GATE_ATTN_OUTPUT = True  # 门控注意力输出
NEURAL_MEM_MOMENTUM = False  # 内存动量
NEURAL_MEM_MOMENTUM_ORDER = 1
NEURAL_MEM_QK_NORM = False  # QK规范化
NEURAL_MEM_MAX_LR = 1e-3  # 最大学习率
USE_MEM_ATTENTION_MODEL = True  # 使用内存注意力模型
SLIDING_WINDOWS = False
STORE_ATTN_POOL_CHUNKS = False
MEMORY_MODEL_PER_LAYER_LEARNED_LR = False
NEURAL_MEM_WEIGHT_RESIDUAL = True
NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW = False

# 性能相关设置 - 禁用加速功能
USE_ACCELERATED_SCAN = False  
USE_FLEX_ATTN = False         
USE_FAST_INFERENCE = False    

# 特殊标记
INSTRUCTION_TOKEN = 1  # 指令标记的ID
ANSWER_TOKEN = 2      # 回答标记的ID
SEP_TOKEN = 3         # 分隔符标记的ID
PAD_TOKEN = 0         # 填充标记的ID

# helpers
def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    if token == INSTRUCTION_TOKEN:
        return "[INST]"
    elif token == ANSWER_TOKEN:
        return "[ANS]"
    elif token == SEP_TOKEN:
        return "[SEP]"
    elif token == PAD_TOKEN:
        return "[PAD]"
    else:
        return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

def encode_text(text):
    """将普通文本编码为token"""
    return torch.tensor([ord(c) for c in text], dtype=torch.long)

def encode_instruction_answer_pair(instruction, answer, input_text=""):
    """编码指令-回答对，支持Project Zomboid格式的数据"""
    # 截断过长的输入
    if len(instruction) > MAX_INSTRUCTION_LEN:
        instruction = instruction[:MAX_INSTRUCTION_LEN]
    
    # 处理输入文本
    input_text = input_text or ""  # 确保input不是None
    
    # 处理回答，移除<think>...</think>标签和内容
    clean_answer = answer
    if "<think>" in answer and "</think>" in answer:
        think_parts = answer.split("</think>")
        if len(think_parts) > 1:
            clean_answer = think_parts[1].strip()
    
    if "<answer>" in clean_answer:
        clean_answer = clean_answer.replace("<answer>", "").replace("</answer>", "")
    
    if len(clean_answer) > MAX_ANSWER_LEN:
        clean_answer = clean_answer[:MAX_ANSWER_LEN]
        
    # 编码为token序列
    inst_tokens = encode_text(instruction)
    
    # 如果有输入文本，添加输入文本
    if input_text:
        input_tokens = encode_text(input_text)
        # 构建序列：[INST] instruction [SEP] input [SEP] [ANS] answer
        seq = torch.cat([
            torch.tensor([INSTRUCTION_TOKEN]),
            inst_tokens,
            torch.tensor([SEP_TOKEN]),
            input_tokens,
            torch.tensor([SEP_TOKEN, ANSWER_TOKEN]),
            encode_text(clean_answer)
        ])
    else:
        # 构建序列：[INST] instruction [SEP] [ANS] answer
        seq = torch.cat([
            torch.tensor([INSTRUCTION_TOKEN]),
            inst_tokens,
            torch.tensor([SEP_TOKEN, ANSWER_TOKEN]),
            encode_text(clean_answer)
        ])
    
    return seq

# 自定义create_block_mask函数
def create_custom_mac_block_mask(seq_len, window_size, persist_mem_len, sliding_window_attn):
    """创建自定义注意力mask，无需Triton"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 创建一个简单的block mask
    total_len = seq_len + persist_mem_len
    block_mask = torch.ones(1, 1, total_len, total_len, device=device)
    
    # 持久化内存对所有位置都可见
    block_mask[:, :, :, :persist_mem_len] = 1
    
    # 自身位置和周围窗口可见
    for i in range(persist_mem_len, total_len):
        start = max(persist_mem_len, i - window_size)
        end = min(total_len, i + window_size + 1)
        block_mask[:, :, i, start:end] = 1
    
    # 前持久化内存部分对所有位置可见
    block_mask[:, :, :persist_mem_len, :] = 1
    
    # 转换为注意力掩码（0=保留，1=遮罩）
    block_mask = 1 - block_mask
    block_mask = block_mask * -1e9
    return block_mask

# 测试模型生成
def test_model_generation(model, instruction, device, max_length=100, input_text=""):
    """测试模型对指令的回答生成能力，支持输入文本"""
    model.eval()
    
    # 准备输入序列
    inst_tokens = encode_text(instruction).to(device)
    
    if input_text:
        # [INST] instruction [SEP] input [SEP] [ANS]
        input_tokens = encode_text(input_text).to(device)
        input_seq = torch.cat([
            torch.tensor([INSTRUCTION_TOKEN], device=device),
            inst_tokens,
            torch.tensor([SEP_TOKEN], device=device),
            input_tokens,
            torch.tensor([SEP_TOKEN, ANSWER_TOKEN], device=device)
        ])
    else:
        # [INST] instruction [SEP] [ANS]
        input_seq = torch.cat([
            torch.tensor([INSTRUCTION_TOKEN], device=device),
            inst_tokens,
            torch.tensor([SEP_TOKEN, ANSWER_TOKEN], device=device)
        ])
    
    # 输出原始提示
    print(f'指令: {instruction}')
    if input_text:
        print(f'输入: {input_text}')
    print('='*50)
    
    # 生成回答
    with torch.no_grad():
        output = model.sample(input_seq[None, ...], max_length, use_cache=USE_FAST_INFERENCE)
    
    # 解码并显示
    output_str = decode_tokens(output[0].cpu())
    answer_part = output_str.split('[ANS]')[-1]  # 获取回答部分
    
    print(f'生成回答:')
    print('-'*50)
    print(answer_part)
    print('-'*50)
    
    return answer_part

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 显示CUDA信息
if torch.cuda.is_available():
    print(f"CUDA设备数量: {torch.cuda.device_count()}")
    print(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")
    print(f"当前CUDA设备: {torch.cuda.current_device()}")
    print(f"CUDA可用内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 创建记忆模型
if USE_MEM_ATTENTION_MODEL:
    neural_memory_model = MemoryAttention(
        dim = DIM
    )
else:
    neural_memory_model = MemoryMLP(
        dim = DIM,
        depth = MEMORY_DEPTH
    )

print("创建模型...")

# 修补titans_pytorch中的函数
try:
    from titans_pytorch.mac_transformer import create_mac_block_mask
    import types
    
    # 仅当原始函数存在时尝试替换
    if hasattr(create_mac_block_mask, '__module__') and 'titans_pytorch.mac_transformer' in create_mac_block_mask.__module__:
        import titans_pytorch.mac_transformer
        titans_pytorch.mac_transformer.create_mac_block_mask = create_custom_mac_block_mask
        print("成功修补create_mac_block_mask函数")
except Exception as e:
    print(f"警告: 无法修补create_mac_block_mask: {e}")

# 创建模型实例
try:
    model = MemoryAsContextTransformer(
        num_tokens = 512,  # 基本ASCII + 特殊标记
        dim = DIM,
        depth = DEPTH,
        segment_len = WINDOW_SIZE,
        num_persist_mem_tokens = NUM_PERSIST_MEM,
        num_longterm_mem_tokens = NUM_LONGTERM_MEM,
        neural_memory_layers = NEURAL_MEM_LAYERS,
        neural_memory_segment_len = MEM_SEGMENT_LEN,
        neural_memory_batch_size = MEM_BATCH_SIZE,
        neural_mem_gate_attn_output = NEURAL_MEM_GATE_ATTN_OUTPUT,
        neural_mem_weight_residual = NEURAL_MEM_WEIGHT_RESIDUAL,
        neural_memory_qkv_receives_diff_views = NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW,
        use_flex_attn = USE_FLEX_ATTN,
        sliding_window_attn = SLIDING_WINDOWS,
        dim_head = DIM_HEAD,
        heads = HEADS,
        neural_memory_model = neural_memory_model,
        neural_memory_kwargs = dict(
            attn_pool_chunks = STORE_ATTN_POOL_CHUNKS,
            qk_rmsnorm = NEURAL_MEM_QK_NORM,
            momentum = NEURAL_MEM_MOMENTUM,
            momentum_order = NEURAL_MEM_MOMENTUM_ORDER,
            default_step_transform_max_lr = NEURAL_MEM_MAX_LR,
            use_accelerated_scan = USE_ACCELERATED_SCAN,
            per_parameter_lr_modulation = MEMORY_MODEL_PER_LAYER_LEARNED_LR
        )
    ).to(device)
    print("模型创建成功")
except Exception as e:
    print(f"创建模型错误: {e}")
    raise

class InstructionAnswerDataset(Dataset):
    """用于处理指令-回答对的数据集"""
    def __init__(self, data_file, max_len=512):  # 增加最大长度以适应wiki数据
        super().__init__()
        self.max_len = max_len
        self.data = []
        
        # 加载数据
        print(f"加载数据文件: {data_file}")
        try:
            # 判断文件类型（json或jsonl）
            if data_file.endswith('.jsonl'):
                # JSONL格式处理 - 一行一个JSON对象
                with open(data_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():  # 跳过空行
                            try:
                                item = json.loads(line)
                                # wiki3数据格式适配
                                instruction = item.get('instruction', '')
                                answer = item.get('output', '') or item.get('answer', '')
                                input_text = item.get('input', '')
                                
                                # 编码并添加到数据中
                                encoded = encode_instruction_answer_pair(instruction, answer, input_text)
                                if len(encoded) <= self.max_len:
                                    self.data.append(encoded)
                                else:
                                    print(f"警告: 序列长度 {len(encoded)} 超过最大长度 {self.max_len}，跳过")
                            except json.JSONDecodeError:
                                print(f"警告: 跳过无效的JSONL行")
            else:
                # 常规JSON文件处理
                with open(data_file, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                    
                # 处理数据
                for item in raw_data:
                    if 'instruction' in item and ('answer' in item or 'output' in item):
                        instruction = item['instruction']
                        answer = item.get('output', '') or item.get('answer', '')
                        input_text = item.get('input', '')
                        
                        # 编码并添加到数据中
                        encoded = encode_instruction_answer_pair(instruction, answer, input_text)
                        if len(encoded) <= self.max_len:
                            self.data.append(encoded)
                        else:
                            print(f"警告: 序列长度 {len(encoded)} 超过最大长度 {self.max_len}，跳过")
                    
            print(f"成功加载 {len(self.data)} 个指令-回答对")
        except Exception as e:
            print(f"加载数据失败: {e}")
            # 创建一些示例数据用于测试
            self._create_sample_data()
    
    def _create_sample_data(self):
        """创建样本数据用于测试"""
        print("创建样本数据...")
        sample_pairs = [
            ("What is Project Zomboid?", "Project Zomboid is a zombie survival game with an emphasis on realistic survival mechanics."),
            ("How do I find food in Project Zomboid?", "Look for food in refrigerators, cabinets, and grocery stores. You can also forage, fish, farm and trap animals."),
            ("What weapons are good for beginners?", "Baseball bats, kitchen knives, and hammers are good starting weapons. Avoid firearms until you have higher skills."),
            ("How do I heal injuries?", "Use bandages for cuts, splints for fractures, and painkillers for pain. Rest accelerates healing."),
            ("What is the most dangerous zombie?", "Sprinters are the most dangerous as they can catch up to you quickly. Avoid them if possible.")
        ]
        
        for instruction, answer in sample_pairs:
            encoded = encode_instruction_answer_pair(instruction, answer)
            self.data.append(encoded)
        
        print(f"创建了 {len(self.data)} 个样本数据对")

    def __getitem__(self, index):
        # 选择一个数据条目
        item = self.data[index]
        
        # 如果序列长度不足，进行填充
        if len(item) < self.max_len:
            padding = torch.full((self.max_len - len(item),), PAD_TOKEN, dtype=torch.long)
            item = torch.cat([item, padding])
        
        return item.to(device)

    def __len__(self):
        return len(self.data)

def create_or_load_datasets(data_file="data/wiki3.jsonl"):
    """创建或加载数据集"""
    # 确保数据目录存在
    os.makedirs(os.path.dirname(data_file), exist_ok=True)
    
    # 检查数据文件是否存在
    if not os.path.exists(data_file):
        print(f"数据文件 {data_file} 不存在，创建示例数据...")
        # 创建示例数据
        sample_data = []
        sample_pairs = [
            {"instruction": "What is Project Zomboid?", 
             "answer": "Project Zomboid is a zombie survival game with an emphasis on realistic survival mechanics."},
            {"instruction": "How do I find food in Project Zomboid?", 
             "answer": "Look for food in refrigerators, cabinets, and grocery stores. You can also forage, fish, farm and trap animals."},
            {"instruction": "What weapons are good for beginners?", 
             "answer": "Baseball bats, kitchen knives, and hammers are good starting weapons. Avoid firearms until you have higher skills."},
            {"instruction": "How do I heal injuries?", 
             "answer": "Use bandages for cuts, splints for fractures, and painkillers for pain. Rest accelerates healing."},
            {"instruction": "What is the most dangerous zombie?", 
             "answer": "Sprinters are the most dangerous as they can catch up to you quickly. Avoid them if possible."}
        ]
        
        # 添加更多随机的指令-回答对
        for i in range(20):
            instruction = f"This is test instruction {i+1}. Please respond appropriately."
            answer = f"I understand your instruction {i+1}. Here is my response with some context about zombies and survival techniques."
            sample_pairs.append({"instruction": instruction, "answer": answer})
        
        # 保存示例数据
        out_file = data_file.replace('.jsonl', '.json') if data_file.endswith('.jsonl') else data_file
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(sample_pairs, f, ensure_ascii=False, indent=2)
    
    print(f"开始加载数据集: {data_file}")
    # 创建数据集
    full_dataset = InstructionAnswerDataset(data_file)
    
    print(f"全数据集大小: {len(full_dataset)} 样本")
    
    # 分割数据集时，确保PZ相关数据分布均衡
    train_size = int(0.9 * len(full_dataset))  # 增加训练集比例到90%
    val_size = len(full_dataset) - train_size
    
    if val_size == 0:
        # 数据太少，只用于训练
        train_dataset = full_dataset
        val_dataset = full_dataset
        print("警告: 数据量太小，验证集与训练集相同")
    else:
        # 随机分割
        # 设置随机种子确保可重复性
        torch.manual_seed(42)
        indices = torch.randperm(len(full_dataset))
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        
        print(f"训练集大小: {len(train_dataset)} 样本")
        print(f"验证集大小: {len(val_dataset)} 样本")
    
    return train_dataset, val_dataset

# 创建数据集和数据加载器
print("创建数据集...")
train_dataset, val_dataset = create_or_load_datasets()
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0))
val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0))

# 初始化优化器
print("初始化优化器...")
optim = AdoptAtan2(model.parameters(), lr=LEARNING_RATE)

# 保存检查点函数
def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint.pt"):
    # 确保checkpoint目录存在
    os.makedirs("checkpoint", exist_ok=True)
    
    # 保存在checkpoint目录
    filepath = os.path.join("checkpoint", filename)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    try:
        torch.save(checkpoint, filepath)
        print(f"检查点已保存到 {filepath}")
    except Exception as e:
        print(f"保存检查点错误: {e}")

# 加载检查点函数
def load_checkpoint(model, optimizer, checkpoint_path):
    try:
        print(f"尝试加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"成功加载检查点，批次: {start_epoch}, 损失: {loss:.4f}")
        return start_epoch
    except Exception as e:
        print(f"加载检查点失败: {e}")
        return 0

# 尝试加载先前的检查点
start_batch = 0
checkpoint_files = ["instruction_final.pt", "instruction_checkpoint_50.pt"]
for checkpoint_file in checkpoint_files:
    checkpoint_path = os.path.join("checkpoint", checkpoint_file)
    if os.path.exists(checkpoint_path):
        start_batch = load_checkpoint(model, optim, checkpoint_path)
        if start_batch > 0:
            start_batch += 1
            print(f"从批次 {start_batch} 继续训练")
            break

# 记录损失
train_losses = []
val_losses = []

# 训练循环
print(f"开始训练 {NUM_BATCHES} 批次...")
try:
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 5  # 连续5次验证损失没有改善就降低学习率
    
    for i in tqdm.tqdm(range(start_batch, NUM_BATCHES), mininterval=1., desc='训练中'):
        model.train()
        
        # 梯度累积
        total_loss = 0
        for _ in range(GRADIENT_ACCUMULATE_EVERY):
            try:
                data_batch = next(train_loader)
                loss = model(data_batch, return_loss=True)
                loss = loss / GRADIENT_ACCUMULATE_EVERY
                loss.backward()
                total_loss += loss.item()
            except Exception as e:
                print(f"训练批次 {i} 错误: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # 记录损失
        train_loss = total_loss
        train_losses.append(train_loss)
        
        print(f'批次 {i+1}/{NUM_BATCHES}, 训练损失: {train_loss:.4f}')
        
        # 梯度裁剪防止爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        
        # 优化器步骤
        optim.step()
        optim.zero_grad()
        
        # 保存检查点
        if i % SAVE_EVERY == 0 and i > 0:
            try:
                save_checkpoint(model, optim, i, train_loss, f"instruction_checkpoint_{i}.pt")
            except Exception as e:
                print(f"批次 {i} 保存检查点失败: {e}")
        
        # 验证阶段
        if i % VALIDATE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                try:
                    val_data_batch = next(val_loader)
                    val_loss = model(val_data_batch, return_loss=True)
                    current_val_loss = val_loss.item()
                    val_losses.append(current_val_loss)
                    print(f'验证损失: {current_val_loss:.4f}')
                    
                    # 学习率调度 - 如果验证损失连续多次没有改善，降低学习率
                    if current_val_loss < best_val_loss:
                        best_val_loss = current_val_loss
                        patience_counter = 0
                        # 保存最佳模型
                        save_checkpoint(model, optim, i, current_val_loss, "instruction_best.pt")
                        print(f"新的最佳验证损失: {best_val_loss:.4f}, 已保存最佳模型")
                    else:
                        patience_counter += 1
                        print(f"验证损失未改善，耐心计数: {patience_counter}/{patience_limit}")
                        
                        if patience_counter >= patience_limit:
                            # 降低学习率
                            for param_group in optim.param_groups:
                                param_group['lr'] *= 0.5  # 学习率减半
                            new_lr = param_group['lr']
                            print(f"降低学习率至: {new_lr}")
                            patience_counter = 0  # 重置耐心计数器
                except Exception as e:
                    print(f"验证错误: {e}")
        
        # 生成文本样本
        if SHOULD_GENERATE and i % GENERATE_EVERY == 0:
            try:
                print(f"\n批次 {i+1} 生成样本:")
                sample_instructions = [
                    "What critical survival action should the agent take immediately upon hearing an approaching helicopter while outdoors?",
                    "How can the agent gain advance warning of the helicopter event during Days 6-9, including the specific tool and frequency to monitor?"
                ]
                for inst in sample_instructions:
                    test_model_generation(model, inst, device, GENERATE_LENGTH)
            except Exception as e:
                print(f"生成错误: {e}")
    
    # 完成训练
    print(f"训练完成，共 {NUM_BATCHES} 批次!")
    
    # 保存最终检查点
    final_loss = train_losses[-1] if train_losses else 0
    save_checkpoint(model, optim, NUM_BATCHES, final_loss, "instruction_final.pt")
    
    # 保存损失历史
    np.save("instruction_train_losses.npy", np.array(train_losses))
    np.save("instruction_val_losses.npy", np.array(val_losses))
    
    # 最终评估
    print("最终模型评估:")
    if SHOULD_GENERATE:
        test_instructions = [
            "What critical survival action should the agent take immediately upon hearing an approaching helicopter while outdoors?",
            "How can the agent gain advance warning of the helicopter event during Days 6-9, including the specific tool and frequency to monitor?",
            "What specific steps must the agent follow to obtain gasoline from a gas station pump after the main power grid has shut off?",
            "What type of location is a house heavily barricaded with wooden planks from the outside likely to be, and what tools are needed to remove the external barricades?",
            "What potential delayed consequence might occur after the agent sustains a scratch from a zombie, and if the \"Sick\" moodle appears afterward, what is the most likely cause and outcome?"
        ]
        for inst in test_instructions:
            test_model_generation(model, inst, device, GENERATE_LENGTH)

except KeyboardInterrupt:
    print("用户中断训练")
    try:
        current_batch = start_batch + len(train_losses)
        current_loss = train_losses[-1] if train_losses else 0
        save_checkpoint(model, optim, current_batch, current_loss, "instruction_interrupted.pt")
        print(f"已保存中断检查点，批次 {current_batch}")
    except:
        print("中断时无法保存检查点")
except Exception as e:
    print(f"训练过程错误: {e}")
    import traceback
    traceback.print_exc() 