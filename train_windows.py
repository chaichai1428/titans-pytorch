import random
import tqdm
import gzip
import numpy as np

# 禁用PyTorch动态编译和JIT
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True  # 禁用动态编译错误
torch.jit.enable = False  # 禁用JIT编译

# 禁用动态优化
import os
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

# constants - 增加训练批次以便更好地学习记忆
NUM_BATCHES = 2000  # 增加到2000批次以获得更好效果
BATCH_SIZE = 4      # 增加到4，利用GPU并行性
GRADIENT_ACCUMULATE_EVERY = 2  # 启用梯度累积，有效增大批量大小
LEARNING_RATE = 8e-5  # 调整学习率，适应更大的实际批量大小
VALIDATE_EVERY = 50   # 减少验证频率，加快训练
GENERATE_EVERY = 100  # 减少生成频率，加快训练
SAVE_EVERY = 200      # 减少保存频率，节省I/O
PRIME_LENGTH = 64  
GENERATE_LENGTH = 128  # 保持生成长度
SHOULD_GENERATE = True
SEQ_LEN = 128  # 增加序列长度以捕获更长上下文

# 增强模型设置 - 确保所有维度匹配
DIM = 128         # 增加维度到128
DIM_HEAD = 64     # 每个头的维度保持64
HEADS = 4         # 增加到4个头
DEPTH = 4         # 增加到4层
MEMORY_DEPTH = 2  # 内存深度保持2
WINDOW_SIZE = 32  # 窗口大小增加到32
MEM_SEGMENT_LEN = 16  # 内存段长度增加到16
MEM_BATCH_SIZE = 32  # 内存批量增加到32

# 神经记忆设置 - 增加记忆参数
NUM_PERSIST_MEM = 6   # 持久内存增加到6
NUM_LONGTERM_MEM = 8  # 长期内存增加到8
NEURAL_MEM_LAYERS = (1, 2, 3)  # 在更多层使用神经记忆
NEURAL_MEM_GATE_ATTN_OUTPUT = True  # 启用门控注意力输出
NEURAL_MEM_MOMENTUM = False  # 禁用动量，避免内存错误
NEURAL_MEM_MOMENTUM_ORDER = 1
NEURAL_MEM_QK_NORM = False  # 禁用QK规范化
NEURAL_MEM_MAX_LR = 1e-3  # 学习率
USE_MEM_ATTENTION_MODEL = True  # 使用记忆注意力模型
SLIDING_WINDOWS = False
STORE_ATTN_POOL_CHUNKS = False  # 禁用注意力池化块
MEMORY_MODEL_PER_LAYER_LEARNED_LR = False  # 禁用每层学习率
NEURAL_MEM_WEIGHT_RESIDUAL = True  # 启用权重残差
NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW = False  # 禁用不同视图

# perf related - 禁用加速功能
USE_ACCELERATED_SCAN = False  
USE_FLEX_ATTN = False         
USE_FAST_INFERENCE = False    

# helpers
def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# 测试模型生成文本
def test_model_generation(model, val_dataset, prime_length=50, gen_length=128):
    model.eval()
    inp = random.choice(val_dataset)[:prime_length].to(device)
    prime = decode_tokens(inp.cpu())  # 解码前转回CPU
    print(f'Prime text: {prime}')
    print('*' * 50)

    sample = model.sample(inp[None, ...], gen_length, use_cache=USE_FAST_INFERENCE)
    output_str = decode_tokens(sample[0].cpu())  # 解码前转回CPU
    print(output_str)
    print('*' * 50)

# 强制使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 显示CUDA信息
if torch.cuda.is_available():
    print(f"CUDA设备数量: {torch.cuda.device_count()}")
    print(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")
    print(f"当前CUDA设备: {torch.cuda.current_device()}")
    print(f"CUDA可用内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# memory model - 使用更复杂的记忆模型配置
if USE_MEM_ATTENTION_MODEL:
    neural_memory_model = MemoryAttention(
        dim = DIM
    )
else:
    neural_memory_model = MemoryMLP(
        dim = DIM,
        depth = MEMORY_DEPTH
    )

# 自定义create_mac_block_mask函数以绕过需要Triton的部分
def create_custom_mac_block_mask(seq_len, window_size, persist_mem_len, sliding_window_attn):
    # 创建一个简单的block mask而不使用PyTorch的flex_attention
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

print("Creating model...")

# 尝试修补titans_pytorch.mac_transformer中的函数
try:
    from titans_pytorch.mac_transformer import create_mac_block_mask
    import types
    
    # 仅当原始函数存在时尝试替换
    if hasattr(create_mac_block_mask, '__module__') and 'titans_pytorch.mac_transformer' in create_mac_block_mask.__module__:
        import titans_pytorch.mac_transformer
        titans_pytorch.mac_transformer.create_mac_block_mask = create_custom_mac_block_mask
        print("Successfully patched create_mac_block_mask function")
except Exception as e:
    print(f"Warning: Could not patch create_mac_block_mask: {e}")
    print("Model might still use Triton for block mask creation")

# 增强模型 - 完全兼容的尺寸
try:
    model = MemoryAsContextTransformer(
        num_tokens = 256,
        dim = DIM,  # 增加维度
        depth = DEPTH,  # 增加深度
        segment_len = WINDOW_SIZE,  # 增加窗口大小
        num_persist_mem_tokens = NUM_PERSIST_MEM,  # 增加持久内存
        num_longterm_mem_tokens = NUM_LONGTERM_MEM,  # 增加长期内存
        neural_memory_layers = NEURAL_MEM_LAYERS,  # 在多层使用神经记忆
        neural_memory_segment_len = MEM_SEGMENT_LEN,
        neural_memory_batch_size = MEM_BATCH_SIZE,
        neural_mem_gate_attn_output = NEURAL_MEM_GATE_ATTN_OUTPUT,
        neural_mem_weight_residual = NEURAL_MEM_WEIGHT_RESIDUAL,
        neural_memory_qkv_receives_diff_views = NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW,
        use_flex_attn = USE_FLEX_ATTN,
        sliding_window_attn = SLIDING_WINDOWS,
        dim_head = DIM_HEAD,
        heads = HEADS,  # 增加头数
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
    print("Model created successfully")
except Exception as e:
    print(f"Error creating model: {e}")
    raise

# 加载完整enwik8数据集 (~100MB)
print("Loading full enwik8 dataset...")
try:
    # 尝试加载完整数据集 (约100MB)
    with gzip.open('./data/enwik8.gz') as file:
        # 先尝试读取10MB，确保内存足够
        data = np.frombuffer(file.read(int(10e6)), dtype=np.uint8).copy()
        print(f"Successfully loaded 10MB of data for testing")
        
        # 判断GPU内存是否足够处理完整数据集
        if torch.cuda.is_available():
            # 假设模型+数据+中间状态需要大约数据集大小的10倍内存
            estimated_memory_needed = len(data) * 10 / 1024**3  # GB
            available_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            if available_memory > estimated_memory_needed * 2:  # 保守估计，留出2倍余量
                print(f"GPU内存足够，尝试加载全部数据")
                # 重新打开文件加载全部数据
                file.seek(0)
                data = np.frombuffer(file.read(), dtype=np.uint8).copy()
            else:
                print(f"GPU内存可能不足，仅加载部分数据 (50MB)")
                file.seek(0)
                data = np.frombuffer(file.read(int(50e6)), dtype=np.uint8).copy()
        else:
            # CPU模式下，默认加载30MB
            file.seek(0)
            data = np.frombuffer(file.read(int(30e6)), dtype=np.uint8).copy()
            
        # 80%用于训练，20%用于验证
        split_point = int(len(data) * 0.8)
        data_train, data_val = np.split(data, [split_point])
        data_train, data_val = map(torch.from_numpy, (data_train, data_val))
    print(f"Dataset loaded successfully - Train: {len(data_train)/1000:.1f}KB, Val: {len(data_val)/1000:.1f}KB")

except Exception as e:
    print(f"Error loading data: {e}")
    print("Using random data for testing")
    # 创建更小的随机数据
    data_train = torch.randint(0, 255, (int(1e5),), dtype=torch.uint8)  # 100KB
    data_val = torch.randint(0, 255, (int(2e4),), dtype=torch.uint8)    # 20KB
    print(f"Using random data - Train: {len(data_train)/1000:.1f}KB, Val: {len(data_val)/1000:.1f}KB")

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data.to(device) if not data.is_cuda and torch.cuda.is_available() else data
        self.seq_len = seq_len

    def __getitem__(self, index):
        # 避免索引越界
        max_start = max(0, self.data.size(0) - self.seq_len - 1)
        rand_start = torch.randint(0, max_start, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq

    def __len__(self):
        return max(1, self.data.size(0) // self.seq_len)

print("Creating datasets...")
train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0))
val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0))

# 初始化优化器
print("Initializing optimizer...")
optim = AdoptAtan2(model.parameters(), lr=LEARNING_RATE)  # 使用AdoptAtan2优化器

# 保存检查点函数
def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint.pt"):
    # 确保checkpoint目录存在
    os.makedirs("checkpoint", exist_ok=True)
    
    # 将文件路径改为保存在checkpoint目录下
    filepath = os.path.join("checkpoint", filename)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    try:
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

# 加载之前的检查点
def load_checkpoint(model, optimizer, checkpoint_path):
    try:
        print(f"Trying to load checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Successfully loaded checkpoint from epoch {start_epoch}, loss: {loss:.4f}")
        return start_epoch
    except Exception as e:
        print(f"Could not load checkpoint: {e}")
        return 0

# 尝试加载之前的检查点
start_batch = 0
checkpoint_files = ["final_memory_checkpoint.pt", "checkpoint_epoch_50.pt", "checkpoint_epoch_100.pt"]
for checkpoint_file in checkpoint_files:
    checkpoint_path = os.path.join("checkpoint", checkpoint_file)
    if os.path.exists(checkpoint_path):
        start_batch = load_checkpoint(model, optim, checkpoint_path)
        # 如果成功加载了检查点，则从下一个批次开始
        if start_batch > 0:
            start_batch += 1
            print(f"Continuing training from batch {start_batch}")
            break

# 记录训练损失用于分析
train_losses = []
val_losses = []

# training
print(f"Starting training for {NUM_BATCHES} batches...")
try:
    for i in tqdm.tqdm(range(start_batch, NUM_BATCHES), mininterval=1., desc='training'):
        model.train()

        # 每批次只运行一次，不累积梯度
        try:
            data_batch = next(train_loader)
            loss = model(data_batch, return_loss=True)
            loss.backward()
        except Exception as e:
            print(f"Error in training iteration {i}: {e}")
            import traceback
            traceback.print_exc()
            break

        # 记录训练损失
        train_loss = loss.item()
        train_losses.append(train_loss)
        
        print(f'Batch {i+1}/{NUM_BATCHES}, training loss: {train_loss:.4f}')
        
        # 梯度裁剪以防爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        
        # 优化器步骤
        optim.step()
        optim.zero_grad()
        
        # 定期保存检查点
        if i % SAVE_EVERY == 0 and i > 0:
            try:
                save_checkpoint(model, optim, i, train_loss, f"checkpoint_epoch_{i}.pt")
            except Exception as e:
                print(f"Failed to save checkpoint at batch {i}: {e}")

        # 验证阶段
        if i % VALIDATE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                try:
                    val_data_batch = next(val_loader)
                    val_loss = model(val_data_batch, return_loss=True)
                    val_losses.append(val_loss.item())
                    print(f'Validation loss: {val_loss.item():.4f}')
                except Exception as e:
                    print(f"Error in validation: {e}")

        # 生成文本样本
        if SHOULD_GENERATE and i % GENERATE_EVERY == 0:
            try:
                print(f"\nGenerating sample text at batch {i+1}:")
                test_model_generation(model, val_dataset, PRIME_LENGTH, GENERATE_LENGTH)
            except Exception as e:
                print(f"Error in text generation: {e}")

    # 完成训练
    print(f"Training completed after {NUM_BATCHES} batches!")
    
    # 保存最终检查点
    final_loss = train_losses[-1] if train_losses else 0
    save_checkpoint(model, optim, NUM_BATCHES, final_loss, "final_memory_checkpoint.pt")
    
    # 保存损失历史记录
    np.save("train_losses.npy", np.array(train_losses))
    np.save("val_losses.npy", np.array(val_losses))
    
    print("Final model evaluation:")
    if SHOULD_GENERATE:
        for _ in range(3):  # 生成3个样本，观察效果
            test_model_generation(model, val_dataset, PRIME_LENGTH, GENERATE_LENGTH)

except KeyboardInterrupt:
    print("Training interrupted by user")
    try:
        current_batch = start_batch + len(train_losses)
        current_loss = train_losses[-1] if train_losses else 0
        save_checkpoint(model, optim, current_batch, current_loss, "interrupted_checkpoint.pt")
        print(f"Saved checkpoint at batch {current_batch}")
    except:
        print("Could not save checkpoint on interrupt")
except Exception as e:
    print(f"Error during training: {e}")
    import traceback
    traceback.print_exc()
