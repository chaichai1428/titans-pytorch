import torch
import os
import random
import glob

# 禁用PyTorch动态编译和JIT
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch.jit.enable = False

# 禁用动态优化
os.environ["PYTORCH_JIT"] = "0"
os.environ["TORCH_COMPILE"] = "0"
os.environ["PYTORCH_NO_CUDA_MEMORY_EFFICIENCY_WARNING"] = "1"

# 使用CUDA设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 显示CUDA信息
if torch.cuda.is_available():
    print(f"CUDA设备数量: {torch.cuda.device_count()}")
    print(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")
    print(f"当前CUDA设备: {torch.cuda.current_device()}")
    print(f"CUDA可用内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 从titans_pytorch加载必要的模型
from titans_pytorch import (
    MemoryAsContextTransformer,
    MemoryAttention
)

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

def encode_text(text):
    """将文本编码为张量，并移动到适当设备"""
    return torch.tensor([ord(c) for c in text], dtype=torch.long).to(device)

def find_latest_checkpoint():
    """查找最新的检查点文件"""
    # 首先检查checkpoint目录
    checkpoint_dir = "checkpoint"
    if os.path.exists(checkpoint_dir) and os.path.isdir(checkpoint_dir):
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
        if checkpoint_files:
            # 尝试按照文件名中的数字排序
            try:
                checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if '_' in x else 0, reverse=True)
            except:
                # 如果排序失败就按照修改时间排序
                checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # 优先使用final_memory_checkpoint.pt
            for cf in checkpoint_files:
                if "final_memory_checkpoint.pt" in cf:
                    return cf
            
            # 否则使用最新的检查点
            return checkpoint_files[0]
    
    # 如果checkpoint目录中没有找到，则检查当前目录
    checkpoint_files = glob.glob("*.pt")
    if not checkpoint_files:
        print("错误: 未找到检查点文件!")
        return None
    
    # 尝试按照文件名中的数字排序
    try:
        checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if '_' in x else 0, reverse=True)
    except:
        # 如果排序失败就按照修改时间排序
        checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return checkpoint_files[0]

def load_model(checkpoint_path):
    """加载模型和检查点"""
    print(f"加载检查点: {checkpoint_path}")
    
    # 模型参数 - 与train_windows.py一致，使用增强版参数
    DIM = 128
    DIM_HEAD = 64
    HEADS = 4
    DEPTH = 4
    NUM_PERSIST_MEM = 6
    NUM_LONGTERM_MEM = 8
    WINDOW_SIZE = 32
    MEM_SEGMENT_LEN = 16
    MEM_BATCH_SIZE = 32
    NEURAL_MEM_LAYERS = (1, 2, 3)
    
    # 创建记忆模型
    neural_memory_model = MemoryAttention(dim=DIM)
    
    # 创建模型实例
    model = MemoryAsContextTransformer(
        num_tokens = 256,
        dim = DIM,
        depth = DEPTH,
        segment_len = WINDOW_SIZE,
        num_persist_mem_tokens = NUM_PERSIST_MEM,
        num_longterm_mem_tokens = NUM_LONGTERM_MEM,
        neural_memory_layers = NEURAL_MEM_LAYERS,
        neural_memory_segment_len = MEM_SEGMENT_LEN,
        neural_memory_batch_size = MEM_BATCH_SIZE,
        neural_mem_gate_attn_output = True,
        neural_mem_weight_residual = True,
        neural_memory_qkv_receives_diff_views = False,
        use_flex_attn = False,
        sliding_window_attn = False,
        dim_head = DIM_HEAD,
        heads = HEADS,
        neural_memory_model = neural_memory_model
    ).to(device)
    
    # 加载检查点
    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"成功加载批次: {checkpoint['epoch']} 的检查点，损失: {checkpoint['loss']:.4f}")
        print("(使用strict=False参数加载，可能有部分参数不匹配)")
        model.eval()  # 设置为评估模式
        return model
    except Exception as e:
        print(f"加载检查点失败: {e}")
        return None

def display_test_header(test_name, description=""):
    """显示测试标题"""
    print("\n" + "="*60)
    print(f"测试: {test_name}")
    if description:
        print(f"{description}")
    print("="*60)

def test_text_completion(model, prompt, max_length=100):
    """简单文本补全测试"""
    display_test_header("简单文本补全", "评估模型生成流畅文本的能力")
    
    print(f"提示文本: '{prompt}'")
    
    # 编码并生成
    inp = encode_text(prompt)  # 已在encode_text函数中移动到设备
    with torch.no_grad():
        output = model.sample(inp[None, ...], max_length)
    
    # 解码并显示
    generated_text = decode_tokens(output[0].cpu())  # 移回CPU解码
    
    print("\n生成结果:")
    print("-"*60)
    print(generated_text)
    print("-"*60)
    
    # 简单评估
    print(f"生成长度: {len(generated_text)} 字符")
    print(f"新生成: {len(generated_text) - len(prompt)} 字符")

def test_pattern_continuation(model, pattern="ABCABC", repetitions=3, max_length=100):
    """模式延续测试 - 评估短期记忆能力"""
    display_test_header("模式延续", "评估模型识别和延续简单模式的能力")
    
    # 创建重复模式作为输入
    input_text = pattern * repetitions
    print(f"输入模式: '{pattern}' (重复 {repetitions} 次)")
    print(f"完整输入: '{input_text}'")
    
    # 编码并生成
    inp = encode_text(input_text)
    with torch.no_grad():
        output = model.sample(inp[None, ...], max_length)
    
    # 解码并显示
    full_output = decode_tokens(output[0].cpu())
    continuation = full_output[len(input_text):]
    
    print("\n生成的延续:")
    print("-"*60)
    print(continuation)
    print("-"*60)
    
    # 检查延续中的模式
    pattern_count = 0
    for i in range(0, len(continuation) - len(pattern) + 1):
        if continuation[i:i+len(pattern)] == pattern:
            pattern_count += 1
    
    print(f"模式在延续中出现: {pattern_count} 次")
    
    # 评估记忆能力
    if pattern_count >= 2:
        print("记忆能力评价: 出色 - 模型成功识别并多次延续了模式")
    elif pattern_count == 1:
        print("记忆能力评价: 良好 - 模型能识别并至少延续一次模式")
    else:
        print("记忆能力评价: 有限 - 模型未能延续模式")

def test_key_word_recall(model, max_length=100):
    """关键词回忆测试 - 评估长距离记忆能力"""
    display_test_header("关键词回忆", "评估模型记住和回忆关键信息的能力")
    
    # 创建测试文本
    key_phrase = "ZOMBOID SURVIVAL GUIDE"
    prefix = f"IMPORTANT MESSAGE: {key_phrase}. "
    middle = "This text is just a filler and should be ignored. " * 3
    suffix = "NOW PLEASE RECALL THE IMPORTANT MESSAGE: "
    
    input_text = prefix + middle + suffix
    
    print(f"关键短语: '{key_phrase}'")
    print(f"输入总长度: {len(input_text)} 字符")
    print(f"关键短语与回忆提示之间的距离: {len(middle)} 字符")
    
    # 编码并生成
    inp = encode_text(input_text)
    with torch.no_grad():
        output = model.sample(inp[None, ...], max_length)
    
    # 解码并显示
    full_output = decode_tokens(output[0].cpu())
    continuation = full_output[len(input_text):]
    
    print("\n生成的回忆:")
    print("-"*60)
    print(continuation)
    print("-"*60)
    
    # 检查关键词的回忆
    keywords = key_phrase.split()
    found_keywords = [word for word in keywords if word in continuation]
    
    print(f"成功回忆的关键词: {found_keywords if found_keywords else '无'}")
    recall_rate = len(found_keywords) / len(keywords) * 100
    print(f"关键词回忆率: {recall_rate:.1f}%")
    
    # 评估记忆能力
    if recall_rate >= 70:
        print("长距离记忆能力评价: 出色 - 模型能够回忆大部分关键词")
    elif recall_rate >= 30:
        print("长距离记忆能力评价: 良好 - 模型能够回忆部分关键词")
    else:
        print("长距离记忆能力评价: 有限 - 模型回忆关键词的能力较弱")

def test_window_memory(model, max_length=50):
    """窗口记忆测试 - 评估模型对前后文的关注能力"""
    display_test_header("窗口记忆测试", "评估模型对长序列前后文的关注能力")
    
    # 创建测试序列: 前部, 中部和后部有明显区别
    early_marker = "EARLY_DATA_12345"
    middle_filler = "X" * 30  # 填充中间部分
    recent_marker = "RECENT_DATA_67890"
    
    input_text = early_marker + middle_filler + recent_marker + " CONTINUE: "
    
    print(f"早期信息: '{early_marker}'")
    print(f"填充长度: {len(middle_filler)} 字符")
    print(f"近期信息: '{recent_marker}'")
    
    # 编码并生成
    inp = encode_text(input_text)
    with torch.no_grad():
        output = model.sample(inp[None, ...], max_length)
    
    # 解码并显示
    full_output = decode_tokens(output[0].cpu())
    continuation = full_output[len(input_text):]
    
    print("\n生成的延续:")
    print("-"*60)
    print(continuation)
    print("-"*60)
    
    # 分析生成文本中包含的标记
    early_in_output = early_marker in continuation or early_marker[:5] in continuation
    recent_in_output = recent_marker in continuation or recent_marker[:5] in continuation
    
    # 打印分析结果
    print(f"包含早期信息: {'是' if early_in_output else '否'}")
    print(f"包含近期信息: {'是' if recent_in_output else '否'}")
    
    # 评估窗口记忆能力
    if early_in_output and recent_in_output:
        print("窗口记忆评价: 出色 - 模型能够同时记住远距离和近距离的信息")
    elif recent_in_output:
        print("窗口记忆评价: 符合预期 - 模型更关注近期信息")
    elif early_in_output:
        print("窗口记忆评价: 意外 - 模型更关注早期信息而非近期信息")
    else:
        print("窗口记忆评价: 有限 - 模型无法有效回忆窗口内的信息")

def test_custom_input(model, prompt, max_length=100):
    """自定义输入测试"""
    display_test_header("自定义输入测试", "使用您提供的文本测试模型")
    
    print(f"输入文本: '{prompt}'")
    
    # 编码并生成
    inp = encode_text(prompt)
    with torch.no_grad():
        output = model.sample(inp[None, ...], max_length)
    
    # 解码并显示
    generated_text = decode_tokens(output[0].cpu())
    
    print("\n完整输出:")
    print("-"*60)
    print(generated_text)
    print("-"*60)
    
    # 分离并显示新生成的部分
    new_text = generated_text[len(prompt):]
    print("\n新生成部分:")
    print("-"*60)
    print(new_text)
    print("-"*60)

def main():
    print("\n" + "="*60)
    print("内存增强Transformer模型性能测试")
    print("="*60)
    
    # 查找最新的检查点
    checkpoint_path = find_latest_checkpoint()
    if not checkpoint_path:
        print("无法继续测试: 未找到检查点文件")
        return
    
    # 加载模型
    model = load_model(checkpoint_path)
    if not model:
        print("无法继续测试: 模型加载失败")
        return
    
    # 执行一系列测试
    
    # 1. 简单文本补全测试
    test_text_completion(model, "The quick brown fox jumps over ")
    
    # 2. 模式延续测试
    test_pattern_continuation(model, pattern="123 ", repetitions=4)
    
    # 3. 关键词回忆测试
    test_key_word_recall(model)
    
    # 4. 窗口记忆测试
    test_window_memory(model)
    
    # 5. 自定义输入测试
    custom_prompt = "In the world of Project Zomboid, survivors must "
    test_custom_input(model, custom_prompt)
    
    print("\n" + "="*60)
    print("性能测试完成!")
    print("="*60)

if __name__ == "__main__":
    main() 