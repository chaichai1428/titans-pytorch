import torch
import os
import json
import argparse

# 禁用PyTorch动态编译和JIT
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch.jit.enable = False

# 禁用动态优化
os.environ["PYTORCH_JIT"] = "0"
os.environ["TORCH_COMPILE"] = "0"
os.environ["PYTORCH_NO_CUDA_MEMORY_EFFICIENCY_WARNING"] = "1"

from titans_pytorch import (
    MemoryAsContextTransformer,
    MemoryAttention
)

# 特殊标记
INSTRUCTION_TOKEN = 1  # 指令标记的ID
ANSWER_TOKEN = 2      # 回答标记的ID
SEP_TOKEN = 3         # 分隔符标记的ID
PAD_TOKEN = 0         # 填充标记的ID

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
    """将文本编码为token"""
    return torch.tensor([ord(c) for c in text], dtype=torch.long)

def test_model_generation(model, instruction, device, max_length=256, input_text="", temperature=0.7):
    """测试模型对指令的回答生成能力"""
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
    print(f'\n指令: {instruction}')
    if input_text:
        print(f'输入: {input_text}')
    print('='*80)
    
    # 生成回答
    with torch.no_grad():
        try:
            output = model.sample(
                input_seq[None, ...],
                max_length,
                temperature=temperature,
                use_cache=True
            )
        except Exception as e:
            print(f"生成时出错，尝试不同参数: {e}")
            output = model.sample(
                input_seq[None, ...],
                max_length,
                use_cache=False
            )
    
    # 解码并显示
    output_str = decode_tokens(output[0].cpu())
    answer_part = output_str.split('[ANS]')[-1]  # 获取回答部分
    
    print(f'生成回答:')
    print('-'*80)
    print(answer_part)
    print('-'*80)
    
    return answer_part

def create_model(device):
    """创建模型实例"""
    # 模型参数 - 必须与训练时相匹配
    DIM = 256
    DIM_HEAD = 64
    HEADS = 8
    DEPTH = 6
    NUM_PERSIST_MEM = 12
    NUM_LONGTERM_MEM = 16
    WINDOW_SIZE = 64
    MEM_SEGMENT_LEN = 32
    MEM_BATCH_SIZE = 64
    NEURAL_MEM_LAYERS = (1, 2, 3, 4)
    
    # 创建记忆模型
    neural_memory_model = MemoryAttention(dim=DIM)
    
    # 创建模型实例
    model = MemoryAsContextTransformer(
        num_tokens = 256,  # 基本ASCII + 特殊标记
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
    
    return model

def load_checkpoint(model, checkpoint_path):
    """加载检查点"""
    try:
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        epoch = checkpoint.get('epoch', 'unknown')
        loss = checkpoint.get('loss', 'unknown')
        print(f"Successfully loaded checkpoint from epoch {epoch}, loss: {loss}")
        return True
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return False

def load_test_questions(questions_file=None):
    """加载测试问题"""
    default_questions = [
        "What critical survival action should the agent take immediately upon hearing an approaching helicopter while outdoors?",
        "How can the agent gain advance warning of the helicopter event during Days 6-9, including the specific tool and frequency to monitor?",
        "What specific steps must the agent follow to obtain gasoline from a gas station pump after the main power grid has shut off?",
        "What type of location is a house heavily barricaded with wooden planks from the outside likely to be, and what tools are needed to remove the external barricades?",
        "What potential delayed consequence might occur after the agent sustains a scratch from a zombie, and if the \"Sick\" moodle appears afterward, what is the most likely cause and outcome?"
    ]
    
    if questions_file and os.path.exists(questions_file):
        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            print(f"Loaded {len(questions)} questions from {questions_file}")
            return questions
        except Exception as e:
            print(f"Error loading questions from file: {e}")
    
    return default_questions

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Test the instruction model")
    parser.add_argument("--checkpoint", type=str, default="checkpoint/instruction_best.pt", help="Path to the checkpoint file")
    parser.add_argument("--questions", type=str, default=None, help="Path to a JSON file containing test questions")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling (0.0-1.0)")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum length of generated text")
    parser.add_argument("--cpu", action="store_true", help="Force using CPU even if CUDA is available")
    args = parser.parse_args()
    
    # 设置设备
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Device Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 创建模型
    model = create_model(device)
    
    # 加载检查点
    checkpoint_path = args.checkpoint
    if not load_checkpoint(model, checkpoint_path):
        print("Exiting due to checkpoint loading failure")
        return
    
    # 设置模型为评估模式
    model.eval()
    
    # 加载测试问题
    questions = load_test_questions(args.questions)
    
    # 测试生成
    print("\n" + "="*80)
    print("Starting text generation tests")
    print("="*80)
    
    for i, question in enumerate(questions, 1):
        print(f"\nTest {i}/{len(questions)}")
        test_model_generation(
            model, 
            question, 
            device, 
            max_length=args.max_length, 
            temperature=args.temperature
        )

if __name__ == "__main__":
    main() 