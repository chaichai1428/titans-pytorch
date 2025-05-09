import json
import os
import re

def extract_pz_questions(input_file="data/wiki3.jsonl", output_file="data/pz_test_questions.json"):
    """
    从wiki3.jsonl中提取Project Zomboid相关问题，并转换为便于测试的格式
    """
    if not os.path.exists(input_file):
        print(f"输入文件不存在: {input_file}")
        return
    
    print(f"正在从 {input_file} 提取PZ问题...")
    
    # 存储提取的问题
    questions = []
    pz_question_count = 0
    total_count = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        total_count += 1
                        item = json.loads(line)
                        instruction = item.get('instruction', '')
                        
                        # 检查是否为PZ相关问题
                        if '<pz>' in instruction:
                            # 提取问题，移除<pz>标签
                            clean_question = re.sub(r'<pz>(.*?)</pz>', r'\1', instruction).strip()
                            questions.append(clean_question)
                            pz_question_count += 1
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    print(f"处理完成! 总共找到 {pz_question_count} 个PZ问题, 共 {total_count} 条数据.")
    
    # 创建包含原始问题和转换后问题的测试集
    test_questions = questions
    
    # 保存测试问题
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(test_questions, f, ensure_ascii=False, indent=2)
        print(f"测试问题已保存到: {output_file}")
    except Exception as e:
        print(f"保存文件时出错: {e}")
        return
    
    return questions

if __name__ == "__main__":
    # 默认使用data/wiki3.jsonl作为输入
    questions = extract_pz_questions()
    
    if questions:
        print("\n示例问题:")
        for i, q in enumerate(questions[:5], 1):
            print(f"{i}. {q}") 