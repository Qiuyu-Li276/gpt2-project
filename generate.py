import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    pipeline,
)
print("库安装和导入完成！")

# --- 修改点 1: 指定最终模型的路径 ---
final_model_path = "./gpt2-sports-finetuned-final"

print(f"从 '{final_model_path}' 加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(final_model_path)


print("\n--- 步骤 7: 生成文本 ---")
# ↓↓↓ 在这里输入你的 prompt (提示) ↓↓↓
prompt = "在今天凌晨结束的一场关键的篮球比赛中，"
# ↑↑↑ 在这里输入你的 prompt (提示) ↑↑↑

# 加载我们微调好的模型和分词器
# pipeline 会自动从 final_model_path 加载模型
print("正在创建文本生成 pipeline...")
generator = pipeline('text-generation', model=final_model_path, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# 生成文本
print("正在生成文本...")
generated_texts = generator(
    prompt,
    max_new_tokens=800,
    num_return_sequences=1,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.9,
    repetition_penalty=1.2,
)

# 打印并后处理生成的文本
print("\n\n--- 生成的新闻稿 ---")
full_text = generated_texts[0]['generated_text']
sentences = [s.strip() for s in full_text.replace(" ", "").split('。') if s.strip()]
final_output = '。'.join(sentences) + '。' if sentences else ""
print(final_output)

# 分析生成结果
num_sentences = len(sentences)
print(f"\n--- 分析 ---")
print(f"生成的文本包含大约 {num_sentences} 句话。")
if num_sentences >= 10:
    print("成功满足不少于十句话的要求！")
else:
    print("未达到十句话的要求，可以尝试增加 max_new_tokens 参数或调整 prompt。")