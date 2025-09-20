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

print("\n--- 步骤 2: 加载数据集 ---")
json_file_path = 'train_sports_cleaned.json'

try:
    with open(json_file_path, 'r', encoding='utf-8') as f:
        # 假设 train.json 是一个字符串列表
        your_data = json.load(f)
    
    # 检查数据格式是否正确
    if not isinstance(your_data, list) or not all(isinstance(item, str) for item in your_data):
        raise TypeError("JSON文件内容必须是一个字符串列表 (a list of strings)。")

    # 将数据列表转换成 Hugging Face 的 Dataset 对象
    dataset = Dataset.from_dict({"text": your_data})
    print(f"成功从 '{json_file_path}' 加载了 {len(dataset)} 篇文章。")
    print("数据集预览:")
    print(dataset[0])

except FileNotFoundError:
    print(f"错误: 文件 '{json_file_path}' 未找到。请确保您已成功上传该文件到 Colab 环境中。")
    # 为了让代码能继续运行以展示流程，这里创建一个虚拟数据集
    dataset = Dataset.from_dict({"text": ["这是一个示例，因为 train.json 文件未找到。"]})
except (json.JSONDecodeError, TypeError) as e:
    print(f"错误: 加载或解析JSON文件时出错。请检查文件格式。错误信息: {e}")
    dataset = Dataset.from_dict({"text": ["这是一个示例，因为 train.json 文件格式错误。"]})


# --------------------------
# 步骤 3: 加载模型和分词器
# --------------------------
print("\n--- 步骤 3: 加载预训练模型和分词器 ---")
model_name = "uer/gpt2-chinese-cluecorpussmall"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 加载预训练模型
model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))
print(f"模型 '{model_name}' 加载完成。")


# --------------------------
# 步骤 4: 数据预处理
# --------------------------
print("\n--- 步骤 4: 数据预处理 ---")
# 分词函数
def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 文本块处理函数
block_size = 128
def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_dataset = tokenized_dataset.map(group_texts, batched=True, batch_size=1000)
print(f"数据处理完成，生成 {len(lm_dataset)} 个训练样本。")


# --------------------------
# 步骤 5: 配置与训练
# --------------------------
print("\n--- 步骤 5: 配置训练参数并开始训练 ---")
training_args = TrainingArguments(
    output_dir="./gpt2-sports-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=10,  # 训练轮次，可根据数据集大小调整
    per_device_train_batch_size=160, # 根据GPU显存调整
    save_steps=10_000, # 由于数据集可能不大，我们将只在最后保存
    save_total_limit=2,
    logging_steps=50,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset,
    tokenizer=tokenizer,
)

# 开始训练
trainer.train()
print("模型微调完成！")


# --------------------------
# 步骤 6: 保存最终模型
# --------------------------
print("\n--- 步骤 6: 保存最终模型 ---")
final_model_path = "./gpt2-sports-finetuned-final"
trainer.save_model(final_model_path)
print(f"最终模型已保存至: {final_model_path}")