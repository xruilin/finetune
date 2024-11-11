import os
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig
from datasets import load_dataset
import bitsandbytes as bnb


# 解析字幕文件，假设数据已经存储在 json 文件中
def load_data(dataset_path):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 将数据转换为 Hugging Face Dataset 格式
    dataset = [{"instruction": item["instruction"], "output": item["output"]} for item in data]
    return Dataset.from_list(dataset)


# 数据预处理，tokenize 输入和输出文本
def preprocess_function(examples, tokenizer, max_length=128):
    inputs = tokenizer(examples['instruction'], padding="max_length", truncation=True, max_length=max_length)
    outputs = tokenizer(examples['output'], padding="max_length", truncation=True, max_length=max_length)
    
    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'labels': outputs['input_ids']  # 使用 output 的 input_ids 作为标签
    }


# 划分训练集、验证集和测试集
def split_data(dataset):
    # 划分 80% 训练集, 10% 验证集, 10% 测试集
    train_val_test_split = dataset.train_test_split(test_size=0.2)  # 80% 训练集, 20% 测试集
    train_val = train_val_test_split["train"]
    test_dataset = train_val_test_split["test"]

    # 在训练集上再进行划分，分出 10% 用作验证集
    train_dataset = train_val.train_test_split(test_size=0.1)["train"]  # 90% 训练集
    val_dataset = train_val.train_test_split(test_size=0.1)["test"]    # 10% 验证集

    return train_dataset, val_dataset, test_dataset


# 设置训练过程的参数
def train(model_name, dataset, output_dir, batch_size=4, num_epochs=3, learning_rate=5e-5):
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
    # 设置 pad_token 为 eos_token
    tokenizer.pad_token = tokenizer.eos_token
    # 划分数据集
    train_dataset, val_dataset, test_dataset = split_data(dataset)

    # 预处理数据集
    train_dataset = train_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,             # 保存模型和日志的路径
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=20,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=1000,
        evaluation_strategy="no",
        fp16=True,
        save_total_limit=2,
        load_best_model_at_end=False,
        remove_unused_columns=False
    )

    # 配置LoRA微调
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # LoRA 会修改哪些模型层
        lora_dropout=0.1,
        bias="none"
    )
    model = get_peft_model(model, lora_config)

    # 使用 Trainer 进行训练
    trainer = Trainer(
        model=model,                       # 训练的模型
        args=training_args,                # 训练参数
        train_dataset=train_dataset,       # 训练数据集
        eval_dataset=val_dataset,          # 验证数据集
    )

    # 开始训练
    trainer.train()

    # 保存训练后的模型
    trainer.save_model(output_dir)  # 保存微调后的模型
    tokenizer.save_pretrained(output_dir)  # 保存分词器
    print(f"训练完成，模型已保存至 {output_dir}")

    # 评估模型
    results = trainer.evaluate(test_dataset)
    print(f"测试集评估结果: {results}")


if __name__ == "__main__":

    # 数据集路径
    dataset_path = "/home/Llama7B/data/dataset.json"  # 你的数据集路径
    output_dir = "./fine_tuned_model5"  # 微调后的模型保存路径
    model_name = "huggyllama/llama-7b"  # 预训练的模型路径，替换为你的路径

    # 加载数据集
    dataset = load_data(dataset_path)

    # 开始训练
    train(model_name, dataset, output_dir)
