import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

# 配置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 基础模型和分词器的路径
model_name = "huggyllama/llama-7b"  # 基础LLaMA模型的名称（可以是 Hugging Face 模型库中的名称）
model_path = "/home/Llama7B/fine_tuned_model5"  # LoRA适配器权重所在路径

# 配置 BitsAndBytes (量化)
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # 加载 8-bit 权重
    llm_int8_enable_fp32_cpu_offload=True  # 启用 32-bit CPU offload
)

# 加载基础模型和分词器（包含量化配置）
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto", 
    quantization_config=bnb_config  # 使用 BitsAndBytes 配置
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载 LoRA 配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none"
)

# 应用 LoRA 到模型
model = get_peft_model(model, lora_config)

# 使用 safetensors 加载 LoRA 适配器权重（加载到 CPU）
adapter_state_dict = load_file(f"{model_path}/adapter_model.safetensors", device="cpu")

# 将加载的权重转移到 GPU
model.load_state_dict(adapter_state_dict, strict=False)  # 加载 LoRA 权重

# 将模型移到 GPU
model.to(device)

# 设置模型为评估模式
model.eval()

# 测试输入
test_input = "Major General Zhang Taiwan, Commander of the Capital Garrison"

# 对测试输入进行编码
inputs = tokenizer(test_input, return_tensors="pt").to(device)

# 生成文本
with torch.no_grad():
    output = model.generate(
        inputs["input_ids"],
        max_length=100,  # 设置生成的最大长度
        num_beams=5,  # 设置束搜索的宽度
        no_repeat_ngram_size=2,  # 设置避免重复的n-gram大小
        temperature=1.0, # 设置温度
        top_k=50,         # 从 top 50 的词汇中随机选择下一个词
        top_p=0.9,        # 使用 90% 累积概率的词汇
        num_return_sequences=3,
        do_sample=True
    )

# 解码生成的结果

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# 输出生成的文本
print("Generated Text:", generated_text)

# 循环生成10次不同的结果
for i in range(10):
    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_length=150,
            num_beams=5,
            no_repeat_ngram_size=2,
            temperature=1.0,
            top_k=50,
            top_p=0.9,
            num_return_sequences=1,
            do_sample=True
        )

    # 解码生成的结果
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # 输出生成的文本
    print(f"Generated Text {i+1}: {generated_text}")