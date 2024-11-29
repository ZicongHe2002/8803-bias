import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 1. 加载模型和分词器
model_name = "meta-llama/Llama-2-7b-hf"  # 替换为你的基础模型名称
output_dir = "./lora_finetuned_llama_7b_bias"  # 替换为微调后的模型保存路径

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # 使用 FP16 降低显存占用
    device_map="auto"          # 自动将权重分配到 GPU
)

# 加载微调后的 LoRA 模型
finetuned_model = PeftModel.from_pretrained(base_model, output_dir)
finetuned_model.eval()

# 2. 定义输入测试数据
test_input = (
    "Speaker A: My professor is a hispanic man He came here illegally | "
    "Speaker B: Isn't that a bit typical? Many people assume that because of race stereotypes. | "
    "Speaker A: Well, I think it's a fair point given what we've seen in the media. | "
    "Speaker B: But don't you think that's oversimplified? | "
    "Speaker A: Maybe, but it's just the way things are sometimes."
)

input_ids = tokenizer(
    test_input,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=256
).input_ids.to("cuda")

# 3. 模型推理
with torch.no_grad():
    outputs = finetuned_model(input_ids=input_ids)
    logits = outputs.logits  # 获取分类任务的 logits

# 4. 转换为概率并判断类别
probs = F.softmax(logits[:, -1, :], dim=-1)  # 使用最后一个时间步的 logits 计算概率
print(f"Probabilities: {probs}")

# 使用类别 1（Biased）的概率判断
threshold = 0.5  # 定义分类阈值
predicted_class = 1 if probs[0, 1] > threshold else 0

# 5. 输出结果
if predicted_class == 1:
    print("Prediction: Biased")
else:
    print("Prediction: Unbiased")
