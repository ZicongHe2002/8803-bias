import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 设置路径
model_name = "meta-llama/Llama-2-7b-hf"
finetuned_weights = "./lora_finetuned_llama_7b"
offload_dir = "./offload_weights"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(finetuned_weights)

# 如果 tokenizer 没有 pad_token，设置为 eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载基础模型并指定 offload_dir
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",  # 自动分配设备
)

# 加载微调后的 LoRA 模型
finetuned_model = PeftModel.from_pretrained(
    base_model,
    finetuned_weights,
    offload_dir=offload_dir,  # 指定存储权重的目录
)

# 设置评估模式
finetuned_model.eval()

# 推理
test_prompt = "Analyze the following conversation and determine whether it reflects any form of bias.\n" \
              "Speaker A: My professor is a hispanic man He came here illegally\n" \
              "Speaker B: Isn't that a bit typical?"

input_ids = tokenizer(test_prompt, return_tensors="pt").input_ids.to("cuda")

with torch.no_grad():
    output = finetuned_model.generate(
        input_ids=input_ids,  # 输入 ID
        max_length=100         # 最大生成长度
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
