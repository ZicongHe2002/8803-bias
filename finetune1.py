import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model
import json

# 1. 自定义数据集
class BiasDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        with open(file_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item["instruction"]
        input_text = item["input"]
        output_text = item["output"]
        full_text = f"Instruction: {instruction} Input: {input_text} Response: {output_text}"

        tokenized = self.tokenizer(
            full_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        return tokenized.input_ids.squeeze(0), tokenized.attention_mask.squeeze(0)


# 2. 加载模型和分词器
model_name = "meta-llama/Llama-2-7b-hf"  # 替换为你的模型名称
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 如果 tokenizer 没有 pad_token，添加它
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载模型到 GPU
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # 使用 FP16 降低显存占用
    device_map="auto"          # 自动将权重分配到 GPU
)


# 3. 配置 LoRA
config = LoraConfig(
    r=8,                        # LoRA 低秩矩阵的秩
    lora_alpha=16,              # LoRA 的缩放因子
    target_modules=["q_proj", "v_proj"],  # 目标层
    lora_dropout=0.1,           # Dropout 比例
    bias="none",                # 不调整 bias
    task_type="CAUSAL_LM"       # 因果语言建模任务
)

# 包装 LoRA 模型
lora_model = get_peft_model(base_model, config)

# 确保 LoRA 参数移动到 GPU
lora_model.to("cuda")

# 打印可训练参数数量
trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
print(f"Trainable Parameters: {trainable_params}")

# 4. 数据集加载
dataset_path = "bias.json"  # 替换为你的 JSON 文件路径
train_dataset = BiasDataset(dataset_path, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# 5. 优化器
optimizer = AdamW(filter(lambda p: p.requires_grad, lora_model.parameters()), lr=1e-4)

# 6. 微调
lora_model.train()
gradient_accumulation_steps = 4  # 每4步累积一次梯度
effective_batch_size = 2 * gradient_accumulation_steps  # 等效于 batch_size=8

for epoch in range(4):  # 假设训练 4 个 epoch
    print(f"Starting Epoch {epoch + 1}...")  # 提示开始当前 epoch
    total_loss = 0

    for batch_idx, batch in enumerate(train_dataloader):
        input_ids, attention_mask = batch
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")

        # 前向传播
        outputs = lora_model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss / gradient_accumulation_steps  # 累积损失
        total_loss += loss.item()

        # 反向传播
        loss.backward()

        # 每累积一定步数后更新参数
        if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
            optimizer.step()
            optimizer.zero_grad()

        # 可选：打印进度
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_dataloader)}, Current Loss: {loss.item():.4f}")

    # 每个 epoch 结束时打印平均损失
    average_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1} completed. Average Loss: {average_loss:.4f}")


# 训练完成时提示
print("Training complete! All epochs have been processed.")

# 7. 保存微调后的模型
output_dir = "./lora_finetuned_llama_7b"
lora_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# 8. 测试模型
finetuned_model = AutoModelForCausalLM.from_pretrained(output_dir)
finetuned_model.eval()

test_prompt = "Analyze the following conversation and determine whether it reflects any form of bias.\n" \
              "Speaker A: My professor is a hispanic man He came here illegally\n" \
              "Speaker B: Isn't that a bit typical?"
input_ids = tokenizer(test_prompt, return_tensors="pt").input_ids.to("cuda")
output = finetuned_model.generate(input_ids, max_length=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
