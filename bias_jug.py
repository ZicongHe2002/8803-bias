import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model
import json
from torch.nn import CrossEntropyLoss
# 1. 自定义数据集
class BiasDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=256):
        with open(file_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item["input"]  # 对话内容
        label_text = item["output"].split(".")[0]  # 提取 Biased 或 Unbiased
        label = 1 if label_text == "Biased" else 0  # 转换为数值标签

        tokenized = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        return tokenized.input_ids.squeeze(0), tokenized.attention_mask.squeeze(0), label

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
    task_type="SEQUENCE_CLASSIFICATION"  # 分类任务
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
loss_fn = CrossEntropyLoss()  # 使用交叉熵损失

for epoch in range(3):  # 假设训练 4 个 epoch
    print(f"Starting Epoch {epoch + 1}...")
    total_loss = 0

    for batch_idx, batch in enumerate(train_dataloader):
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")
        labels = labels.to("cuda")  # 形状 [batch_size]

        # 前向传播
        outputs = lora_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # 模型输出 logits, 形状: [batch_size, sequence_length, vocab_size]

        # 取最后一个时间步的 logits
        logits = logits[:, -1, :]  # 形状变为 [batch_size, vocab_size]
        loss = loss_fn(logits, labels)  # 计算交叉熵损失

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1} completed. Average Loss: {total_loss / len(train_dataloader):.4f}")

print("Training complete! All epochs have been processed.")

# 训练完成时提示
print("Training complete! All epochs have been processed.")

# 7. 保存微调后的模型
output_dir = "./lora_finetuned_llama_7b_bias"
lora_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# 8. 测试模型
finetuned_model = AutoModelForCausalLM.from_pretrained(output_dir)
finetuned_model.eval()

test_prompt = "Speaker A: My professor is a hispanic man He came here illegally | Speaker B: Isn't that a bit typical? Many people assume that because of race stereotypes."
input_ids = tokenizer(test_prompt, return_tensors="pt", padding=True, truncation=True).input_ids.to("cuda")

outputs = finetuned_model(input_ids=input_ids)
logits = outputs.logits  # 分类输出
predicted_class = torch.argmax(logits, dim=-1).item()  # 获取预测类别

if predicted_class == 1:
    print("Prediction: Biased")
else:
    print("Prediction: Unbiased")

