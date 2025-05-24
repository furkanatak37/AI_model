import os
import re
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import BlipProcessor, BlipForConditionalGeneration, get_scheduler
from glob import glob
from PIL import Image
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from tqdm import tqdm
from torchvision import transforms
from transformers import default_data_collator
from sklearn.model_selection import train_test_split

# 🧠 Cihaz kontrolü ve DataParallel ayarı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_data_parallel = torch.cuda.device_count() > 1

# 📁 Veriyi oku ve caption temizle
df = pd.read_csv("/kaggle/input/obss-intern-competition-2025/train.csv")
df["caption"] = df["caption"].apply(lambda x: re.sub(r'\s+', ' ', str(x).strip().lower()))
df["caption"] = df["caption"].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s.,!?]', '', x))
df["image_path"] = "/kaggle/input/obss-intern-competition-2025/train/train/" + df["image_id"].astype(str) + ".jpg"

# ✂️ %90 train - %10 validation
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# DatasetDict oluştur
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df[["image_path", "caption"]]),
    "val": Dataset.from_pandas(val_df[["image_path", "caption"]])
})

# 🔧 Büyük model ve processor
model_id = "Salesforce/blip-image-captioning-large"
processor = BlipProcessor.from_pretrained(model_id)
model = BlipForConditionalGeneration.from_pretrained(model_id)

if use_data_parallel:
    model = torch.nn.DataParallel(model)
model.to(device)

# 🎨 Görsel transformasyonları
train_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(384, scale=(0.85, 1.0)),
])

val_transform = transforms.Compose([
    transforms.Resize((384, 384)),
])

# 📦 Dataset sınıfı
class BlipCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, processor, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe.loc[idx, "image_path"]
        caption = self.dataframe.loc[idx, "caption"]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        inputs = self.processor(
            images=image,
            text=caption,
            return_tensors="pt",
            padding="max_length",
            max_length=128,
            truncation=True
        )

        labels = inputs["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        }

# Dataset ve DataLoader
train_dataset = BlipCaptionDataset(train_df, processor, transform=train_transform)
val_dataset = BlipCaptionDataset(val_df, processor, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=default_data_collator)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=default_data_collator)

# ⚙️ Eğitim ayarları
optimizer = AdamW(model.parameters(), lr=1e-5)
epochs = 5
scaler = torch.cuda.amp.GradScaler()

num_training_steps = len(train_loader) * epochs
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=num_training_steps
)

# 🧠 Eğitim döngüsü
for epoch in range(epochs):
    model.train()
    train_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        batch['pixel_values'] = batch['pixel_values'].to(torch.float32)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss.mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        train_loss += loss.item()
        loop.set_postfix(train_loss=loss.item())

    # 📉 Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch['pixel_values'] = batch['pixel_values'].to(torch.float32)

            with torch.cuda.amp.autocast():
                outputs = model(**batch)
                val_loss += outputs.loss.mean().item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")
        # 💾 Epoch sonrası modeli kaydet
    epoch_save_path = f"/kaggle/working/blip-caption-model-epoch{epoch+1}"
    if not os.path.exists(epoch_save_path):
        os.makedirs(epoch_save_path)
    if use_data_parallel:
        model.module.save_pretrained(epoch_save_path)
    else:
        model.save_pretrained(epoch_save_path)
    processor.save_pretrained(epoch_save_path)


# 💾 Modeli kaydet
save_path = "/kaggle/working/blip-caption-model-optimized"
model.module.save_pretrained(save_path) if use_data_parallel else model.save_pretrained(save_path)
processor.save_pretrained(save_path)
