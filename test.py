import os
import pandas as pd
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from tqdm import tqdm

# ✅ Modeli yükle
model_path = "/kaggle/working/blip-caption-model-optimized"  # daha önce kaydettiğin yol
processor = BlipProcessor.from_pretrained(model_path)
model = BlipForConditionalGeneration.from_pretrained(model_path)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ✅ Görsel klasörü
image_folder = "/kaggle/input/obss-intern-competition-2025/test/test"  # test görsellerin buradaysa
image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith(".jpg")]

# ✅ Caption'ları üret
results = []
for image_path in tqdm(image_paths):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs)
    
    caption = processor.tokenizer.decode(output[0], skip_special_tokens=True)
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    results.append({"image_id": image_id, "caption": caption})

# ✅ CSV olarak kaydet
df = pd.DataFrame(results)
df.to_csv("/kaggle/working/captions5.csv", index=False)
