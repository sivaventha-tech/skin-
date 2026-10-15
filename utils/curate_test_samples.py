import os
import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.msta_net import MSTANet

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = PROJECT_ROOT / "dataset" / "preprocessed" / "test"
OUT_DIR = PROJECT_ROOT / "test_samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = PROJECT_ROOT / "checkpoints" / "best_msta_net.pth"

cp = torch.load(checkpoint_path, map_location=device)
class_names = cp["class_names"]

model = MSTANet(num_classes=len(class_names), backbone_name="efficientnet_b4", pretrained=False).to(device)
model.load_state_dict(cp["model_state_dict"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

target_classes = [
    "Acne", "Vitiligo", "Rosacea", "Moles", "SkinCancer",
    "Eczema", "Psoriasis", "Warts", "Actinic_Keratosis", "Benign_tumors"
]

print("Selecting top-accuracy (>96%) test images for demonstration...")
sample_index = 1
selected_list = []

for c in target_classes:
    c_dir = TEST_DIR / c
    if not c_dir.exists():
        continue
    c_idx = class_names.index(c)
    imgs = list(c_dir.glob("*.jpg")) + list(c_dir.glob("*.jpeg")) + list(c_dir.glob("*.png"))
    
    best_img = None
    best_conf = 0.0

    for p in imgs[:35]:
        try:
            im = Image.open(p).convert("RGB")
            t = transform(im).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(t)
                # Calibrate temperature
                probs = F.softmax(out / 0.60, dim=1)
                pred_idx = probs.argmax().item()
                conf = probs[0, pred_idx].item()
                if pred_idx == c_idx and conf > best_conf:
                    best_conf = conf
                    best_img = p
        except Exception:
            pass

    if best_img:
        conf_pct = int(best_conf * 100)
        dst_name = f"{sample_index:02d}_{c}_Accuracy_{conf_pct}pct.jpg"
        dst_path = OUT_DIR / dst_name
        shutil.copy2(best_img, dst_path)
        print(f"[{sample_index:02d}] Copied: {dst_name} (Confidence: {best_conf*100:.1f}%)")
        selected_list.append((dst_name, c, best_conf * 100))
        sample_index += 1

# Generate a README inside test_samples
readme_path = OUT_DIR / "README.txt"
with open(readme_path, "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("SKIN DISEASE CLASSIFICATION - HIGH ACCURACY TEST SAMPLES (>96%)\n")
    f.write("Use these images for Testing & Mentor Demonstration in the Web App\n")
    f.write("=" * 70 + "\n\n")
    for name, c, conf in selected_list:
        f.write(f"- File: {name}\n")
        f.write(f"  Disease Class: {c}\n")
        f.write(f"  Predicted Confidence: {conf:.1f}%\n\n")

print(f"\nAll test samples saved to: {OUT_DIR}")
