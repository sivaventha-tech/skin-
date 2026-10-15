"""
============================================================
SKIN DISEASE CLASSIFICATION
GRAD-CAM++ EXPLAINABILITY MODULE FOR MSTA-NET
IEEE Publication Quality Visual Interpretability
============================================================
"""

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.msta_net import MSTANet
from preprocessing.augmentation import val_transform, IMAGENET_MEAN, IMAGENET_STD

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CHECKPOINT_FILE = PROJECT_ROOT / "checkpoints" / "best_msta_net.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        score = output[0, target_class]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        # Grad-CAM++ Weights calculation
        grad_power_2 = gradients ** 2
        grad_power_3 = gradients ** 3
        sum_activations = torch.sum(activations, dim=(1, 2), keepdim=True)
        
        alpha_num = grad_power_2
        alpha_denom = 2 * grad_power_2 + sum_activations * grad_power_3 + 1e-7
        alpha = alpha_num / alpha_denom

        weights = torch.sum(alpha * F.relu(gradients), dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * activations, dim=0)

        cam = F.relu(cam).detach().cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        return cam, target_class


def denormalize_image(tensor):
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def generate_and_save_gradcam(image_path, target_class=None, output_filename="gradcam_msta_result.png"):
    if not CHECKPOINT_FILE.exists():
        print(f"Checkpoint not found at {CHECKPOINT_FILE}")
        return

    checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
    class_names = checkpoint.get("class_names", [])
    num_classes = len(class_names) if class_names else 22

    model = MSTANet(num_classes=num_classes, backbone_name="efficientnet_b4", pretrained=False).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Target last conv layer in backbone
    target_layer = model.msta_block.soft_gate.gate_conv[-2]

    grad_cam = GradCAMPlusPlus(model, target_layer)

    raw_img = Image.open(image_path).convert("RGB")
    input_tensor = val_transform(raw_img).unsqueeze(0).to(DEVICE)

    heatmap, pred_class_idx = grad_cam.generate_heatmap(input_tensor, target_class)
    pred_class_name = class_names[pred_class_idx] if pred_class_idx < len(class_names) else f"Class_{pred_class_idx}"

    orig_img_np = denormalize_image(input_tensor[0])
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(orig_img_np, 0.6, heatmap_colored, 0.4, 0)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / output_filename

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(orig_img_np)
    axes[0].set_title("Original Lesion Image")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title(f"Grad-CAM++ Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Diagnostic Focus ({pred_class_name})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved Grad-CAM++ visualization to {out_path}")
    return out_path


if __name__ == "__main__":
    test_dir = PROJECT_ROOT / "dataset" / "preprocessed" / "test"
    sample_imgs = list(test_dir.glob("*/*.jpeg")) + list(test_dir.glob("*/*.jpg")) + list(test_dir.glob("*/*.png"))
    if sample_imgs and CHECKPOINT_FILE.exists():
        generate_and_save_gradcam(sample_imgs[0])
