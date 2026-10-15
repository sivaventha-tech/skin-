"""
============================================================
SKIN DISEASE CLASSIFICATION
MSTA-NET TWO-STAGE OPTIMIZED FINE-TUNING PIPELINE
Targeting Highest Accuracy (>95%+ on benchmark / >65-98% ensemble)
============================================================
"""

import os
import sys
import json
import random
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.msta_net import MSTANet, FocalPolyLoss
from preprocessing.augmentation import train_transform, val_transform

# Paths
DATASET_DIR = PROJECT_ROOT / "dataset" / "preprocessed"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
BEST_MODEL_FILE = CHECKPOINT_DIR / "best_msta_net.pth"
HISTORY_FILE = RESULTS_DIR / "msta_net_training_history.json"

BATCH_SIZE = 32
STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 15
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

    return total_loss / total, correct / total


def main():
    set_seed(RANDOM_SEED)
    print("=" * 80, flush=True)
    print("MSTA-NET TWO-STAGE HIGH ACCURACY TRAINING PIPELINE", flush=True)
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print("=" * 80, flush=True)

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_transform)

    num_classes = len(train_dataset.classes)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Class weights for focal loss
    labels = np.array(train_dataset.targets)
    cls_weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
    cls_weights_tensor = torch.tensor(cls_weights, dtype=torch.float32, device=DEVICE)
    criterion = FocalPolyLoss(alpha=cls_weights_tensor, gamma=2.0, poly_epsilon=1.0, label_smoothing=0.05)

    model = MSTANet(num_classes=num_classes, backbone_name="efficientnet_b4", pretrained=True).to(DEVICE)
    scaler = torch.amp.GradScaler('cuda')

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}

    # ========================================================
    # STAGE 1: Head Warmup (Backbone Frozen)
    # ========================================================
    print("\n--- [STAGE 1] Head Warmup (Backbone Frozen) ---", flush=True)
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer_stage1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)

    for epoch in range(1, STAGE1_EPOCHS + 1):
        ep_start = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer_stage1, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion)
        ep_duration = time.time() - ep_start

        history["train_loss"].append(tr_loss)
        history["train_accuracy"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(f"Stage 1 - Epoch {epoch:02d}/{STAGE1_EPOCHS:02d} [{ep_duration:.1f}s] - Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc*100:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "class_names": train_dataset.classes,
                "model_name": "MSTA-Net (EfficientNet-B4 Backbone)"
            }, BEST_MODEL_FILE)

    # ========================================================
    # STAGE 2: Full End-to-End Fine-Tuning (Backbone Unfrozen)
    # ========================================================
    print("\n--- [STAGE 2] Full End-to-End Fine-Tuning (Backbone Unfrozen) ---", flush=True)
    for param in model.backbone.parameters():
        param.requires_grad = True

    param_groups = model.get_layer_parameter_groups(lr_backbone=1e-4, lr_head=5e-4, weight_decay=1e-4)
    optimizer_stage2 = optim.AdamW(param_groups)
    scheduler_stage2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_stage2, T_max=STAGE2_EPOCHS, eta_min=1e-6)

    for epoch in range(1, STAGE2_EPOCHS + 1):
        ep_start = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer_stage2, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler_stage2.step()
        ep_duration = time.time() - ep_start

        history["train_loss"].append(tr_loss)
        history["train_accuracy"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(f"Stage 2 - Epoch {epoch:02d}/{STAGE2_EPOCHS:02d} [{ep_duration:.1f}s] - Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc*100:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": STAGE1_EPOCHS + epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "class_names": train_dataset.classes,
                "model_name": "MSTA-Net (EfficientNet-B4 Backbone)"
            }, BEST_MODEL_FILE)
            print(f"  -> Saved Best MSTA-Net Checkpoint (Val Acc: {best_val_acc*100:.2f}%)", flush=True)

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

    print("\n--- Final Test Set Evaluation ---", flush=True)
    model.load_state_dict(torch.load(BEST_MODEL_FILE, weights_only=False)["model_state_dict"])
    test_loss, test_acc = validate(model, test_loader, criterion)
    print(f"MSTA-Net Final Test Accuracy: {test_acc*100:.2f}% (Test Loss: {test_loss:.4f})", flush=True)


if __name__ == "__main__":
    main()
