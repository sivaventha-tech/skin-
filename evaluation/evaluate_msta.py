"""
============================================================
SKIN DISEASE CLASSIFICATION
MSTA-NET & BASE PAPER COMPREHENSIVE EVALUATION SCRIPT
IEEE Publication Benchmark Metrics & Figures
============================================================
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.msta_net import MSTANet
from preprocessing.augmentation import val_transform

DATASET_DIR = PROJECT_ROOT / "dataset" / "preprocessed"
TEST_DIR = DATASET_DIR / "test"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = RESULTS_DIR / "reports"
FIGURES_DIR = RESULTS_DIR / "figures"

MSTA_CHECKPOINT = CHECKPOINT_DIR / "best_msta_net.pth"

BASE_PAPER_ACCURACY = 94.57  # Reported base paper accuracy %

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model():
    print("=" * 80)
    print("MSTA-NET & BASE PAPER EVALUATION PIPELINE")
    print("=" * 80)

    test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    class_names = test_dataset.classes
    num_classes = len(class_names)

    print(f"Loaded test set with {len(test_dataset)} samples across {num_classes} classes.")

    if not MSTA_CHECKPOINT.exists():
        print(f"Error: MSTA checkpoint not found at {MSTA_CHECKPOINT}")
        sys.exit(1)

    checkpoint = torch.load(MSTA_CHECKPOINT, map_location=DEVICE)
    model = MSTANet(num_classes=num_classes, backbone_name="efficientnet_b4", pretrained=False).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Metrics
    acc = accuracy_score(all_targets, all_preds) * 100.0
    prec_macro = precision_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0
    rec_macro = recall_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0
    f1_macro = f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0

    prec_weighted = precision_score(all_targets, all_preds, average="weighted", zero_division=0) * 100.0
    rec_weighted = recall_score(all_targets, all_preds, average="weighted", zero_division=0) * 100.0
    f1_weighted = f1_score(all_targets, all_preds, average="weighted", zero_division=0) * 100.0

    print("\n" + "=" * 80)
    print(f"MSTA-NET PROPOSED MODEL ACCURACY: {acc:.2f}%")
    print(f"BASE PAPER ACCURACY BENCHMARK   : {BASE_PAPER_ACCURACY:.2f}%")
    accuracy_diff = acc - BASE_PAPER_ACCURACY
    print(f"ACCURACY IMPROVEMENT            : {accuracy_diff:+.2f}%")
    print("=" * 80)

    # Save metrics JSON
    metrics_summary = {
        "model_name": "Proposed MSTA-Net (Multi-Scale Tri-Stage Adaptive Attention)",
        "base_paper_accuracy_percent": BASE_PAPER_ACCURACY,
        "proposed_accuracy_percent": acc,
        "accuracy_improvement_percent": accuracy_diff,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
        "f1_weighted": f1_weighted,
        "test_samples": len(test_dataset),
        "num_classes": num_classes
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    with open(REPORTS_DIR / "msta_net_metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=4)

    # Classification report text
    clf_report = classification_report(all_targets, all_preds, target_names=class_names, digits=4)
    with open(REPORTS_DIR / "msta_net_classification_report.txt", "w") as f:
        f.write("MSTA-NET PROPOSED MODEL TEST SET CLASSIFICATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(clf_report)

    # IEEE Comparative CSV
    df_comp = pd.DataFrame([
        {
            "Architecture": "Base Paper (Tri-Attention EfficientNet-B0 + TS-GRA)",
            "Dataset": "HAM10000",
            "Accuracy (%)": BASE_PAPER_ACCURACY,
            "Classes": 7,
            "Explainability": "Grad-CAM"
        },
        {
            "Architecture": "Proposed MSTA-Net (Multi-Scale Tri-Stage Adaptive Attention)",
            "Dataset": "Skin Disease Cleaned Dataset",
            "Accuracy (%)": acc,
            "Classes": num_classes,
            "Explainability": "Grad-CAM++"
        }
    ])
    df_comp.to_csv(REPORTS_DIR / "base_paper_vs_proposed_comparison.csv", index=False)

    # IEEE LaTeX Table Export
    latex_table = f"""\\begin{{table}}[htbp]
\\caption{{Performance Comparison: Base Paper vs. Proposed MSTA-Net Architecture}}
\\label{{tab:performance_comparison}}
\\begin{{center}}
\\begin{{tabular}}{{|l|c|c|c|c|}}
\\hline
\\textbf{{Model Architecture}} & \\textbf{{Dataset}} & \\textbf{{Classes}} & \\textbf{{Accuracy (\\%)}} & \\textbf{{F1-Score (\\%)}} \\\\
\\hline
Base Paper Baseline (TS-GRA) & HAM10000 & 7 & {BASE_PAPER_ACCURACY:.2f} & 93.80 \\\\
\\textbf{{Proposed MSTA-Net (Ours)}} & \\textbf{{Cleaned Dataset}} & \\textbf{{{num_classes}}} & \\textbf{{{acc:.2f}}} & \\textbf{{{f1_weighted:.2f}}} \\\\
\\hline
\\end{{tabular}}
\\end{{center}}
\\end{{table}}"""

    with open(REPORTS_DIR / "ieee_results_table.tex", "w") as f:
        f.write(latex_table)

    # Confusion Matrix Figure
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title(f"Proposed MSTA-Net Confusion Matrix (Accuracy: {acc:.2f}%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "msta_net_confusion_matrix.png", dpi=300)
    plt.close()

    print(f"\nEvaluation completed successfully. Reports saved in {REPORTS_DIR}")


if __name__ == "__main__":
    evaluate_model()
