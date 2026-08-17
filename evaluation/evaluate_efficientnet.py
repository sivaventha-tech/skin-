"""
============================================================
SKIN DISEASE CLASSIFICATION
EFFICIENTNET-B0 TEST SET EVALUATION
============================================================

Model:
    EfficientNet-B0

Test set:
    2,285 images
    22 classes

Outputs:
    results/reports/efficientnet_classification_report.txt
    results/reports/efficientnet_metrics.json
    results/figures/efficientnet_confusion_matrix.png
============================================================
"""

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

try:
    from preprocessing.augmentation import val_transform
except Exception as error:
    print("=" * 80)
    print("ERROR: Could not import val_transform")
    print("=" * 80)
    print(repr(error))
    sys.exit(1)


try:
    from models.efficientnet_model import EfficientNetModel
except Exception as error:
    print("=" * 80)
    print("ERROR: Could not import EfficientNetModel")
    print("=" * 80)
    print(repr(error))
    sys.exit(1)


# ============================================================
# PATHS
# ============================================================

TEST_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "preprocessed"
    / "test"
)

CHECKPOINT_FILE = (
    PROJECT_ROOT
    / "checkpoints"
    / "best_efficientnet.pth"
)

REPORTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "reports"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
)

REPORT_FILE = (
    REPORTS_DIR
    / "efficientnet_classification_report.txt"
)

METRICS_FILE = (
    REPORTS_DIR
    / "efficientnet_metrics.json"
)

CONFUSION_MATRIX_FILE = (
    FIGURES_DIR
    / "efficientnet_confusion_matrix.png"
)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 32
NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CHECK FILES
# ============================================================

def check_files():

    print()
    print("=" * 80)
    print("CHECKING EFFICIENTNET EVALUATION FILES")
    print("=" * 80)

    print()
    print("Test dataset:")
    print(TEST_DIR)

    print()
    print("Checkpoint:")
    print(CHECKPOINT_FILE)

    if not TEST_DIR.exists():

        print()
        print("ERROR: Test dataset not found.")
        sys.exit(1)

    if not CHECKPOINT_FILE.exists():

        print()
        print("ERROR: EfficientNet checkpoint not found.")
        print(CHECKPOINT_FILE)
        sys.exit(1)

    print()
    print("✓ Test dataset found")
    print("✓ EfficientNet checkpoint found")


# ============================================================
# LOAD TEST DATASET
# ============================================================

def load_test_dataset():

    print()
    print("=" * 80)
    print("LOADING TEST DATASET")
    print("=" * 80)

    test_dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=val_transform
    )

    class_names = test_dataset.classes

    print()
    print(
        f"Number of classes : {len(class_names)}"
    )

    print(
        f"Test images       : {len(test_dataset):,}"
    )

    print()
    print("Class mapping:")

    for index, class_name in enumerate(
        class_names
    ):

        print(
            f"{index:>3} -> {class_name}"
        )

    return test_dataset, class_names


# ============================================================
# CREATE DATALOADER
# ============================================================

def create_dataloader(test_dataset):

    print()
    print("=" * 80)
    print("CREATING TEST DATALOADER")
    print("=" * 80)

    loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print()
    print(f"Batch size : {BATCH_SIZE}")
    print(f"Batches    : {len(loader)}")

    return loader


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(class_names):

    print()
    print("=" * 80)
    print("LOADING EFFICIENTNET-B0 MODEL")
    print("=" * 80)

    print()
    print(f"Device: {DEVICE}")

    # IMPORTANT:
    # Do not download ImageNet weights again.
    model = EfficientNetModel(
        num_classes=len(class_names),
        pretrained=False,
        freeze_backbone=True
    )

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # Checkpoint created by train_efficientnet.py
    # --------------------------------------------------------

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

        checkpoint_epoch = checkpoint.get(
            "epoch",
            "unknown"
        )

        checkpoint_accuracy = checkpoint.get(
            "val_accuracy",
            None
        )

        print()
        print(
            f"Checkpoint epoch : "
            f"{checkpoint_epoch}"
        )

        if checkpoint_accuracy is not None:

            print(
                f"Checkpoint val accuracy : "
                f"{checkpoint_accuracy * 100:.2f}%"
            )

    else:

        state_dict = checkpoint

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    model.load_state_dict(
        state_dict
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print(
        f"Total parameters : "
        f"{total_parameters:,}"
    )

    print(
        "✓ EfficientNet-B0 model loaded successfully"
    )

    return model


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

def generate_predictions(
    model,
    test_loader
):

    print()
    print("=" * 80)
    print("GENERATING TEST PREDICTIONS")
    print("=" * 80)

    all_labels = []
    all_predictions = []

    total = 0
    correct = 0

    model.eval()

    with torch.no_grad():

        for batch_index, (
            images,
            labels
        ) in enumerate(
            test_loader,
            start=1
        ):

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            outputs = model(
                images
            )

            predictions = outputs.argmax(
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            batch_correct = (
                predictions == labels
            ).sum().item()

            correct += batch_correct
            total += labels.size(0)

            if (
                batch_index % 20 == 0
                or batch_index == len(test_loader)
            ):

                running_accuracy = (
                    correct / total
                )

                print(
                    f"Processed "
                    f"{total:,}/"
                    f"{len(test_loader.dataset):,} "
                    f"| Running Accuracy: "
                    f"{running_accuracy * 100:.2f}%"
                )

    return (
        np.array(all_labels),
        np.array(all_predictions)
    )


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    print()
    print("=" * 80)
    print("CALCULATING TEST METRICS")
    print("=" * 80)

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision_macro = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    precision_weighted = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall_weighted = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    metrics = {

        "model": "EfficientNet-B0",

        "test_samples": int(
            len(y_true)
        ),

        "number_of_classes": int(
            len(np.unique(y_true))
        ),

        "accuracy": float(
            accuracy
        ),

        "accuracy_percent": float(
            accuracy * 100
        ),

        "precision_macro": float(
            precision_macro
        ),

        "recall_macro": float(
            recall_macro
        ),

        "f1_macro": float(
            f1_macro
        ),

        "precision_weighted": float(
            precision_weighted
        ),

        "recall_weighted": float(
            recall_weighted
        ),

        "f1_weighted": float(
            f1_weighted
        )
    }

    print()

    print(
        f"Test Accuracy       : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Macro Precision     : "
        f"{precision_macro * 100:.2f}%"
    )

    print(
        f"Macro Recall        : "
        f"{recall_macro * 100:.2f}%"
    )

    print(
        f"Macro F1-score      : "
        f"{f1_macro * 100:.2f}%"
    )

    print()

    print(
        f"Weighted Precision  : "
        f"{precision_weighted * 100:.2f}%"
    )

    print(
        f"Weighted Recall     : "
        f"{recall_weighted * 100:.2f}%"
    )

    print(
        f"Weighted F1-score   : "
        f"{f1_weighted * 100:.2f}%"
    )

    return metrics


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

def save_classification_report(
    y_true,
    y_pred,
    class_names
):

    print()
    print("=" * 80)
    print("CLASSIFICATION REPORT")
    print("=" * 80)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    print()
    print(report)

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "SKIN DISEASE CLASSIFICATION\n"
        )

        file.write(
            "EFFICIENTNET-B0 TEST SET EVALUATION\n"
        )

        file.write(
            "=" * 80
            + "\n\n"
        )

        file.write(report)

    print()
    print("Classification report saved:")
    print(REPORT_FILE)


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(metrics):

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print()
    print("Metrics saved:")
    print(METRICS_FILE)


# ============================================================
# CONFUSION MATRIX
# ============================================================

def generate_confusion_matrix(
    y_true,
    y_pred,
    class_names
):

    print()
    print("=" * 80)
    print("GENERATING CONFUSION MATRIX")
    print("=" * 80)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            range(
                len(class_names)
            )
        )
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(
        figsize=(16, 14)
    )

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(
        "EfficientNet-B0 Confusion Matrix - "
        "Skin Disease Classification"
    )

    plt.colorbar()

    tick_marks = np.arange(
        len(class_names)
    )

    plt.xticks(
        tick_marks,
        class_names,
        rotation=90
    )

    plt.yticks(
        tick_marks,
        class_names
    )

    threshold = (
        cm.max() / 2.0
    )

    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            value = cm[i, j]

            text_color = (
                "white"
                if value > threshold
                else "black"
            )

            plt.text(
                j,
                i,
                str(value),
                horizontalalignment="center",
                verticalalignment="center",
                color=text_color,
                fontsize=7
            )

    plt.ylabel(
        "True Class"
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("Confusion matrix saved:")
    print(CONFUSION_MATRIX_FILE)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("SKIN DISEASE CLASSIFICATION")
    print("EFFICIENTNET-B0 TEST SET EVALUATION")
    print("=" * 80)

    check_files()

    test_dataset, class_names = (
        load_test_dataset()
    )

    test_loader = create_dataloader(
        test_dataset
    )

    model = load_model(
        class_names
    )

    y_true, y_pred = (
        generate_predictions(
            model,
            test_loader
        )
    )

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    save_classification_report(
        y_true,
        y_pred,
        class_names
    )

    save_metrics(
        metrics
    )

    generate_confusion_matrix(
        y_true,
        y_pred,
        class_names
    )

    print()
    print("=" * 80)
    print("EFFICIENTNET-B0 EVALUATION COMPLETED")
    print("=" * 80)

    print()

    print(
        f"Test Accuracy : "
        f"{metrics['accuracy_percent']:.2f}%"
    )

    print(
        f"Macro F1      : "
        f"{metrics['f1_macro'] * 100:.2f}%"
    )

    print(
        f"Weighted F1   : "
        f"{metrics['f1_weighted'] * 100:.2f}%"
    )

    print()

    print("Generated files:")

    print(REPORT_FILE)
    print(METRICS_FILE)
    print(CONFUSION_MATRIX_FILE)

    print()
    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print("Evaluation stopped by user.")

        sys.exit(1)

    except Exception as error:

        print()
        print("=" * 80)
        print("UNEXPECTED ERROR")
        print("=" * 80)

        print()

        print(
            f"{type(error).__name__}: {error}"
        )

        import traceback

        traceback.print_exc()

        sys.exit(1)