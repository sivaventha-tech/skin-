"""
============================================================
SKIN DISEASE CLASSIFICATION
MODEL EVALUATION
============================================================

Evaluates:
    ResNet50

Test set:
    dataset/preprocessed/test

Outputs:
    results/reports/resnet50_classification_report.txt
    results/reports/resnet50_metrics.json
    results/figures/resnet50_confusion_matrix.png

Metrics:
    Accuracy
    Precision
    Recall
    F1-score
    Macro F1
    Weighted F1

============================================================
"""

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

try:

    from preprocessing.augmentation import (
        val_transform
    )

except Exception as error:

    print("=" * 80)
    print("ERROR: Could not import augmentation.py")
    print("=" * 80)

    print(
        repr(error)
    )

    sys.exit(1)


try:

    from models.resnet50_model import (
        ResNet50Model
    )

except Exception as error:

    print("=" * 80)
    print("ERROR: Could not import ResNet50Model")
    print("=" * 80)

    print(
        repr(error)
    )

    sys.exit(1)


# ============================================================
# PATHS
# ============================================================

DATASET_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "preprocessed"
)

TEST_DIR = (
    DATASET_DIR
    / "test"
)

CHECKPOINT_FILE = (
    PROJECT_ROOT
    / "checkpoints"
    / "best_resnet50.pth"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

REPORTS_DIR = (
    RESULTS_DIR
    / "reports"
)

FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
)

REPORT_FILE = (
    REPORTS_DIR
    / "resnet50_classification_report.txt"
)

METRICS_FILE = (
    REPORTS_DIR
    / "resnet50_metrics.json"
)

CONFUSION_MATRIX_FILE = (
    FIGURES_DIR
    / "resnet50_confusion_matrix.png"
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
# HEADER
# ============================================================

def print_header(title):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# CHECK FILES
# ============================================================

def check_files():

    print_header(
        "CHECKING EVALUATION FILES"
    )

    print()
    print(
        "Test dataset:"
    )

    print(
        TEST_DIR
    )

    print()
    print(
        "Checkpoint:"
    )

    print(
        CHECKPOINT_FILE
    )

    if not TEST_DIR.exists():

        print()
        print(
            "ERROR: Test dataset not found."
        )

        sys.exit(1)

    if not CHECKPOINT_FILE.exists():

        print()
        print(
            "ERROR: ResNet50 checkpoint not found."
        )

        print(
            CHECKPOINT_FILE
        )

        print()
        print(
            "Train ResNet50 first."
        )

        sys.exit(1)

    print()
    print(
        "✓ Test dataset found"
    )

    print(
        "✓ ResNet50 checkpoint found"
    )


# ============================================================
# LOAD TEST DATASET
# ============================================================

def load_test_dataset():

    print_header(
        "LOADING TEST DATASET"
    )

    test_dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=val_transform
    )

    class_names = test_dataset.classes

    print()
    print(
        f"Number of classes : "
        f"{len(class_names)}"
    )

    print(
        f"Test images       : "
        f"{len(test_dataset):,}"
    )

    print()

    print(
        "Class mapping:"
    )

    for index, class_name in enumerate(
        class_names
    ):

        print(
            f"{index:>3} -> {class_name}"
        )

    return (
        test_dataset,
        class_names
    )


# ============================================================
# CREATE DATALOADER
# ============================================================

def create_dataloader(
    test_dataset
):

    print_header(
        "CREATING TEST DATALOADER"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print()
    print(
        f"Batch size : {BATCH_SIZE}"
    )

    print(
        f"Batches    : {len(test_loader)}"
    )

    return test_loader


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    class_names
):

    print_header(
        "LOADING RESNET50 MODEL"
    )

    print()
    print(
        f"Device: {DEVICE}"
    )

    # --------------------------------------------------------
    # Create architecture WITHOUT downloading weights
    # --------------------------------------------------------

    model = ResNet50Model(
        num_classes=len(class_names),
        pretrained=False,
        freeze_backbone=False
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # Handle checkpoint format
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ) and "model_state_dict" in checkpoint:

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

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
    # Load parameters
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
        "✓ Model loaded successfully"
    )

    return model


# ============================================================
# PREDICTION
# ============================================================

def generate_predictions(
    model,
    test_loader
):

    print_header(
        "GENERATING TEST PREDICTIONS"
    )

    all_predictions = []

    all_labels = []

    total = 0

    correct = 0

    with torch.no_grad():

        for batch_index, (
            images,
            labels
        ) in enumerate(test_loader, 1):

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            outputs = model(
                images
            )

            predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
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

                current_accuracy = (
                    correct / total
                )

                print(
                    f"Processed "
                    f"{total:,}/"
                    f"{len(test_loader.dataset):,} "
                    f"| Running Accuracy: "
                    f"{current_accuracy * 100:.2f}%"
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

    print_header(
        "CALCULATING TEST METRICS"
    )

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

        "model": "ResNet50",

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

def generate_classification_report(
    y_true,
    y_pred,
    class_names
):

    print_header(
        "CLASSIFICATION REPORT"
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    print()

    print(
        report
    )

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
            "RESNET50 TEST SET EVALUATION\n"
        )

        file.write(
            "=" * 80
            + "\n\n"
        )

        file.write(
            report
        )

    print()

    print(
        "Classification report saved:"
    )

    print(
        REPORT_FILE
    )


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(
    metrics
):

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

    print(
        "Metrics saved:"
    )

    print(
        METRICS_FILE
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def generate_confusion_matrix(
    y_true,
    y_pred,
    class_names
):

    print_header(
        "GENERATING CONFUSION MATRIX"
    )

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

    # --------------------------------------------------------
    # Large figure because there are 22 classes
    # --------------------------------------------------------

    plt.figure(
        figsize=(16, 14)
    )

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(
        "ResNet50 Confusion Matrix - Skin Disease Classification"
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

    # --------------------------------------------------------
    # Add values
    # --------------------------------------------------------

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

            if value > threshold:

                text_color = "white"

            else:

                text_color = "black"

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

    print(
        "Confusion matrix saved:"
    )

    print(
        CONFUSION_MATRIX_FILE
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("SKIN DISEASE CLASSIFICATION")
    print("RESNET50 TEST SET EVALUATION")
    print("=" * 80)

    # --------------------------------------------------------
    # Check
    # --------------------------------------------------------

    check_files()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    (
        test_dataset,
        class_names
    ) = load_test_dataset()

    # --------------------------------------------------------
    # Loader
    # --------------------------------------------------------

    test_loader = create_dataloader(
        test_dataset
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = load_model(
        class_names
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    (
        y_true,
        y_pred
    ) = generate_predictions(
        model,
        test_loader
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    generate_classification_report(
        y_true,
        y_pred,
        class_names
    )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    save_metrics(
        metrics
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    generate_confusion_matrix(
        y_true,
        y_pred,
        class_names
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("RESNET50 EVALUATION COMPLETED")
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

    print(
        "Generated files:"
    )

    print(
        REPORT_FILE
    )

    print(
        METRICS_FILE
    )

    print(
        CONFUSION_MATRIX_FILE
    )

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
        print(
            "Evaluation stopped by user."
        )

        sys.exit(1)

    except Exception as error:

        print()
        print("=" * 80)
        print("UNEXPECTED ERROR")
        print("=" * 80)

        print()

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        import traceback

        traceback.print_exc()

        sys.exit(1)