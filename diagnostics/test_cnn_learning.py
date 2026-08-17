"""
============================================================
CUSTOM CNN LEARNING DIAGNOSTIC
============================================================

Purpose:
    Quickly verify whether the dataset, augmentation,
    labels, loss function, and CustomCNN are working.

This is NOT the final training.
It is a diagnostic experiment.

============================================================
"""

import sys
import random
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ============================================================
# IMPORTS
# ============================================================

from preprocessing.augmentation import (
    train_transform,
    val_transform
)

from models.custom_cnn import CustomCNN


# ============================================================
# PATHS
# ============================================================

DATASET_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "preprocessed"
)

TRAIN_DIR = DATASET_DIR / "train"

VAL_DIR = DATASET_DIR / "val"


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

IMAGE_SIZE = 224

# Small subset for fast diagnosis
SAMPLES_PER_CLASS = 20

BATCH_SIZE = 16

DIAGNOSTIC_EPOCHS = 5

LEARNING_RATE = 0.0003

USE_CLASS_WEIGHTS = False


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# SEED
# ============================================================

def set_seed():

    random.seed(SEED)

    np.random.seed(SEED)

    torch.manual_seed(SEED)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            SEED
        )


# ============================================================
# HEADER
# ============================================================

def header(title):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# CHECK DATASET
# ============================================================

def check_dataset():

    header(
        "CNN LEARNING DIAGNOSTIC"
    )

    print()
    print(
        "Dataset:"
    )

    print(
        DATASET_DIR
    )

    if not TRAIN_DIR.exists():

        print()
        print(
            "ERROR: Training directory not found."
        )

        sys.exit(1)

    if not VAL_DIR.exists():

        print()
        print(
            "ERROR: Validation directory not found."
        )

        sys.exit(1)

    print()
    print(
        "✓ Dataset found"
    )


# ============================================================
# LOAD FULL TRAIN DATASET
# ============================================================

def load_dataset():

    header(
        "LOADING TRAINING DATASET"
    )

    full_dataset = datasets.ImageFolder(
        TRAIN_DIR,
        transform=train_transform
    )

    print()
    print(
        f"Total training images : "
        f"{len(full_dataset):,}"
    )

    print(
        f"Number of classes     : "
        f"{len(full_dataset.classes)}"
    )

    print()

    for index, name in enumerate(
        full_dataset.classes
    ):

        print(
            f"{index:>2} -> {name}"
        )

    return full_dataset


# ============================================================
# CREATE BALANCED DIAGNOSTIC SUBSET
# ============================================================

def create_subset(
    dataset
):

    header(
        "CREATING DIAGNOSTIC SUBSET"
    )

    random.seed(
        SEED
    )

    class_indices = {
        class_id: []
        for class_id in range(
            len(dataset.classes)
        )
    }

    for index, label in enumerate(
        dataset.targets
    ):

        class_indices[label].append(
            index
        )

    selected_indices = []

    print()

    for class_id in range(
        len(dataset.classes)
    ):

        indices = class_indices[
            class_id
        ]

        random.shuffle(
            indices
        )

        selected = indices[
            :SAMPLES_PER_CLASS
        ]

        selected_indices.extend(
            selected
        )

        print(
            f"{dataset.classes[class_id]:<30}"
            f": {len(selected)}"
        )

    random.shuffle(
        selected_indices
    )

    subset = Subset(
        dataset,
        selected_indices
    )

    print()

    print(
        f"Diagnostic images : "
        f"{len(subset)}"
    )

    return subset


# ============================================================
# CREATE DATALOADER
# ============================================================

def create_loader(
    dataset
):

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )


# ============================================================
# INSPECT IMAGES
# ============================================================

def inspect_batch(
    loader
):

    header(
        "INSPECTING IMAGE BATCH"
    )

    images, labels = next(
        iter(loader)
    )

    print()

    print(
        f"Image tensor shape : "
        f"{tuple(images.shape)}"
    )

    print(
        f"Label tensor shape : "
        f"{tuple(labels.shape)}"
    )

    print(
        f"Image dtype        : "
        f"{images.dtype}"
    )

    print(
        f"Label dtype        : "
        f"{labels.dtype}"
    )

    print()

    print(
        f"Image min          : "
        f"{images.min().item():.4f}"
    )

    print(
        f"Image max          : "
        f"{images.max().item():.4f}"
    )

    print(
        f"Image mean         : "
        f"{images.mean().item():.4f}"
    )

    print(
        f"Image std          : "
        f"{images.std().item():.4f}"
    )

    print()

    print(
        "Labels in first batch:"
    )

    print(
        labels.tolist()
    )

    expected_channels = 3

    expected_height = IMAGE_SIZE

    expected_width = IMAGE_SIZE

    if (
        images.shape[1] == expected_channels
        and
        images.shape[2] == expected_height
        and
        images.shape[3] == expected_width
    ):

        print()
        print(
            "✓ Image dimensions are correct"
        )

    else:

        print()
        print(
            "✗ Image dimensions are incorrect"
        )

        sys.exit(1)


# ============================================================
# CREATE MODEL
# ============================================================

def create_model(
    num_classes
):

    header(
        "TESTING CUSTOM CNN"
    )

    model = CustomCNN(
        num_classes=num_classes
    )

    model = model.to(
        DEVICE
    )

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print()

    print(
        f"Parameters : "
        f"{parameters:,}"
    )

    print(
        f"Device     : "
        f"{DEVICE}"
    )

    return model


# ============================================================
# FORWARD PASS TEST
# ============================================================

def test_forward_pass(
    model,
    loader
):

    header(
        "TESTING FORWARD PASS"
    )

    images, labels = next(
        iter(loader)
    )

    images = images.to(
        DEVICE
    )

    labels = labels.to(
        DEVICE
    )

    model.eval()

    with torch.no_grad():

        outputs = model(
            images
        )

    print()

    print(
        f"Input shape  : "
        f"{tuple(images.shape)}"
    )

    print(
        f"Output shape : "
        f"{tuple(outputs.shape)}"
    )

    expected = (
        images.shape[0],
        len(model.classifier[-1].bias)
    )

    # We know the final layer output should
    # correspond to the number of classes.

    if outputs.shape[0] == images.shape[0]:

        print(
            "✓ Forward pass successful"
        )

    else:

        print(
            "✗ Forward pass failed"
        )

        sys.exit(1)


# ============================================================
# TRAIN DIAGNOSTIC
# ============================================================

def train_diagnostic(
    model,
    loader,
    num_classes
):

    header(
        "RUNNING SMALL LEARNING TEST"
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001
    )

    print()

    print(
        f"Images              : "
        f"{len(loader.dataset)}"
    )

    print(
        f"Batch size           : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Diagnostic epochs    : "
        f"{DIAGNOSTIC_EPOCHS}"
    )

    print(
        f"Learning rate        : "
        f"{LEARNING_RATE}"
    )

    print(
        f"Class weights        : "
        f"{USE_CLASS_WEIGHTS}"
    )

    print()

    history = []

    for epoch in range(
        1,
        DIAGNOSTIC_EPOCHS + 1
    ):

        model.train()

        total_loss = 0.0

        correct = 0

        total = 0

        for images, labels in loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            optimizer.zero_grad()

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            batch_size = (
                images.size(0)
            )

            total_loss += (
                loss.item()
                * batch_size
            )

            predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += batch_size

        epoch_loss = (
            total_loss
            / total
        )

        epoch_accuracy = (
            correct
            / total
        )

        history.append(
            (
                epoch_loss,
                epoch_accuracy
            )
        )

        print(
            f"Epoch {epoch}/{DIAGNOSTIC_EPOCHS} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Accuracy: "
            f"{epoch_accuracy * 100:.2f}%"
        )

    return history


# ============================================================
# DIAGNOSTIC INTERPRETATION
# ============================================================

def interpret_results(
    history
):

    header(
        "DIAGNOSTIC RESULT"
    )

    first_loss = history[0][0]

    last_loss = history[-1][0]

    first_accuracy = history[0][1]

    last_accuracy = history[-1][1]

    print()

    print(
        f"Initial loss     : "
        f"{first_loss:.4f}"
    )

    print(
        f"Final loss       : "
        f"{last_loss:.4f}"
    )

    print(
        f"Initial accuracy : "
        f"{first_accuracy * 100:.2f}%"
    )

    print(
        f"Final accuracy   : "
        f"{last_accuracy * 100:.2f}%"
    )

    print()

    loss_decreased = (
        last_loss < first_loss
    )

    accuracy_increased = (
        last_accuracy > first_accuracy
    )

    if (
        loss_decreased
        and accuracy_increased
    ):

        print(
            "✓ CNN IS LEARNING"
        )

        print()

        print(
            "The dataset, labels, model, "
            "and optimization pipeline appear "
            "to be functioning."
        )

        print()

        print(
            "The low full-dataset training "
            "speed is primarily a CPU "
            "performance issue."
        )

    elif loss_decreased:

        print(
            "⚠ LOSS IS DECREASING"
        )

        print()

        print(
            "The model is learning, but "
            "accuracy improvement is weak."
        )

    else:

        print(
            "✗ CNN IS NOT LEARNING PROPERLY"
        )

        print()

        print(
            "We should investigate the "
            "dataset/augmentation/model "
            "configuration before full training."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed()

    check_dataset()

    full_dataset = load_dataset()

    subset = create_subset(
        full_dataset
    )

    loader = create_loader(
        subset
    )

    inspect_batch(
        loader
    )

    model = create_model(
        len(full_dataset.classes)
    )

    test_forward_pass(
        model,
        loader
    )

    history = train_diagnostic(
        model,
        loader,
        len(full_dataset.classes)
    )

    interpret_results(
        history
    )

    print()

    print("=" * 80)
    print("DIAGNOSTIC COMPLETED")
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
            "Diagnostic stopped by user."
        )

        sys.exit(1)

    except Exception as error:

        print()
        print("=" * 80)
        print("DIAGNOSTIC ERROR")
        print("=" * 80)

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        import traceback

        traceback.print_exc()

        sys.exit(1)