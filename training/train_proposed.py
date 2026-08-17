"""
============================================================
SKIN DISEASE CLASSIFICATION
PROPOSED MODEL TRAINING
ATTENTION-ENHANCED EFFICIENTNET-B0
============================================================

Train      : 10,549
Validation : 2,252
Test       : 2,285
Classes    : 22

Backbone:
    EfficientNet-B0 pretrained on ImageNet

Proposed contribution:
    Channel Attention
    Spatial Attention

Initially:
    EfficientNet backbone = frozen
    Attention + classifier = trainable

============================================================
"""

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
    from preprocessing.augmentation import (
        train_transform,
        val_transform
    )
except Exception as error:
    print("=" * 80)
    print("ERROR: Could not import augmentation.py")
    print("=" * 80)
    print(repr(error))
    sys.exit(1)


try:
    from models.proposed_model import ProposedModel
except Exception as error:
    print("=" * 80)
    print("ERROR: Could not import ProposedModel")
    print("=" * 80)
    print(repr(error))
    sys.exit(1)


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
TEST_DIR = DATASET_DIR / "test"

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
)

BEST_MODEL_FILE = (
    CHECKPOINT_DIR
    / "best_proposed_model.pth"
)

HISTORY_FILE = (
    RESULTS_DIR
    / "proposed_training_history.json"
)

CLASS_NAMES_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "metadata"
    / "class_names.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 32

NUM_EPOCHS = 12

LEARNING_RATE = 0.0001

WEIGHT_DECAY = 0.0001

NUM_WORKERS = 0

RANDOM_SEED = 42

EARLY_STOPPING_PATIENCE = 4

MIN_DELTA = 0.0001


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

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# CHECK DATASET
# ============================================================

def check_dataset():

    print()
    print("=" * 80)
    print("CHECKING PROPOSED MODEL DATASET")
    print("=" * 80)

    print()
    print("Dataset:")
    print(DATASET_DIR)

    required_dirs = [
        TRAIN_DIR,
        VAL_DIR,
        TEST_DIR
    ]

    for directory in required_dirs:

        if not directory.exists():

            print()
            print("ERROR: Missing directory:")
            print(directory)

            sys.exit(1)

    print()
    print("✓ Train directory found")
    print("✓ Validation directory found")
    print("✓ Test directory found")


# ============================================================
# LOAD DATASETS
# ============================================================

def load_datasets():

    print()
    print("=" * 80)
    print("LOADING DATASETS")
    print("=" * 80)

    train_dataset = datasets.ImageFolder(
        TRAIN_DIR,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        VAL_DIR,
        transform=val_transform
    )

    test_dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=val_transform
    )

    if train_dataset.class_to_idx != val_dataset.class_to_idx:

        raise RuntimeError(
            "Train and validation class mappings differ."
        )

    if train_dataset.class_to_idx != test_dataset.class_to_idx:

        raise RuntimeError(
            "Train and test class mappings differ."
        )

    class_names = train_dataset.classes

    print()

    print(
        f"Number of classes : {len(class_names)}"
    )

    print(
        f"Train images      : {len(train_dataset):,}"
    )

    print(
        f"Validation images : {len(val_dataset):,}"
    )

    print(
        f"Test images       : {len(test_dataset):,}"
    )

    print()
    print("Class mapping:")

    for index, name in enumerate(class_names):

        print(
            f"{index:>3} -> {name}"
        )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        class_names
    )


# ============================================================
# DATALOADERS
# ============================================================

def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset
):

    print()
    print("=" * 80)
    print("CREATING DATALOADERS")
    print("=" * 80)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    print()
    print(f"Batch size        : {BATCH_SIZE}")
    print(f"Workers           : {NUM_WORKERS}")
    print(f"Train batches     : {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches      : {len(test_loader)}")

    return (
        train_loader,
        val_loader,
        test_loader
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(
    train_dataset
):

    print()
    print("=" * 80)
    print("CALCULATING CLASS WEIGHTS")
    print("=" * 80)

    labels = np.array(
        train_dataset.targets
    )

    classes = np.unique(
        labels
    )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels
    )

    weights_tensor = torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE
    )

    print()

    for index, weight in enumerate(weights):

        print(
            f"{index:>2}. "
            f"{train_dataset.classes[index]:<30}"
            f"{weight:.4f}"
        )

    return weights_tensor


# ============================================================
# SAVE CLASS NAMES
# ============================================================

def save_class_names(class_names):

    CLASS_NAMES_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        CLASS_NAMES_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            class_names,
            file,
            indent=4
        )


# ============================================================
# CREATE MODEL
# ============================================================

def create_model(num_classes):

    print()
    print("=" * 80)
    print("CREATING PROPOSED MODEL")
    print("=" * 80)

    model = ProposedModel(
        num_classes=num_classes,
        pretrained=True,
        freeze_backbone=True
    )

    model = model.to(
        DEVICE
    )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    frozen_parameters = (
        total_parameters
        - trainable_parameters
    )

    print()

    print(
        "Model name        : "
        "Attention-Enhanced EfficientNet-B0"
    )

    print(
        f"Number of classes : {num_classes}"
    )

    print(
        "Input size        : 224 x 224"
    )

    print(
        f"Total parameters  : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable params  : "
        f"{trainable_parameters:,}"
    )

    print(
        f"Frozen params     : "
        f"{frozen_parameters:,}"
    )

    print(
        f"Device            : {DEVICE}"
    )

    print()
    print("Architecture contribution:")
    print("✓ EfficientNet-B0 backbone")
    print("✓ Channel Attention")
    print("✓ Spatial Attention")
    print("✓ Global Average Pooling")
    print("✓ Enhanced classification head")

    return model


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
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

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        batch_size = images.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

    return (
        total_loss / total,
        correct / total
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            batch_size = images.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += batch_size

    return (
        total_loss / total,
        correct / total
    )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    val_loss,
    val_accuracy,
    class_names
):

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "val_loss":
            float(val_loss),

        "val_accuracy":
            float(val_accuracy),

        "class_names":
            class_names,

        "num_classes":
            len(class_names),

        "image_size":
            224,

        "model_name":
            "Attention-Enhanced EfficientNet-B0",

        "attention":
            "Channel + Spatial"
    }

    torch.save(
        checkpoint,
        BEST_MODEL_FILE
    )


# ============================================================
# SAVE HISTORY
# ============================================================

def save_history(history):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history,
            file,
            indent=4
        )


# ============================================================
# PLOT HISTORY
# ============================================================

def plot_history(history):

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        epochs,
        history["train_accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "Proposed Model Training and Validation Accuracy"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "proposed_training_accuracy.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        epochs,
        history["train_loss"],
        label="Training Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Proposed Model Training and Validation Loss"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "proposed_training_loss.png",
        dpi=300
    )

    plt.close()


# ============================================================
# TRAINING LOOP
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    class_names
):

    print()
    print("=" * 80)
    print("STARTING PROPOSED MODEL TRAINING")
    print("=" * 80)

    print()

    print(
        "Model:"
    )

    print(
        "Attention-Enhanced EfficientNet-B0"
    )

    print()

    print(
        f"Maximum epochs       : {NUM_EPOCHS}"
    )

    print(
        f"Batch size            : {BATCH_SIZE}"
    )

    print(
        f"Learning rate         : {LEARNING_RATE}"
    )

    print(
        f"Weight decay          : {WEIGHT_DECAY}"
    )

    print(
        f"Early stopping        : "
        f"{EARLY_STOPPING_PATIENCE} epochs"
    )

    print(
        f"Device                : {DEVICE}"
    )

    print()

    history = {

        "train_loss": [],

        "train_accuracy": [],

        "val_loss": [],

        "val_accuracy": [],

        "learning_rate": [],

        "epoch_time_seconds": []
    }

    best_accuracy = 0.0

    best_loss = float("inf")

    best_epoch = 0

    patience = 0

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        start_time = time.time()

        print("-" * 80)

        print(
            f"Epoch {epoch}/{NUM_EPOCHS}"
        )

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer
            )
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss, val_accuracy = (
            validate(
                model,
                val_loader,
                criterion
            )
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(
            val_loss
        )

        learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        elapsed = (
            time.time()
            - start_time
        )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        history[
            "train_loss"
        ].append(
            float(train_loss)
        )

        history[
            "train_accuracy"
        ].append(
            float(train_accuracy)
        )

        history[
            "val_loss"
        ].append(
            float(val_loss)
        )

        history[
            "val_accuracy"
        ].append(
            float(val_accuracy)
        )

        history[
            "learning_rate"
        ].append(
            float(learning_rate)
        )

        history[
            "epoch_time_seconds"
        ].append(
            float(elapsed)
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print()

        print(
            f"Train Loss : {train_loss:.4f}"
        )

        print(
            f"Train Acc  : "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Val Loss   : {val_loss:.4f}"
        )

        print(
            f"Val Acc    : "
            f"{val_accuracy * 100:.2f}%"
        )

        print(
            f"LR         : "
            f"{learning_rate:.8f}"
        )

        print(
            f"Time       : "
            f"{elapsed:.1f} sec"
        )

        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        if (
            val_accuracy
            > best_accuracy + MIN_DELTA
        ):

            best_accuracy = val_accuracy

            best_loss = val_loss

            best_epoch = epoch

            patience = 0

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                val_loss,
                val_accuracy,
                class_names
            )

            print()

            print(
                "✓ BEST PROPOSED MODEL SAVED"
            )

            print(
                f"Best validation accuracy: "
                f"{best_accuracy * 100:.2f}%"
            )

        else:

            patience += 1

            print()

            print(
                f"No improvement: "
                f"{patience}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        save_history(
            history
        )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            patience
            >= EARLY_STOPPING_PATIENCE
        ):

            print()

            print(
                "Early stopping triggered."
            )

            break

    return (
        history,
        best_accuracy,
        best_loss,
        best_epoch
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(
        RANDOM_SEED
    )

    print()
    print("=" * 80)
    print("SKIN DISEASE CLASSIFICATION")
    print("PROPOSED ATTENTION-ENHANCED EFFICIENTNET-B0")
    print("=" * 80)

    print()

    print(
        f"Device: {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            "GPU:"
        )

        print(
            torch.cuda.get_device_name(0)
        )

    else:

        print(
            "CPU mode enabled."
        )

        print(
            "The EfficientNet backbone is frozen."
        )

        print(
            "Only attention modules and classifier "
            "will be trained."
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    check_dataset()

    (
        train_dataset,
        val_dataset,
        test_dataset,
        class_names
    ) = load_datasets()

    save_class_names(
        class_names
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    (
        train_loader,
        val_loader,
        test_loader
    ) = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset
    )

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    class_weights = (
        calculate_class_weights(
            train_dataset
        )
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model(
        num_classes=len(
            class_names
        )
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    trainable_parameters = filter(
        lambda parameter:
            parameter.requires_grad,
        model.parameters()
    )

    optimizer = optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = (
        optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2
        )
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    (
        history,
        best_accuracy,
        best_loss,
        best_epoch
    ) = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        class_names
    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    save_history(
        history
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    plot_history(
        history
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("PROPOSED MODEL TRAINING COMPLETED")
    print("=" * 80)

    print()

    print(
        f"Best epoch               : "
        f"{best_epoch}"
    )

    print(
        f"Best validation accuracy : "
        f"{best_accuracy * 100:.2f}%"
    )

    print(
        f"Best validation loss     : "
        f"{best_loss:.4f}"
    )

    print()

    print(
        "Best checkpoint:"
    )

    print(
        BEST_MODEL_FILE
    )

    print()

    print(
        "Training history:"
    )

    print(
        HISTORY_FILE
    )

    print()

    print(
        "Training figures:"
    )

    print(
        FIGURES_DIR
        / "proposed_training_accuracy.png"
    )

    print(
        FIGURES_DIR
        / "proposed_training_loss.png"
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
        print("=" * 80)
        print("TRAINING STOPPED BY USER")
        print("=" * 80)

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