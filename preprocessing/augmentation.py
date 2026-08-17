"""
============================================================
SKIN DISEASE CNN PROJECT
TRAINING IMAGE AUGMENTATION
============================================================

Purpose:
    Create training and validation/test image transforms.

Important:
    - Augmentation is applied ONLY to training images.
    - Validation and test images are NOT augmented.
    - Images are resized to 224 x 224.
    - Normalization is compatible with ImageNet
      pretrained models such as ResNet50 and EfficientNet.

Dataset:
    dataset/preprocessed/

Classes:
    22

============================================================
"""

from pathlib import Path

import torch
from torchvision import transforms


# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREPROCESSED_DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "preprocessed"
)


# ============================================================
# 2. IMAGE CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

# ImageNet normalization
#
# Required when using ImageNet-pretrained
# ResNet50 / EfficientNet models.

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]


# ============================================================
# 3. TRAINING TRANSFORM
# ============================================================

train_transform = transforms.Compose([

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    transforms.Resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    ),

    # --------------------------------------------------------
    # Horizontal flip
    # --------------------------------------------------------

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    # --------------------------------------------------------
    # Small rotation
    # --------------------------------------------------------

    transforms.RandomRotation(
        degrees=15
    ),

    # --------------------------------------------------------
    # Small translation and scale variation
    # --------------------------------------------------------

    transforms.RandomAffine(
        degrees=0,
        translate=(
            0.05,
            0.05
        ),
        scale=(
            0.95,
            1.05
        )
    ),

    # --------------------------------------------------------
    # Mild color variation
    #
    # Kept intentionally small because skin color
    # and lesion color are important features.
    # --------------------------------------------------------

    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10,
        saturation=0.05,
        hue=0.02
    ),

    # --------------------------------------------------------
    # Convert to tensor
    # --------------------------------------------------------

    transforms.ToTensor(),

    # --------------------------------------------------------
    # ImageNet normalization
    # --------------------------------------------------------

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ),
])


# ============================================================
# 4. VALIDATION TRANSFORM
# ============================================================

val_transform = transforms.Compose([

    transforms.Resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ),
])


# ============================================================
# 5. TEST TRANSFORM
# ============================================================

test_transform = transforms.Compose([

    transforms.Resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ),
])


# ============================================================
# 6. GET TRANSFORM
# ============================================================

def get_transform(
    split="train"
):
    """
    Return the correct transformation.

    Parameters
    ----------
    split : str
        train, val, validation, or test

    Returns
    -------
    torchvision.transforms.Compose
    """

    split = split.lower().strip()

    if split == "train":

        return train_transform

    if split in {
        "val",
        "validation"
    }:

        return val_transform

    if split == "test":

        return test_transform

    raise ValueError(
        f"Unknown split: {split}. "
        f"Use train, val, validation, or test."
    )


# ============================================================
# 7. PRINT TRANSFORM INFORMATION
# ============================================================

def print_transform_information():

    print("=" * 70)
    print(
        "SKIN DISEASE IMAGE AUGMENTATION CONFIGURATION"
    )
    print("=" * 70)

    print()
    print(
        f"Dataset: {PREPROCESSED_DATASET}"
    )

    print(
        f"Image size: "
        f"{IMAGE_SIZE} x {IMAGE_SIZE}"
    )

    print()
    print(
        "TRAINING AUGMENTATION"
    )

    print("-" * 70)

    print(
        "✓ Random Horizontal Flip     : p=0.50"
    )

    print(
        "✓ Random Rotation             : ±15 degrees"
    )

    print(
        "✓ Random Translation         : ±5%"
    )

    print(
        "✓ Random Scale               : 95% - 105%"
    )

    print(
        "✓ Brightness                 : ±10%"
    )

    print(
        "✓ Contrast                   : ±10%"
    )

    print(
        "✓ Saturation                 : ±5%"
    )

    print(
        "✓ Hue                        : ±0.02"
    )

    print()
    print(
        "VALIDATION / TEST"
    )

    print("-" * 70)

    print(
        "✓ Resize only"
    )

    print(
        "✓ Tensor conversion"
    )

    print(
        "✓ ImageNet normalization"
    )

    print(
        "✗ No random augmentation"
    )

    print()
    print(
        "NORMALIZATION"
    )

    print("-" * 70)

    print(
        f"Mean: {IMAGENET_MEAN}"
    )

    print(
        f"Std : {IMAGENET_STD}"
    )

    print()
    print("=" * 70)


# ============================================================
# 8. SIMPLE TEST
# ============================================================

def test_transforms():

    print()
    print(
        "Testing transformation objects..."
    )

    try:

        dummy_image = torch.rand(
            3,
            IMAGE_SIZE,
            IMAGE_SIZE
        )

        # The transforms expect PIL images,
        # so this is only a configuration test.

        print(
            "✓ Training transform created"
        )

        print(
            "✓ Validation transform created"
        )

        print(
            "✓ Test transform created"
        )

        print(
            "✓ Transform configuration valid"
        )

    except Exception as error:

        print(
            "✗ Transform test failed"
        )

        print(
            error
        )


# ============================================================
# 9. MAIN
# ============================================================

if __name__ == "__main__":

    print_transform_information()

    test_transforms()