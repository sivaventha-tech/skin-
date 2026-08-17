"""
============================================================
SKIN DISEASE CNN PROJECT
Dataset Inspection and Validation
============================================================

Dataset structure expected:

dataset/
└── raw/
    └── skin_disease_dataset/
        └── SkinDisease/
            └── SkinDisease/
                ├── train/
                │   ├── Acne/
                │   ├── Actinic_Keratosis/
                │   ├── ...
                │   └── Warts/
                │
                └── test/
                    ├── Acne/
                    ├── Actinic_Keratosis/
                    ├── ...
                    └── Warts/

This script:

1. Detects train/test folders
2. Detects all disease classes
3. Counts images per class
4. Checks corrupted images
5. Checks unsupported files
6. Detects exact duplicate images
7. Detects cross-label duplicates
8. Checks train/test leakage
9. Analyzes image dimensions
10. Analyzes color modes
11. Generates class distribution graph
12. Saves class_names.json
13. Saves detailed CSV reports

============================================================
"""

import sys
import json
import hashlib
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, UnidentifiedImageError


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "skin_disease_dataset"
)

# Actual Kaggle structure
DATASET_BASE = DATASET_ROOT / "SkinDisease" / "SkinDisease"

TRAIN_DIR = DATASET_BASE / "train"
TEST_DIR = DATASET_BASE / "test"

METADATA_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "metadata"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
)

REPORTS_DIR = (
    RESULTS_DIR
    / "reports"
)


# Create output directories
METADATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. SUPPORTED IMAGE EXTENSIONS
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff"
}


# ============================================================
# 3. DISPLAY FUNCTIONS
# ============================================================

def print_line(character="=", length=80):
    print(character * length)


def print_header(title):

    print()
    print_line("=")
    print(title)
    print_line("=")


# ============================================================
# 4. CHECK DATASET STRUCTURE
# ============================================================

def check_dataset_structure():

    print_header(
        "SKIN DISEASE CNN - DATASET INSPECTION"
    )

    print("Project Root:")
    print(PROJECT_ROOT)

    print()

    print("Dataset Root:")
    print(DATASET_ROOT)

    print()

    print("Dataset Base:")
    print(DATASET_BASE)

    print()

    # Check root
    if not DATASET_ROOT.exists():

        print()
        print("ERROR: Dataset root does not exist.")
        print()

        print(
            "Expected:"
        )

        print(DATASET_ROOT)

        sys.exit(1)

    # Check actual dataset base
    if not DATASET_BASE.exists():

        print()
        print(
            "ERROR: Expected nested SkinDisease directory "
            "was not found."
        )

        print()

        print(
            "Expected structure:"
        )

        print(
            "dataset/raw/skin_disease_dataset/"
        )

        print(
            "└── SkinDisease/"
        )

        print(
            "    └── SkinDisease/"
        )

        print(
            "        ├── train/"
        )

        print(
            "        └── test/"
        )

        sys.exit(1)

    # Check train
    if not TRAIN_DIR.exists():

        print()
        print(
            "ERROR: Train directory not found:"
        )

        print(TRAIN_DIR)

        sys.exit(1)

    # Check test
    if not TEST_DIR.exists():

        print()
        print(
            "ERROR: Test directory not found:"
        )

        print(TEST_DIR)

        sys.exit(1)

    print()
    print("Dataset structure verified successfully.")

    print()

    print("Train Directory:")
    print(TRAIN_DIR)

    print()

    print("Test Directory:")
    print(TEST_DIR)


# ============================================================
# 5. FIND CLASS DIRECTORIES
# ============================================================

def get_class_directories():

    train_classes = {
        item.name
        for item in TRAIN_DIR.iterdir()
        if item.is_dir()
    }

    test_classes = {
        item.name
        for item in TEST_DIR.iterdir()
        if item.is_dir()
    }

    all_classes = sorted(
        train_classes | test_classes,
        key=lambda x: x.lower()
    )

    train_only = sorted(
        train_classes - test_classes
    )

    test_only = sorted(
        test_classes - train_classes
    )

    if train_only:

        print()
        print(
            "WARNING: Classes found only in TRAIN:"
        )

        for class_name in train_only:
            print("  -", class_name)

    if test_only:

        print()
        print(
            "WARNING: Classes found only in TEST:"
        )

        for class_name in test_only:
            print("  -", class_name)

    return all_classes


# ============================================================
# 6. FIND IMAGE FILES
# ============================================================

def get_image_files(directory):

    image_files = []

    for file_path in directory.rglob("*"):

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:

            image_files.append(file_path)

    return sorted(image_files)


# ============================================================
# 7. CALCULATE MD5 HASH
# ============================================================

def calculate_md5(
    file_path,
    chunk_size=1024 * 1024
):

    md5 = hashlib.md5()

    try:

        with open(
            file_path,
            "rb"
        ) as file:

            while True:

                chunk = file.read(
                    chunk_size
                )

                if not chunk:
                    break

                md5.update(chunk)

        return md5.hexdigest()

    except Exception:

        return None


# ============================================================
# 8. CHECK IMAGE VALIDITY
# ============================================================

def validate_image(file_path):

    try:

        with Image.open(
            file_path
        ) as image:

            image.verify()

        return True

    except (
        UnidentifiedImageError,
        OSError,
        ValueError
    ):

        return False

    except Exception:

        return False


# ============================================================
# 9. GET IMAGE INFORMATION
# ============================================================

def get_image_info(file_path):

    try:

        with Image.open(
            file_path
        ) as image:

            width, height = image.size

            mode = image.mode

            format_name = image.format

        return (
            width,
            height,
            mode,
            format_name
        )

    except Exception:

        return (
            None,
            None,
            None,
            None
        )


# ============================================================
# 10. ANALYZE SPLIT
# ============================================================

def analyze_split(
    split_name,
    split_directory,
    class_names,
    records,
    hash_records,
    corrupted_files,
    unsupported_files,
    dimensions,
    color_modes
):

    print()
    print_line("-")

    print(
        f"{split_name.upper()} DATASET"
    )

    print_line("-")

    split_counts = {}

    for class_name in class_names:

        class_directory = (
            split_directory
            / class_name
        )

        if not class_directory.exists():

            split_counts[class_name] = 0

            continue

        all_files = list(
            class_directory.rglob("*")
        )

        image_files = []

        for file_path in all_files:

            if not file_path.is_file():
                continue

            extension = (
                file_path.suffix.lower()
            )

            if extension in SUPPORTED_EXTENSIONS:

                image_files.append(
                    file_path
                )

            else:

                # Ignore hidden/system files
                if not file_path.name.startswith("."):

                    unsupported_files.append(
                        {
                            "split": split_name,
                            "class": class_name,
                            "file": str(file_path)
                        }
                    )

        image_files.sort()

        split_counts[class_name] = (
            len(image_files)
        )

        for image_path in image_files:

            valid = validate_image(
                image_path
            )

            width = None
            height = None
            mode = None
            image_format = None
            image_hash = None

            if not valid:

                corrupted_files.append(
                    {
                        "split": split_name,
                        "class": class_name,
                        "file": str(image_path)
                    }
                )

            else:

                (
                    width,
                    height,
                    mode,
                    image_format
                ) = get_image_info(
                    image_path
                )

                if width and height:

                    dimensions[
                        (width, height)
                    ] += 1

                if mode:

                    color_modes[
                        mode
                    ] += 1

                image_hash = (
                    calculate_md5(
                        image_path
                    )
                )

                if image_hash:

                    hash_records[
                        image_hash
                    ].append(
                        {
                            "split": split_name,
                            "class": class_name,
                            "file": str(image_path)
                        }
                    )

            records.append(
                {
                    "split": split_name,
                    "class": class_name,
                    "file": str(image_path),
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "format": image_format,
                    "valid": valid,
                    "md5": image_hash
                }
            )

    # Print class counts

    print()

    print(
        f"{'No.':<5}"
        f"{'Class':<30}"
        f"{'Images':>12}"
    )

    print_line("-")

    for index, class_name in enumerate(
        class_names,
        start=1
    ):

        count = split_counts.get(
            class_name,
            0
        )

        print(
            f"{index:<5}"
            f"{class_name:<30}"
            f"{count:>12,}"
        )

    print_line("-")

    total = sum(
        split_counts.values()
    )

    print(
        f"{'TOTAL':<35}"
        f"{total:>12,}"
    )

    return split_counts


# ============================================================
# 11. DUPLICATE ANALYSIS
# ============================================================

def analyze_duplicates(
    hash_records
):

    duplicate_groups = {}

    for image_hash, files in (
        hash_records.items()
    ):

        if len(files) > 1:

            duplicate_groups[
                image_hash
            ] = files

    same_label_groups = {}

    cross_label_groups = {}

    train_test_groups = {}

    for image_hash, files in (
        duplicate_groups.items()
    ):

        labels = {
            item["class"]
            for item in files
        }

        splits = {
            item["split"]
            for item in files
        }

        if len(labels) == 1:

            same_label_groups[
                image_hash
            ] = files

        else:

            cross_label_groups[
                image_hash
            ] = files

        if (
            "train" in splits
            and "test" in splits
        ):

            train_test_groups[
                image_hash
            ] = files

    return (
        duplicate_groups,
        same_label_groups,
        cross_label_groups,
        train_test_groups
    )


# ============================================================
# 12. PRINT DUPLICATE INFORMATION
# ============================================================

def print_duplicate_summary(
    duplicate_groups,
    same_label_groups,
    cross_label_groups,
    train_test_groups
):

    print_header(
        "DUPLICATE IMAGE ANALYSIS"
    )

    print(
        f"Duplicate groups              : "
        f"{len(duplicate_groups):,}"
    )

    print(
        f"Same-label duplicate groups   : "
        f"{len(same_label_groups):,}"
    )

    print(
        f"Cross-label duplicate groups : "
        f"{len(cross_label_groups):,}"
    )

    print(
        f"Train/Test leakage groups     : "
        f"{len(train_test_groups):,}"
    )

    print_line()

    # --------------------------------------------------------
    # Cross-label duplicates
    # --------------------------------------------------------

    if cross_label_groups:

        print()
        print(
            "⚠ CROSS-LABEL DUPLICATES"
        )

        print_line("-")

        print(
            "The same image hash appears "
            "under different disease labels."
        )

        print()

        shown = 0

        for image_hash, files in (
            cross_label_groups.items()
        ):

            print(
                f"Group {shown + 1}"
            )

            print(
                f"MD5: {image_hash}"
            )

            for item in files:

                print(
                    f"  [{item['split']}] "
                    f"[{item['class']}] "
                    f"{item['file']}"
                )

            print()

            shown += 1

            if shown >= 15:

                break

        if len(cross_label_groups) > 15:

            print(
                f"... and "
                f"{len(cross_label_groups) - 15} "
                f"more groups."
            )

    else:

        print()
        print(
            "✓ No cross-label duplicates detected."
        )

    # --------------------------------------------------------
    # Train/test leakage
    # --------------------------------------------------------

    print()

    if train_test_groups:

        print(
            "⚠ TRAIN/TEST DUPLICATE LEAKAGE DETECTED"
        )

        print_line("-")

        print(
            f"{len(train_test_groups):,} "
            "duplicate groups appear in both "
            "train and test."
        )

    else:

        print(
            "✓ No exact train/test duplicate leakage detected."
        )

    print_line("-")


# ============================================================
# 13. CLASS DISTRIBUTION
# ============================================================

def create_class_distribution(
    class_names,
    train_counts,
    test_counts
):

    train_values = [
        train_counts.get(
            class_name,
            0
        )
        for class_name in class_names
    ]

    test_values = [
        test_counts.get(
            class_name,
            0
        )
        for class_name in class_names
    ]

    x = range(
        len(class_names)
    )

    width = 0.4

    plt.figure(
        figsize=(16, 9)
    )

    train_positions = [
        value - width / 2
        for value in x
    ]

    test_positions = [
        value + width / 2
        for value in x
    ]

    plt.bar(
        train_positions,
        train_values,
        width=width,
        label="Train"
    )

    plt.bar(
        test_positions,
        test_values,
        width=width,
        label="Test"
    )

    plt.xticks(
        list(x),
        class_names,
        rotation=65,
        ha="right"
    )

    plt.xlabel(
        "Skin Disease Class"
    )

    plt.ylabel(
        "Number of Images"
    )

    plt.title(
        "Skin Disease Dataset Class Distribution"
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        FIGURES_DIR
        / "class_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print(
        "Class distribution graph saved:"
    )

    print(output_path)


# ============================================================
# 14. SAVE CLASS NAMES
# ============================================================

def save_class_names(
    class_names
):

    output_path = (
        METADATA_DIR
        / "class_names.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            class_names,
            file,
            indent=4,
            ensure_ascii=False
        )

    print()
    print(
        "Class names saved:"
    )

    print(output_path)


# ============================================================
# 15. SAVE REPORTS
# ============================================================

def save_reports(
    records,
    corrupted_files,
    unsupported_files,
    duplicate_groups,
    cross_label_groups,
    train_test_groups
):

    # --------------------------------------------------------
    # Complete image report
    # --------------------------------------------------------

    dataframe = pd.DataFrame(
        records
    )

    image_report_path = (
        REPORTS_DIR
        / "dataset_image_report.csv"
    )

    dataframe.to_csv(
        image_report_path,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Corrupted images
    # --------------------------------------------------------

    corrupted_path = (
        REPORTS_DIR
        / "corrupted_images.csv"
    )

    pd.DataFrame(
        corrupted_files
    ).to_csv(
        corrupted_path,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Unsupported files
    # --------------------------------------------------------

    unsupported_path = (
        REPORTS_DIR
        / "unsupported_files.csv"
    )

    pd.DataFrame(
        unsupported_files
    ).to_csv(
        unsupported_path,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Duplicate groups
    # --------------------------------------------------------

    duplicate_records = []

    for image_hash, files in (
        duplicate_groups.items()
    ):

        for item in files:

            duplicate_records.append(
                {
                    "md5": image_hash,
                    "split": item["split"],
                    "class": item["class"],
                    "file": item["file"]
                }
            )

    duplicate_path = (
        REPORTS_DIR
        / "duplicate_images.csv"
    )

    pd.DataFrame(
        duplicate_records
    ).to_csv(
        duplicate_path,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Cross-label duplicates
    # --------------------------------------------------------

    cross_label_records = []

    for image_hash, files in (
        cross_label_groups.items()
    ):

        for item in files:

            cross_label_records.append(
                {
                    "md5": image_hash,
                    "split": item["split"],
                    "class": item["class"],
                    "file": item["file"]
                }
            )

    cross_label_path = (
        REPORTS_DIR
        / "cross_label_duplicates.csv"
    )

    pd.DataFrame(
        cross_label_records
    ).to_csv(
        cross_label_path,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Train/test leakage
    # --------------------------------------------------------

    leakage_records = []

    for image_hash, files in (
        train_test_groups.items()
    ):

        for item in files:

            leakage_records.append(
                {
                    "md5": image_hash,
                    "split": item["split"],
                    "class": item["class"],
                    "file": item["file"]
                }
            )

    leakage_path = (
        REPORTS_DIR
        / "train_test_leakage.csv"
    )

    pd.DataFrame(
        leakage_records
    ).to_csv(
        leakage_path,
        index=False,
        encoding="utf-8"
    )

    print()
    print(
        "Reports saved successfully."
    )

    print(
        image_report_path
    )

    print(
        duplicate_path
    )

    print(
        cross_label_path
    )

    print(
        leakage_path
    )


# ============================================================
# 16. IMAGE DIMENSION SUMMARY
# ============================================================

def print_dimension_summary(
    dimensions
):

    print_header(
        "IMAGE DIMENSIONS"
    )

    if not dimensions:

        print(
            "No valid image dimensions found."
        )

        return

    for (
        width,
        height
    ), count in dimensions.most_common(20):

        print(
            f"{width} x {height:<10} : "
            f"{count:,} images"
        )


# ============================================================
# 17. COLOR MODE SUMMARY
# ============================================================

def print_color_mode_summary(
    color_modes
):

    print_header(
        "IMAGE COLOR MODES"
    )

    for mode, count in (
        color_modes.most_common()
    ):

        print(
            f"{mode:<10} : "
            f"{count:,} images"
        )


# ============================================================
# 18. CLASS SUMMARY
# ============================================================

def print_class_summary(
    class_names,
    train_counts,
    test_counts
):

    print_header(
        "CLASS DISTRIBUTION SUMMARY"
    )

    print(
        f"{'No.':<5}"
        f"{'Class':<30}"
        f"{'Train':>10}"
        f"{'Test':>10}"
        f"{'Total':>10}"
    )

    print_line("-")

    total_train = 0
    total_test = 0

    for index, class_name in enumerate(
        class_names,
        start=1
    ):

        train_count = (
            train_counts.get(
                class_name,
                0
            )
        )

        test_count = (
            test_counts.get(
                class_name,
                0
            )
        )

        total_count = (
            train_count
            + test_count
        )

        total_train += train_count
        total_test += test_count

        print(
            f"{index:<5}"
            f"{class_name:<30}"
            f"{train_count:>10,}"
            f"{test_count:>10,}"
            f"{total_count:>10,}"
        )

    print_line("-")

    print(
        f"{'TOTAL':<35}"
        f"{total_train:>10,}"
        f"{total_test:>10,}"
        f"{total_train + total_test:>10,}"
    )


# ============================================================
# 19. MAIN ANALYSIS
# ============================================================

def main():

    # --------------------------------------------------------
    # Check structure
    # --------------------------------------------------------

    check_dataset_structure()

    # --------------------------------------------------------
    # Find classes
    # --------------------------------------------------------

    class_names = (
        get_class_directories()
    )

    if not class_names:

        print()
        print(
            "ERROR: No class directories found."
        )

        sys.exit(1)

    print()

    print(
        f"Classes detected: "
        f"{len(class_names)}"
    )

    # --------------------------------------------------------
    # Expected classes
    # --------------------------------------------------------

    expected_classes = {
        "Acne",
        "Actinic_Keratosis",
        "Benign_tumors",
        "Bullous",
        "Candidiasis",
        "DrugEruption",
        "Eczema",
        "Infestations_Bites",
        "Lichen",
        "Lupus",
        "Moles",
        "Psoriasis",
        "Rosacea",
        "Seborrh_Keratoses",
        "SkinCancer",
        "Sun_Sunlight_Damage",
        "Tinea",
        "Unknown_Normal",
        "Vascular_Tumors",
        "Vasculitis",
        "Vitiligo",
        "Warts"
    }

    detected_classes = set(
        class_names
    )

    missing_classes = (
        expected_classes
        - detected_classes
    )

    unexpected_classes = (
        detected_classes
        - expected_classes
    )

    if missing_classes:

        print()

        print(
            "WARNING: Expected classes missing:"
        )

        for class_name in sorted(
            missing_classes
        ):

            print(
                "  -",
                class_name
            )

    if unexpected_classes:

        print()

        print(
            "WARNING: Additional classes detected:"
        )

        for class_name in sorted(
            unexpected_classes
        ):

            print(
                "  -",
                class_name
            )

    # --------------------------------------------------------
    # Data containers
    # --------------------------------------------------------

    records = []

    hash_records = defaultdict(
        list
    )

    corrupted_files = []

    unsupported_files = []

    dimensions = Counter()

    color_modes = Counter()

    # --------------------------------------------------------
    # Analyze train
    # --------------------------------------------------------

    train_counts = analyze_split(
        "train",
        TRAIN_DIR,
        class_names,
        records,
        hash_records,
        corrupted_files,
        unsupported_files,
        dimensions,
        color_modes
    )

    # --------------------------------------------------------
    # Analyze test
    # --------------------------------------------------------

    test_counts = analyze_split(
        "test",
        TEST_DIR,
        class_names,
        records,
        hash_records,
        corrupted_files,
        unsupported_files,
        dimensions,
        color_modes
    )

    # --------------------------------------------------------
    # Print class summary
    # --------------------------------------------------------

    print_class_summary(
        class_names,
        train_counts,
        test_counts
    )

    # --------------------------------------------------------
    # Total image count
    # --------------------------------------------------------

    total_images = sum(
        train_counts.values()
    ) + sum(
        test_counts.values()
    )

    valid_images = (
        total_images
        - len(corrupted_files)
    )

    print_header(
        "DATASET SUMMARY"
    )

    print(
        f"Number of classes        : "
        f"{len(class_names)}"
    )

    print(
        f"Training images          : "
        f"{sum(train_counts.values()):,}"
    )

    print(
        f"Testing images           : "
        f"{sum(test_counts.values()):,}"
    )

    print(
        f"Total images             : "
        f"{total_images:,}"
    )

    print(
        f"Valid images             : "
        f"{valid_images:,}"
    )

    print(
        f"Corrupted images         : "
        f"{len(corrupted_files):,}"
    )

    print(
        f"Unsupported files        : "
        f"{len(unsupported_files):,}"
    )

    # --------------------------------------------------------
    # Class imbalance
    # --------------------------------------------------------

    all_counts = {
        class_name:
        train_counts.get(
            class_name,
            0
        )
        + test_counts.get(
            class_name,
            0
        )
        for class_name in class_names
    }

    if all_counts:

        largest_class = max(
            all_counts,
            key=all_counts.get
        )

        smallest_class = min(
            all_counts,
            key=all_counts.get
        )

        largest_count = (
            all_counts[
                largest_class
            ]
        )

        smallest_count = (
            all_counts[
                smallest_class
            ]
        )

        print()

        print(
            "Largest class:"
        )

        print(
            f"  {largest_class} "
            f"({largest_count:,})"
        )

        print(
            "Smallest class:"
        )

        print(
            f"  {smallest_class} "
            f"({smallest_count:,})"
        )

        if smallest_count > 0:

            imbalance_ratio = (
                largest_count
                / smallest_count
            )

            print(
                f"Imbalance ratio       : "
                f"{imbalance_ratio:.2f}:1"
            )

    # --------------------------------------------------------
    # Duplicate analysis
    # --------------------------------------------------------

    (
        duplicate_groups,
        same_label_groups,
        cross_label_groups,
        train_test_groups
    ) = analyze_duplicates(
        hash_records
    )

    print()

    print(
        f"Duplicate groups        : "
        f"{len(duplicate_groups):,}"
    )

    print(
        f"Same-label duplicates   : "
        f"{len(same_label_groups):,}"
    )

    print(
        f"Cross-label duplicates : "
        f"{len(cross_label_groups):,}"
    )

    print(
        f"Train/Test leakage      : "
        f"{len(train_test_groups):,}"
    )

    # --------------------------------------------------------
    # Detailed duplicate output
    # --------------------------------------------------------

    print_duplicate_summary(
        duplicate_groups,
        same_label_groups,
        cross_label_groups,
        train_test_groups
    )

    # --------------------------------------------------------
    # Image information
    # --------------------------------------------------------

    print_dimension_summary(
        dimensions
    )

    print_color_mode_summary(
        color_modes
    )

    # --------------------------------------------------------
    # Save class names
    # --------------------------------------------------------

    save_class_names(
        class_names
    )

    # --------------------------------------------------------
    # Create class distribution graph
    # --------------------------------------------------------

    create_class_distribution(
        class_names,
        train_counts,
        test_counts
    )

    # --------------------------------------------------------
    # Save reports
    # --------------------------------------------------------

    save_reports(
        records,
        corrupted_files,
        unsupported_files,
        duplicate_groups,
        cross_label_groups,
        train_test_groups
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print_header(
        "DATASET INSPECTION COMPLETED"
    )

    print(
        "Important:"
    )

    print(
        "DO NOT train the model yet."
    )

    print()

    if cross_label_groups:

        print(
            "WARNING:"
        )

        print(
            f"{len(cross_label_groups):,} "
            "cross-label duplicate groups were found."
        )

        print(
            "These must be resolved before training."
        )

    if train_test_groups:

        print(
            "WARNING:"
        )

        print(
            f"{len(train_test_groups):,} "
            "train/test leakage groups were found."
        )

        print(
            "The existing train/test split should "
            "not be used directly for final evaluation."
        )

    print()

    print(
        "Recommended next step:"
    )

    print(
        "Clean duplicates → resolve conflicts → "
        "create a new stratified train/val/test split."
    )

    print_line("=")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "Process interrupted by user."
        )

        sys.exit(1)

    except Exception as error:

        print()
        print_line("=")

        print(
            "UNEXPECTED ERROR"
        )

        print_line("=")

        print(
            str(error)
        )

        print()

        print(
            "Check your dataset structure and "
            "try again."
        )

        sys.exit(1)