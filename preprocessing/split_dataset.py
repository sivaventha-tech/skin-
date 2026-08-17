"""
============================================================
SKIN DISEASE CNN PROJECT
STRATIFIED DATASET SPLITTING
============================================================

Input:
    dataset/cleaned/skin_disease_dataset/

Output:
    dataset/processed/
        ├── train/
        ├── val/
        └── test/

Split:
    Train      = 70%
    Validation = 15%
    Test       = 15%

Features:
    - Reproducible splitting
    - Class-wise stratified split
    - Duplicate leakage verification
    - Class structure verification
    - Split distribution report
============================================================
"""

import sys
import shutil
import hashlib
import random

from pathlib import Path
from collections import defaultdict

import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLEANED_DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "cleaned"
    / "skin_disease_dataset"
)

PROCESSED_DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
)

TRAIN_DIR = PROCESSED_DATASET / "train"
VAL_DIR = PROCESSED_DATASET / "val"
TEST_DIR = PROCESSED_DATASET / "test"

REPORT_DIR = (
    PROJECT_ROOT
    / "results"
    / "reports"
)

SPLIT_REPORT = (
    REPORT_DIR
    / "split_distribution.csv"
)

LEAKAGE_REPORT = (
    REPORT_DIR
    / "split_leakage_report.csv"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


# ============================================================
# 3. PRINT HELPERS
# ============================================================

def print_line(character="=", length=80):
    print(character * length)


def print_header(title):
    print()
    print_line("=")
    print(title)
    print_line("=")


# ============================================================
# 4. MD5 HASH FUNCTION
# ============================================================

def calculate_md5(
    file_path,
    chunk_size=1024 * 1024
):
    """
    Calculate MD5 hash of an image.

    This is used ONLY for duplicate/leakage verification.
    """

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

    except Exception as error:

        print()
        print(
            f"Hash error: {file_path}"
        )

        print(
            f"Reason: {error}"
        )

        return None


# ============================================================
# 5. CHECK RATIOS
# ============================================================

def check_ratios():

    total = (
        TRAIN_RATIO
        + VAL_RATIO
        + TEST_RATIO
    )

    if abs(total - 1.0) > 0.0001:

        print(
            "ERROR: Split ratios must add to 1.0."
        )

        sys.exit(1)


# ============================================================
# 6. CHECK CLEAN DATASET
# ============================================================

def check_clean_dataset():

    print_header(
        "CHECKING CLEAN DATASET"
    )

    print(
        "Input dataset:"
    )

    print(
        CLEANED_DATASET
    )

    if not CLEANED_DATASET.exists():

        print()
        print(
            "ERROR: Cleaned dataset was not found."
        )

        print()
        print(
            "Run:"
        )

        print(
            "python preprocessing\\remove_duplicates.py"
        )

        sys.exit(1)

    class_dirs = [
        item
        for item in CLEANED_DATASET.iterdir()
        if item.is_dir()
    ]

    class_dirs.sort(
        key=lambda path:
        path.name.lower()
    )

    if not class_dirs:

        print(
            "ERROR: No class directories found."
        )

        sys.exit(1)

    print()
    print(
        f"Classes found: {len(class_dirs)}"
    )

    for class_dir in class_dirs:

        print(
            f"  - {class_dir.name}"
        )

    return class_dirs


# ============================================================
# 7. GET IMAGES FOR CLASS
# ============================================================

def get_class_images(
    class_directory
):

    images = []

    for file_path in (
        class_directory.rglob("*")
    ):

        if not file_path.is_file():
            continue

        if (
            file_path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        ):

            images.append(
                file_path
            )

    return sorted(
        images,
        key=lambda path:
        str(path).lower()
    )


# ============================================================
# 8. CALCULATE SPLIT COUNTS
# ============================================================

def calculate_split_counts(
    total
):

    train_count = int(
        total * TRAIN_RATIO
    )

    val_count = int(
        total * VAL_RATIO
    )

    test_count = (
        total
        - train_count
        - val_count
    )

    return (
        train_count,
        val_count,
        test_count
    )


# ============================================================
# 9. PREPARE OUTPUT DIRECTORIES
# ============================================================

def prepare_output_directories():

    print_header(
        "PREPARING PROCESSED DATASET"
    )

    if PROCESSED_DATASET.exists():

        print()
        print(
            "A previous processed dataset exists:"
        )

        print(
            PROCESSED_DATASET
        )

        print()

        response = input(
            "Delete and recreate it? (yes/no): "
        ).strip().lower()

        if response != "yes":

            print()
            print(
                "Operation cancelled."
            )

            sys.exit(0)

        print()
        print(
            "Removing previous processed dataset..."
        )

        shutil.rmtree(
            PROCESSED_DATASET
        )

    TRAIN_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    VAL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    TEST_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 10. SPLIT ONE CLASS
# ============================================================

def split_class_images(
    class_name,
    image_files
):

    files = list(
        image_files
    )

    # Class-specific deterministic seed
    class_seed = (
        RANDOM_SEED
        + sum(
            ord(character)
            for character in class_name
        )
    )

    rng = random.Random(
        class_seed
    )

    rng.shuffle(
        files
    )

    total = len(files)

    (
        train_count,
        val_count,
        test_count
    ) = calculate_split_counts(
        total
    )

    train_files = files[
        :train_count
    ]

    val_files = files[
        train_count:
        train_count + val_count
    ]

    test_files = files[
        train_count + val_count:
    ]

    if (
        len(train_files)
        + len(val_files)
        + len(test_files)
        != total
    ):

        raise RuntimeError(
            f"Split mismatch for class: "
            f"{class_name}"
        )

    return (
        train_files,
        val_files,
        test_files
    )


# ============================================================
# 11. COPY FILES
# ============================================================

def copy_files(
    files,
    destination_directory
):

    destination_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    for source_file in files:

        destination_file = (
            destination_directory
            / source_file.name
        )

        if destination_file.exists():

            # This should normally never happen
            # because the cleaned dataset uses
            # hash-based filenames.

            source_hash = calculate_md5(
                source_file
            )

            destination_hash = calculate_md5(
                destination_file
            )

            if source_hash == destination_hash:

                continue

            counter = 1

            while destination_file.exists():

                destination_file = (
                    destination_directory
                    / f"{source_file.stem}_"
                    f"{counter}"
                    f"{source_file.suffix}"
                )

                counter += 1

        shutil.copy2(
            source_file,
            destination_file
        )


# ============================================================
# 12. CREATE SPLITS
# ============================================================

def create_splits(
    class_dirs
):

    print_header(
        "CREATING TRAIN / VALIDATION / TEST SPLITS"
    )

    print()
    print(
        f"Train ratio      : {TRAIN_RATIO:.0%}"
    )

    print(
        f"Validation ratio : {VAL_RATIO:.0%}"
    )

    print(
        f"Test ratio       : {TEST_RATIO:.0%}"
    )

    print(
        f"Random seed      : {RANDOM_SEED}"
    )

    print()

    records = []

    total_train = 0
    total_val = 0
    total_test = 0

    for class_index, class_dir in enumerate(
        class_dirs,
        start=1
    ):

        class_name = class_dir.name

        images = get_class_images(
            class_dir
        )

        if not images:

            print(
                f"WARNING: No images found "
                f"for {class_name}"
            )

            continue

        (
            train_files,
            val_files,
            test_files
        ) = split_class_images(
            class_name,
            images
        )

        # Destination directories

        train_class_dir = (
            TRAIN_DIR
            / class_name
        )

        val_class_dir = (
            VAL_DIR
            / class_name
        )

        test_class_dir = (
            TEST_DIR
            / class_name
        )

        # Copy files

        copy_files(
            train_files,
            train_class_dir
        )

        copy_files(
            val_files,
            val_class_dir
        )

        copy_files(
            test_files,
            test_class_dir
        )

        # Counts

        train_count = len(
            train_files
        )

        val_count = len(
            val_files
        )

        test_count = len(
            test_files
        )

        total_count = (
            train_count
            + val_count
            + test_count
        )

        total_train += train_count
        total_val += val_count
        total_test += test_count

        print(
            f"{class_index:>2}. "
            f"{class_name:<30}"
            f"Train: {train_count:>5,}  "
            f"Val: {val_count:>5,}  "
            f"Test: {test_count:>5,}  "
            f"Total: {total_count:>5,}"
        )

        records.append(
            {
                "class": class_name,
                "total": total_count,
                "train": train_count,
                "validation": val_count,
                "test": test_count,
                "train_percent": round(
                    train_count
                    / total_count
                    * 100,
                    2
                ),
                "validation_percent": round(
                    val_count
                    / total_count
                    * 100,
                    2
                ),
                "test_percent": round(
                    test_count
                    / total_count
                    * 100,
                    2
                ),
            }
        )

    print_line("-")

    print(
        f"{'TOTAL':<32}"
        f"Train: {total_train:>5,}  "
        f"Val: {total_val:>5,}  "
        f"Test: {total_test:>5,}  "
        f"Total: "
        f"{total_train + total_val + total_test:>5,}"
    )

    return records


# ============================================================
# 13. SAVE SPLIT REPORT
# ============================================================

def save_split_report(
    records
):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    dataframe = pd.DataFrame(
        records
    )

    dataframe.to_csv(
        SPLIT_REPORT,
        index=False,
        encoding="utf-8"
    )

    print()
    print(
        "Split distribution report saved:"
    )

    print(
        SPLIT_REPORT
    )


# ============================================================
# 14. GET HASHES FOR ONE SPLIT
# ============================================================

def get_split_hashes(
    split_directory,
    split_name
):

    hashes = defaultdict(list)

    if not split_directory.exists():

        return hashes

    image_files = []

    for file_path in (
        split_directory.rglob("*")
    ):

        if not file_path.is_file():
            continue

        if (
            file_path.suffix.lower()
            not in SUPPORTED_EXTENSIONS
        ):

            continue

        image_files.append(
            file_path
        )

    image_files.sort(
        key=lambda path:
        str(path).lower()
    )

    print(
        f"  Checking {split_name}: "
        f"{len(image_files):,} images"
    )

    for file_path in image_files:

        image_hash = calculate_md5(
            file_path
        )

        if image_hash:

            hashes[
                image_hash
            ].append(
                {
                    "split": split_name,
                    "class": file_path.parent.name,
                    "file": str(file_path)
                }
            )

    return hashes


# ============================================================
# 15. VERIFY LEAKAGE
# ============================================================

def verify_split_leakage():

    print_header(
        "VERIFYING DATA LEAKAGE"
    )

    print(
        "Calculating hashes for "
        "Train / Validation / Test..."
    )

    print()

    train_hashes = get_split_hashes(
        TRAIN_DIR,
        "train"
    )

    val_hashes = get_split_hashes(
        VAL_DIR,
        "validation"
    )

    test_hashes = get_split_hashes(
        TEST_DIR,
        "test"
    )

    train_set = set(
        train_hashes.keys()
    )

    val_set = set(
        val_hashes.keys()
    )

    test_set = set(
        test_hashes.keys()
    )

    # --------------------------------------------------------
    # Overlap checks
    # --------------------------------------------------------

    train_val = (
        train_set
        & val_set
    )

    train_test = (
        train_set
        & test_set
    )

    val_test = (
        val_set
        & test_set
    )

    all_leakage_hashes = (
        train_val
        | train_test
        | val_test
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()

    print(
        f"Train images      : "
        f"{len(train_set):,}"
    )

    print(
        f"Validation images : "
        f"{len(val_set):,}"
    )

    print(
        f"Test images       : "
        f"{len(test_set):,}"
    )

    print()

    print(
        f"Train ↔ Validation duplicates : "
        f"{len(train_val):,}"
    )

    print(
        f"Train ↔ Test duplicates       : "
        f"{len(train_test):,}"
    )

    print(
        f"Validation ↔ Test duplicates  : "
        f"{len(val_test):,}"
    )

    # --------------------------------------------------------
    # Save leakage report
    # --------------------------------------------------------

    leakage_records = []

    for image_hash in sorted(
        all_leakage_hashes
    ):

        entries = []

        entries.extend(
            train_hashes.get(
                image_hash,
                []
            )
        )

        entries.extend(
            val_hashes.get(
                image_hash,
                []
            )
        )

        entries.extend(
            test_hashes.get(
                image_hash,
                []
            )
        )

        for item in entries:

            leakage_records.append(
                {
                    "md5": image_hash,
                    "split": item["split"],
                    "class": item["class"],
                    "file": item["file"]
                }
            )

    pd.DataFrame(
        leakage_records,
        columns=[
            "md5",
            "split",
            "class",
            "file"
        ]
    ).to_csv(
        LEAKAGE_REPORT,
        index=False,
        encoding="utf-8"
    )

    print()
    print(
        "Leakage report saved:"
    )

    print(
        LEAKAGE_REPORT
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    if all_leakage_hashes:

        print()
        print_line("!")

        print(
            "WARNING: DATA LEAKAGE DETECTED"
        )

        print_line("!")

        return False

    print()
    print_line("=")

    print(
        "DATA LEAKAGE VERIFICATION PASSED"
    )

    print_line("=")

    print(
        "✓ Train/Validation overlap: 0"
    )

    print(
        "✓ Train/Test overlap: 0"
    )

    print(
        "✓ Validation/Test overlap: 0"
    )

    return True


# ============================================================
# 16. VERIFY CLASS STRUCTURE
# ============================================================

def verify_all_classes(
    expected_classes
):

    print_header(
        "VERIFYING CLASS STRUCTURE"
    )

    success = True

    expected_set = set(
        expected_classes
    )

    for split_name, split_dir in [
        ("train", TRAIN_DIR),
        ("validation", VAL_DIR),
        ("test", TEST_DIR),
    ]:

        classes = {
            item.name
            for item in split_dir.iterdir()
            if item.is_dir()
        }

        missing = (
            expected_set
            - classes
        )

        extra = (
            classes
            - expected_set
        )

        print()
        print(
            f"{split_name.upper()}:"
        )

        print(
            f"  Classes found: "
            f"{len(classes)}"
        )

        if missing:

            success = False

            print(
                "  Missing classes:"
            )

            for class_name in sorted(
                missing
            ):

                print(
                    f"    - {class_name}"
                )

        else:

            print(
                "  ✓ All classes present"
            )

        if extra:

            success = False

            print(
                "  Unexpected classes:"
            )

            for class_name in sorted(
                extra
            ):

                print(
                    f"    - {class_name}"
                )

    return success


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

def final_summary():

    print_header(
        "FINAL DATASET SUMMARY"
    )

    split_counts = {}

    for split_name, split_dir in [
        ("Train", TRAIN_DIR),
        ("Validation", VAL_DIR),
        ("Test", TEST_DIR),
    ]:

        count = 0

        for file_path in (
            split_dir.rglob("*")
        ):

            if not file_path.is_file():
                continue

            if (
                file_path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            ):

                count += 1

        split_counts[
            split_name
        ] = count

    total = sum(
        split_counts.values()
    )

    print(
        f"Train images      : "
        f"{split_counts['Train']:,}"
    )

    print(
        f"Validation images : "
        f"{split_counts['Validation']:,}"
    )

    print(
        f"Test images       : "
        f"{split_counts['Test']:,}"
    )

    print(
        f"Total images      : "
        f"{total:,}"
    )

    print()

    print(
        "Output:"
    )

    print(
        PROCESSED_DATASET
    )

    return split_counts


# ============================================================
# 18. MAIN
# ============================================================

def main():

    check_ratios()

    class_dirs = check_clean_dataset()

    expected_classes = [
        class_dir.name
        for class_dir in class_dirs
    ]

    prepare_output_directories()

    records = create_splits(
        class_dirs
    )

    save_split_report(
        records
    )

    class_check = verify_all_classes(
        expected_classes
    )

    leakage_check = verify_split_leakage()

    split_counts = final_summary()

    print()
    print_line("=")

    if (
        class_check
        and leakage_check
    ):

        print(
            "DATASET SPLITTING COMPLETED SUCCESSFULLY"
        )

        print_line("=")

        print()
        print(
            "✓ Stratified class-wise split"
        )

        print(
            "✓ 70% Train"
        )

        print(
            "✓ 15% Validation"
        )

        print(
            "✓ 15% Test"
        )

        print(
            "✓ All 22 classes present"
        )

        print(
            "✓ No duplicate leakage"
        )

        print()
        print(
            "Processed dataset:"
        )

        print(
            PROCESSED_DATASET
        )

    else:

        print(
            "DATASET SPLITTING COMPLETED "
            "WITH WARNINGS"
        )

        if not class_check:

            print(
                "✗ Class structure verification failed."
            )

        if not leakage_check:

            print(
                "✗ Duplicate leakage detected."
            )

        print()
        print(
            "DO NOT TRAIN THE MODEL YET."
        )

    print()


# ============================================================
# 19. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "Operation cancelled by user."
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

        sys.exit(1)