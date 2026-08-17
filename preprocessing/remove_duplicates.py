"""
============================================================
SKIN DISEASE CNN PROJECT
Duplicate and Label-Conflict Cleaning
============================================================

Original dataset:

dataset/
└── raw/
    └── skin_disease_dataset/
        └── SkinDisease/
            └── SkinDisease/
                ├── train/
                └── test/

Cleaned dataset:

dataset/
└── cleaned/
    └── skin_disease_dataset/
        ├── Acne/
        ├── Actinic_Keratosis/
        ├── ...
        └── Warts/

CLEANING POLICY
---------------

1. Unique image
       -> KEEP

2. Exact duplicate with SAME label
       -> KEEP ONE COPY

3. Exact duplicate with DIFFERENT labels
       -> REMOVE ALL COPIES

4. Corrupted image
       -> REMOVE

5. Original Kaggle dataset
       -> NEVER MODIFY

The cleaned dataset will later be split into:

    train/
    val/
    test/

with a stratified split and leakage verification.
============================================================
"""

import sys
import shutil
import hashlib

from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd

from PIL import Image


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------
# ORIGINAL DATASET
# ------------------------------------------------------------

ORIGINAL_DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "skin_disease_dataset"
    / "SkinDisease"
    / "SkinDisease"
)

TRAIN_DIR = ORIGINAL_DATASET / "train"

TEST_DIR = ORIGINAL_DATASET / "test"


# ------------------------------------------------------------
# CLEANED DATASET
# ------------------------------------------------------------

CLEANED_DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "cleaned"
    / "skin_disease_dataset"
)


# ------------------------------------------------------------
# REPORT DIRECTORY
# ------------------------------------------------------------

REPORT_DIR = (
    PROJECT_ROOT
    / "results"
    / "reports"
)


# ============================================================
# 2. REPORT FILES
# ============================================================

CLEANING_REPORT = (
    REPORT_DIR
    / "duplicate_cleaning_report.csv"
)

CONFLICT_REPORT = (
    REPORT_DIR
    / "cross_label_conflict_report.csv"
)

INVALID_REPORT = (
    REPORT_DIR
    / "invalid_images_report.csv"
)

FINAL_CLASS_REPORT = (
    REPORT_DIR
    / "cleaned_class_distribution.csv"
)


# ============================================================
# 3. SUPPORTED IMAGE EXTENSIONS
# ============================================================

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
# 4. DISPLAY FUNCTIONS
# ============================================================

def print_line(
    character="=",
    length=80
):
    print(character * length)


def print_header(title):

    print()
    print_line("=")
    print(title)
    print_line("=")


# ============================================================
# 5. MD5 HASH
# ============================================================

def calculate_md5(
    file_path,
    chunk_size=1024 * 1024
):
    """
    Calculate MD5 hash of an image.

    Exact same files will have the same hash.
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
# 6. IMAGE VALIDATION
# ============================================================

def is_valid_image(file_path):

    try:

        with Image.open(
            file_path
        ) as image:

            image.verify()

        return True

    except Exception:

        return False


# ============================================================
# 7. GET CLASS
# ============================================================

def get_class_name(file_path):

    # Expected:
    #
    # .../train/Acne/image.jpeg
    #
    # parent = Acne

    return file_path.parent.name


# ============================================================
# 8. GET ORIGINAL SPLIT
# ============================================================

def get_split_name(file_path):

    parts = {
        part.lower()
        for part in file_path.parts
    }

    if "train" in parts:

        return "train"

    if "test" in parts:

        return "test"

    return "unknown"


# ============================================================
# 9. CHECK DATASET
# ============================================================

def check_dataset():

    print_header(
        "SKIN DISEASE DATASET - DUPLICATE CLEANING"
    )

    print()
    print("Original dataset:")
    print(ORIGINAL_DATASET)

    print()
    print("Cleaned dataset:")
    print(CLEANED_DATASET)

    print_line()

    # --------------------------------------------------------
    # Check original dataset
    # --------------------------------------------------------

    if not ORIGINAL_DATASET.exists():

        print()
        print(
            "ERROR: Original dataset does not exist."
        )

        print()
        print(
            "Expected:"
        )

        print(
            ORIGINAL_DATASET
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Check train directory
    # --------------------------------------------------------

    if not TRAIN_DIR.exists():

        print()
        print(
            "ERROR: Train directory not found."
        )

        print(
            TRAIN_DIR
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Check test directory
    # --------------------------------------------------------

    if not TEST_DIR.exists():

        print()
        print(
            "ERROR: Test directory not found."
        )

        print(
            TEST_DIR
        )

        sys.exit(1)

    print()
    print(
        "Original dataset structure verified."
    )


# ============================================================
# 10. FIND ALL IMAGE FILES
# ============================================================

def find_all_images():

    print()
    print(
        "Scanning original dataset..."
    )

    image_files = []

    for split_directory in (
        TRAIN_DIR,
        TEST_DIR
    ):

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
        key=lambda path: str(path).lower()
    )

    print()
    print(
        f"Images found: "
        f"{len(image_files):,}"
    )

    return image_files


# ============================================================
# 11. BUILD HASH DATABASE
# ============================================================

def build_hash_database(
    image_files
):

    print()
    print(
        "Calculating image hashes..."
    )

    print(
        "This may take some time."
    )

    print()

    hash_database = defaultdict(list)

    invalid_images = []

    total = len(
        image_files
    )

    for index, file_path in enumerate(
        image_files,
        start=1
    ):

        # ----------------------------------------------------
        # Validate image
        # ----------------------------------------------------

        if not is_valid_image(
            file_path
        ):

            invalid_images.append(
                {
                    "file": str(
                        file_path
                    ),
                    "class": get_class_name(
                        file_path
                    ),
                    "split": get_split_name(
                        file_path
                    ),
                }
            )

            continue

        # ----------------------------------------------------
        # Calculate hash
        # ----------------------------------------------------

        image_hash = calculate_md5(
            file_path
        )

        if image_hash is None:

            invalid_images.append(
                {
                    "file": str(
                        file_path
                    ),
                    "class": get_class_name(
                        file_path
                    ),
                    "split": get_split_name(
                        file_path
                    ),
                }
            )

            continue

        # ----------------------------------------------------
        # Store image information
        # ----------------------------------------------------

        hash_database[
            image_hash
        ].append(
            {
                "file": file_path,
                "class": get_class_name(
                    file_path
                ),
                "split": get_split_name(
                    file_path
                ),
            }
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            index % 500 == 0
            or index == total
        ):

            print(
                f"Processed "
                f"{index:,}/"
                f"{total:,}"
            )

    print()
    print(
        f"Unique image hashes: "
        f"{len(hash_database):,}"
    )

    print(
        f"Invalid images: "
        f"{len(invalid_images):,}"
    )

    return (
        hash_database,
        invalid_images
    )


# ============================================================
# 12. CLASSIFY HASH GROUPS
# ============================================================

def classify_hash_groups(
    hash_database
):
    """
    Separate hashes into:

    unique_images
    same_label_duplicates
    cross_label_conflicts
    """

    unique_images = []

    same_label_duplicates = []

    cross_label_conflicts = []

    for image_hash, files in (
        hash_database.items()
    ):

        # ====================================================
        # CASE 1: UNIQUE IMAGE
        # ====================================================

        if len(files) == 1:

            unique_images.append(
                {
                    "md5": image_hash,
                    **files[0]
                }
            )

            continue

        # ====================================================
        # GET ALL LABELS
        # ====================================================

        labels = {
            item["class"]
            for item in files
        }

        # ====================================================
        # CASE 2: SAME LABEL DUPLICATE
        # ====================================================

        if len(labels) == 1:

            # Sort deterministically.
            # This ensures reproducibility.

            sorted_files = sorted(
                files,
                key=lambda item:
                str(
                    item["file"]
                ).lower()
            )

            kept_file = (
                sorted_files[0]
            )

            removed_files = (
                sorted_files[1:]
            )

            same_label_duplicates.append(
                {
                    "md5": image_hash,
                    "class": kept_file["class"],
                    "kept": str(
                        kept_file["file"]
                    ),
                    "removed": [
                        str(
                            item["file"]
                        )
                        for item in removed_files
                    ],
                }
            )

            # Keep exactly ONE copy

            unique_images.append(
                {
                    "md5": image_hash,
                    **kept_file
                }
            )

        # ====================================================
        # CASE 3: CROSS-LABEL CONFLICT
        # ====================================================

        else:

            # IMPORTANT:
            #
            # We DO NOT add these images to
            # unique_images.
            #
            # Every copy of this hash will be excluded.
            #
            # This prevents incorrect label assignment.

            cross_label_conflicts.append(
                {
                    "md5": image_hash,
                    "labels": sorted(
                        labels
                    ),
                    "files": files
                }
            )

    return (
        unique_images,
        same_label_duplicates,
        cross_label_conflicts
    )


# ============================================================
# 13. COUNT CONFLICT FILES
# ============================================================

def count_conflict_files(
    cross_label_conflicts
):

    return sum(
        len(
            item["files"]
        )
        for item in cross_label_conflicts
    )


# ============================================================
# 14. PRINT CLEANING SUMMARY
# ============================================================

def print_cleaning_summary(
    original_count,
    unique_images,
    same_label_duplicates,
    cross_label_conflicts,
    invalid_images
):

    same_label_removed = sum(
        len(
            item["removed"]
        )
        for item in same_label_duplicates
    )

    conflict_files = (
        count_conflict_files(
            cross_label_conflicts
        )
    )

    final_count = len(
        unique_images
    ) - conflict_files

    print_header(
        "CLEANING ANALYSIS"
    )

    print(
        f"Original image files       : "
        f"{original_count:,}"
    )

    print(
        f"Unique image hashes        : "
        f"{len(unique_images):,}"
    )

    print(
        f"Same-label duplicate groups: "
        f"{len(same_label_duplicates):,}"
    )

    print(
        f"Cross-label conflicts     : "
        f"{len(cross_label_conflicts):,}"
    )

    print(
        f"Invalid images             : "
        f"{len(invalid_images):,}"
    )

    print()

    print(
        f"Same-label files removed  : "
        f"{same_label_removed:,}"
    )

    print(
        f"Conflict files removed    : "
        f"{conflict_files:,}"
    )

    print(
        f"Final clean image count   : "
        f"{final_count:,}"
    )

    print_line()


# ============================================================
# 15. PREPARE CLEAN DATASET
# ============================================================

def prepare_clean_dataset():

    if CLEANED_DATASET.exists():

        print()
        print(
            "A previous cleaned dataset exists:"
        )

        print(
            CLEANED_DATASET
        )

        print()

        response = input(
            "Delete it and recreate? (yes/no): "
        ).strip().lower()

        if response != "yes":

            print()
            print(
                "Operation cancelled."
            )

            sys.exit(0)

        print()
        print(
            "Removing previous cleaned dataset..."
        )

        shutil.rmtree(
            CLEANED_DATASET
        )

    CLEANED_DATASET.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 16. COPY CLEAN IMAGES
# ============================================================

def copy_clean_images(
    clean_images
):
    """
    Copy only images that survived cleaning.

    Cross-label conflicts are already excluded
    from clean_images.
    """

    copied = 0

    class_counts = Counter()

    total = len(
        clean_images
    )

    print()
    print(
        "Copying clean images..."
    )

    print()

    for item in clean_images:

        source_file = Path(
            item["file"]
        )

        class_name = item["class"]

        image_hash = item["md5"]

        # ----------------------------------------------------
        # Create class directory
        # ----------------------------------------------------

        destination_directory = (
            CLEANED_DATASET
            / class_name
        )

        destination_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Use hash-based filename
        # ----------------------------------------------------

        extension = (
            source_file.suffix.lower()
        )

        destination_file = (
            destination_directory
            / f"{image_hash}{extension}"
        )

        # ----------------------------------------------------
        # Copy
        # ----------------------------------------------------

        shutil.copy2(
            source_file,
            destination_file
        )

        copied += 1

        class_counts[
            class_name
        ] += 1

        if (
            copied % 500 == 0
            or copied == total
        ):

            print(
                f"Copied "
                f"{copied:,}/"
                f"{total:,}"
            )

    print()

    print(
        f"Images copied: "
        f"{copied:,}"
    )

    return class_counts


# ============================================================
# 17. SAVE CLEANING REPORT
# ============================================================

def save_cleaning_report(
    same_label_duplicates,
    cross_label_conflicts,
    invalid_images
):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # SAME-LABEL DUPLICATES
    # ========================================================

    duplicate_records = []

    for item in (
        same_label_duplicates
    ):

        for removed_file in (
            item["removed"]
        ):

            duplicate_records.append(
                {
                    "type":
                        "same_label_duplicate",

                    "md5":
                        item["md5"],

                    "class":
                        item["class"],

                    "kept_file":
                        item["kept"],

                    "removed_file":
                        removed_file,
                }
            )

    duplicate_dataframe = pd.DataFrame(
        duplicate_records,
        columns=[
            "type",
            "md5",
            "class",
            "kept_file",
            "removed_file",
        ]
    )

    duplicate_dataframe.to_csv(
        CLEANING_REPORT,
        index=False,
        encoding="utf-8"
    )

    # ========================================================
    # CROSS-LABEL CONFLICTS
    # ========================================================

    conflict_records = []

    for item in (
        cross_label_conflicts
    ):

        label_text = ", ".join(
            item["labels"]
        )

        for file_info in (
            item["files"]
        ):

            conflict_records.append(
                {
                    "md5":
                        item["md5"],

                    "labels":
                        label_text,

                    "split":
                        file_info["split"],

                    "class":
                        file_info["class"],

                    "file":
                        str(
                            file_info["file"]
                        ),
                }
            )

    conflict_dataframe = pd.DataFrame(
        conflict_records,
        columns=[
            "md5",
            "labels",
            "split",
            "class",
            "file",
        ]
    )

    conflict_dataframe.to_csv(
        CONFLICT_REPORT,
        index=False,
        encoding="utf-8"
    )

    # ========================================================
    # INVALID IMAGES
    # ========================================================

    invalid_dataframe = pd.DataFrame(
        invalid_images,
        columns=[
            "file",
            "class",
            "split",
        ]
    )

    invalid_dataframe.to_csv(
        INVALID_REPORT,
        index=False,
        encoding="utf-8"
    )

    print()
    print(
        "Cleaning reports saved:"
    )

    print(
        CLEANING_REPORT
    )

    print(
        CONFLICT_REPORT
    )

    print(
        INVALID_REPORT
    )


# ============================================================
# 18. SAVE FINAL CLASS DISTRIBUTION
# ============================================================

def save_class_distribution(
    class_counts
):

    records = []

    for class_name in sorted(
        class_counts,
        key=lambda name:
        name.lower()
    ):

        records.append(
            {
                "class": class_name,
                "images": class_counts[
                    class_name
                ],
            }
        )

    dataframe = pd.DataFrame(
        records
    )

    dataframe.to_csv(
        FINAL_CLASS_REPORT,
        index=False,
        encoding="utf-8"
    )

    print()
    print(
        "Clean class distribution saved:"
    )

    print(
        FINAL_CLASS_REPORT
    )


# ============================================================
# 19. VERIFY CLEAN DATASET
# ============================================================

def verify_clean_dataset():

    print_header(
        "VERIFYING CLEAN DATASET"
    )

    if not CLEANED_DATASET.exists():

        print(
            "ERROR: Cleaned dataset not found."
        )

        return False

    cleaned_files = []

    for file_path in (
        CLEANED_DATASET.rglob("*")
    ):

        if not file_path.is_file():
            continue

        if (
            file_path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        ):

            cleaned_files.append(
                file_path
            )

    print(
        f"Cleaned images: "
        f"{len(cleaned_files):,}"
    )

    # ========================================================
    # HASH VERIFICATION
    # ========================================================

    hashes = defaultdict(list)

    for file_path in cleaned_files:

        image_hash = calculate_md5(
            file_path
        )

        if image_hash:

            hashes[
                image_hash
            ].append(
                file_path
            )

    duplicate_groups = {
        image_hash: files
        for image_hash, files
        in hashes.items()
        if len(files) > 1
    }

    print(
        f"Duplicate groups remaining: "
        f"{len(duplicate_groups):,}"
    )

    if duplicate_groups:

        print()
        print(
            "ERROR: Duplicate images remain."
        )

        return False

    # ========================================================
    # VERIFY FILE READABILITY
    # ========================================================

    corrupted = []

    for file_path in cleaned_files:

        if not is_valid_image(
            file_path
        ):

            corrupted.append(
                file_path
            )

    print(
        f"Corrupted images remaining: "
        f"{len(corrupted):,}"
    )

    if corrupted:

        print()
        print(
            "ERROR: Corrupted images remain."
        )

        return False

    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    class_counts = Counter()

    for file_path in cleaned_files:

        class_name = (
            file_path.parent.name
        )

        class_counts[
            class_name
        ] += 1

    print()
    print(
        "Clean dataset class distribution:"
    )

    print_line("-")

    print(
        f"{'Class':<30}"
        f"{'Images':>10}"
    )

    print_line("-")

    for class_name in sorted(
        class_counts,
        key=lambda name:
        name.lower()
    ):

        print(
            f"{class_name:<30}"
            f"{class_counts[class_name]:>10,}"
        )

    print_line("-")

    print(
        f"{'TOTAL':<30}"
        f"{sum(class_counts.values()):>10,}"
    )

    # ========================================================
    # CROSS-LABEL CHECK
    # ========================================================

    print()
    print(
        "Checking for cross-label hash conflicts..."
    )

    clean_hash_labels = defaultdict(
        set
    )

    for file_path in cleaned_files:

        image_hash = calculate_md5(
            file_path
        )

        if image_hash:

            clean_hash_labels[
                image_hash
            ].add(
                file_path.parent.name
            )

    remaining_conflicts = {
        image_hash: labels
        for image_hash, labels
        in clean_hash_labels.items()
        if len(labels) > 1
    }

    print(
        f"Cross-label conflicts remaining: "
        f"{len(remaining_conflicts):,}"
    )

    if remaining_conflicts:

        print()
        print(
            "ERROR: Cross-label conflicts remain."
        )

        return False

    # ========================================================
    # SUCCESS
    # ========================================================

    print()
    print_line("=")

    print(
        "CLEAN DATASET VERIFICATION PASSED"
    )

    print_line("=")

    print(
        "✓ No duplicate images"
    )

    print(
        "✓ No corrupted images"
    )

    print(
        "✓ No cross-label conflicts"
    )

    print_line("=")

    return True


# ============================================================
# 20. MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Step 1
    # --------------------------------------------------------

    check_dataset()

    # --------------------------------------------------------
    # Step 2
    # --------------------------------------------------------

    image_files = find_all_images()

    if not image_files:

        print()
        print(
            "ERROR: No images found."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Step 3
    # --------------------------------------------------------

    (
        hash_database,
        invalid_images
    ) = build_hash_database(
        image_files
    )

    # --------------------------------------------------------
    # Step 4
    # --------------------------------------------------------

    (
        clean_images,
        same_label_duplicates,
        cross_label_conflicts
    ) = classify_hash_groups(
        hash_database
    )

    # --------------------------------------------------------
    # Step 5
    # --------------------------------------------------------

    print_cleaning_summary(
        original_count=len(
            image_files
        ),
        unique_images=clean_images,
        same_label_duplicates=
            same_label_duplicates,
        cross_label_conflicts=
            cross_label_conflicts,
        invalid_images=
            invalid_images
    )

    # --------------------------------------------------------
    # Step 6
    # --------------------------------------------------------

    print()
    print_line("=")

    print(
        "CLEANING POLICY"
    )

    print_line("=")

    print(
        "1. Unique image"
        "              -> KEEP"
    )

    print(
        "2. Same-label duplicate"
        "  -> KEEP ONE"
    )

    print(
        "3. Cross-label conflict"
        " -> REMOVE ALL"
    )

    print(
        "4. Invalid image"
        "          -> REMOVE"
    )

    print(
        "5. Original dataset"
        "      -> NEVER MODIFY"
    )

    print_line("=")

    # --------------------------------------------------------
    # Step 7
    # --------------------------------------------------------

    prepare_clean_dataset()

    # --------------------------------------------------------
    # Step 8
    # --------------------------------------------------------

    class_counts = copy_clean_images(
        clean_images
    )

    # --------------------------------------------------------
    # Step 9
    # --------------------------------------------------------

    save_cleaning_report(
        same_label_duplicates,
        cross_label_conflicts,
        invalid_images
    )

    # --------------------------------------------------------
    # Step 10
    # --------------------------------------------------------

    save_class_distribution(
        class_counts
    )

    # --------------------------------------------------------
    # Step 11
    # --------------------------------------------------------

    success = verify_clean_dataset()

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()

    print_line("=")

    if success:

        print(
            "CLEAN DATASET CREATED SUCCESSFULLY"
        )

        print_line("=")

        print()
        print(
            "Clean dataset:"
        )

        print(
            CLEANED_DATASET
        )

        print()
        print(
            "Original Kaggle dataset was NOT modified."
        )

        print()
        print(
            "NEXT STEP:"
        )

        print(
            "Create a new stratified train/validation/test split."
        )

    else:

        print(
            "CLEAN DATASET VERIFICATION FAILED."
        )

        print(
            "Do NOT train the model yet."
        )

    print()


# ============================================================
# 21. PROGRAM ENTRY POINT
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

        print(
            "Please check the error above."
        )

        sys.exit(1)