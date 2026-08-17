"""
============================================================
SKIN DISEASE CNN PROJECT
IMAGE PREPROCESSING
============================================================

INPUT:
    dataset/processed/
        ├── train/
        ├── val/
        └── test/

OUTPUT:
    dataset/preprocessed/
        ├── train/
        ├── val/
        └── test/

PREPROCESSING:
    1. Read image
    2. Verify image
    3. Convert to RGB
    4. Resize to 224 x 224
    5. Save as JPEG
    6. Preserve class structure
    7. Verify output

IMPORTANT:
    - Original dataset is not modified.
    - Processed split is not modified.
    - No augmentation is performed here.
    - Augmentation will be applied ONLY during
      training.

============================================================
"""

import sys
import shutil

from pathlib import Path

from PIL import Image, ImageOps


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------
# INPUT
# ------------------------------------------------------------

INPUT_DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
)

INPUT_TRAIN = INPUT_DATASET / "train"
INPUT_VAL = INPUT_DATASET / "val"
INPUT_TEST = INPUT_DATASET / "test"


# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

OUTPUT_DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "preprocessed"
)

OUTPUT_TRAIN = OUTPUT_DATASET / "train"
OUTPUT_VAL = OUTPUT_DATASET / "val"
OUTPUT_TEST = OUTPUT_DATASET / "test"


# ============================================================
# 2. CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)

JPEG_QUALITY = 95

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
# 3. PRINT FUNCTIONS
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
# 4. CHECK INPUT DATASET
# ============================================================

def check_input_dataset():

    print_header(
        "CHECKING INPUT DATASET"
    )

    print(
        "Input:"
    )

    print(
        INPUT_DATASET
    )

    print()

    # --------------------------------------------------------
    # Check main directory
    # --------------------------------------------------------

    if not INPUT_DATASET.exists():

        print(
            "ERROR: Processed dataset does not exist."
        )

        print()
        print(
            "Expected:"
        )

        print(
            INPUT_DATASET
        )

        print()
        print(
            "Run split_dataset.py first."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Check train
    # --------------------------------------------------------

    if not INPUT_TRAIN.exists():

        print(
            "ERROR: Train directory not found."
        )

        print(
            INPUT_TRAIN
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Check validation
    # --------------------------------------------------------

    if not INPUT_VAL.exists():

        print(
            "ERROR: Validation directory not found."
        )

        print(
            INPUT_VAL
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Check test
    # --------------------------------------------------------

    if not INPUT_TEST.exists():

        print(
            "ERROR: Test directory not found."
        )

        print(
            INPUT_TEST
        )

        sys.exit(1)

    print(
        "✓ Train directory found"
    )

    print(
        "✓ Validation directory found"
    )

    print(
        "✓ Test directory found"
    )


# ============================================================
# 5. GET CLASS NAMES
# ============================================================

def get_class_names():

    class_names = set()

    for split_directory in [
        INPUT_TRAIN,
        INPUT_VAL,
        INPUT_TEST
    ]:

        for item in split_directory.iterdir():

            if item.is_dir():

                class_names.add(
                    item.name
                )

    class_names = sorted(
        class_names,
        key=lambda name:
        name.lower()
    )

    return class_names


# ============================================================
# 6. FIND IMAGE FILES
# ============================================================

def find_images(
    directory
):

    image_files = []

    for file_path in directory.rglob("*"):

        if not file_path.is_file():
            continue

        if (
            file_path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        ):

            image_files.append(
                file_path
            )

    return sorted(
        image_files,
        key=lambda path:
        str(path).lower()
    )


# ============================================================
# 7. RESIZE IMAGE
# ============================================================

def resize_image(
    image
):
    """
    Resize image to 224 x 224.

    ImageOps.fit:
        - preserves aspect ratio
        - performs center crop when required
        - avoids image distortion
    """

    image = ImageOps.fit(
        image,
        IMAGE_SIZE,
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5)
    )

    return image


# ============================================================
# 8. PROCESS ONE IMAGE
# ============================================================

def process_image(
    source_file,
    destination_file
):

    try:

        # ----------------------------------------------------
        # Open image
        # ----------------------------------------------------

        with Image.open(
            source_file
        ) as image:

            # Force loading
            image.load()

            # ------------------------------------------------
            # Convert to RGB
            # ------------------------------------------------

            image = image.convert(
                "RGB"
            )

            # ------------------------------------------------
            # Resize
            # ------------------------------------------------

            image = resize_image(
                image
            )

            # ------------------------------------------------
            # Save JPEG
            # ------------------------------------------------

            image.save(
                destination_file,
                format="JPEG",
                quality=JPEG_QUALITY,
                optimize=True
            )

        return True, None

    except Exception as error:

        return False, str(error)


# ============================================================
# 9. CREATE OUTPUT DIRECTORIES
# ============================================================

def prepare_output_dataset():

    print_header(
        "PREPARING PREPROCESSED DATASET"
    )

    print(
        "Output:"
    )

    print(
        OUTPUT_DATASET
    )

    if OUTPUT_DATASET.exists():

        print()
        print(
            "A previous preprocessed dataset exists:"
        )

        print(
            OUTPUT_DATASET
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
            "Removing previous preprocessed dataset..."
        )

        shutil.rmtree(
            OUTPUT_DATASET
        )

    OUTPUT_TRAIN.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_VAL.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_TEST.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print(
        "✓ Output directories created"
    )


# ============================================================
# 10. PROCESS ONE SPLIT
# ============================================================

def process_split(
    split_name,
    input_directory,
    output_directory
):

    print_header(
        f"PROCESSING {split_name.upper()} DATASET"
    )

    image_files = find_images(
        input_directory
    )

    total = len(
        image_files
    )

    print(
        f"Images found: {total:,}"
    )

    if total == 0:

        print(
            "WARNING: No images found."
        )

        return {
            "total": 0,
            "success": 0,
            "failed": 0,
            "failed_files": []
        }

    success_count = 0

    failed_count = 0

    failed_files = []

    for index, source_file in enumerate(
        image_files,
        start=1
    ):

        # ----------------------------------------------------
        # Determine class
        # ----------------------------------------------------

        relative_path = (
            source_file.relative_to(
                input_directory
            )
        )

        class_name = (
            relative_path.parts[0]
        )

        # ----------------------------------------------------
        # Create class directory
        # ----------------------------------------------------

        destination_class_dir = (
            output_directory
            / class_name
        )

        destination_class_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Destination filename
        #
        # We use source stem + .jpg.
        # Since the split was already cleaned,
        # filenames are hash-based.
        # ----------------------------------------------------

        destination_file = (
            destination_class_dir
            / f"{source_file.stem}.jpg"
        )

        # ----------------------------------------------------
        # Process
        # ----------------------------------------------------

        success, error = process_image(
            source_file,
            destination_file
        )

        if success:

            success_count += 1

        else:

            failed_count += 1

            failed_files.append(
                {
                    "source": str(
                        source_file
                    ),
                    "error": error
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
                f"{index:,}/{total:,} "
                f"| Successful: "
                f"{success_count:,} "
                f"| Failed: "
                f"{failed_count:,}"
            )

    print()

    print(
        f"{split_name} processing completed."
    )

    print(
        f"Total      : {total:,}"
    )

    print(
        f"Successful : {success_count:,}"
    )

    print(
        f"Failed     : {failed_count:,}"
    )

    return {
        "total": total,
        "success": success_count,
        "failed": failed_count,
        "failed_files": failed_files
    }


# ============================================================
# 11. SAVE FAILED IMAGE REPORT
# ============================================================

def save_failed_report(
    all_failures
):

    report_directory = (
        PROJECT_ROOT
        / "results"
        / "reports"
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    report_file = (
        report_directory
        / "preprocessing_errors.csv"
    )

    import pandas as pd

    dataframe = pd.DataFrame(
        all_failures,
        columns=[
            "split",
            "source",
            "error"
        ]
    )

    dataframe.to_csv(
        report_file,
        index=False,
        encoding="utf-8"
    )

    print()
    print(
        "Preprocessing error report:"
    )

    print(
        report_file
    )


# ============================================================
# 12. VERIFY IMAGE SIZE
# ============================================================

def verify_processed_images(
    split_name,
    directory
):

    print()
    print(
        f"Verifying {split_name} images..."
    )

    image_files = find_images(
        directory
    )

    wrong_size = []

    wrong_mode = []

    unreadable = []

    for file_path in image_files:

        try:

            with Image.open(
                file_path
            ) as image:

                # --------------------------------------------
                # Verify size
                # --------------------------------------------

                if image.size != IMAGE_SIZE:

                    wrong_size.append(
                        {
                            "file": str(
                                file_path
                            ),
                            "size": str(
                                image.size
                            )
                        }
                    )

                # --------------------------------------------
                # Verify color mode
                # --------------------------------------------

                if image.mode != "RGB":

                    wrong_mode.append(
                        {
                            "file": str(
                                file_path
                            ),
                            "mode": image.mode
                        }
                    )

                # --------------------------------------------
                # Force read
                # --------------------------------------------

                image.load()

        except Exception as error:

            unreadable.append(
                {
                    "file": str(
                        file_path
                    ),
                    "error": str(
                        error
                    )
                }
            )

    print(
        f"Total images : {len(image_files):,}"
    )

    print(
        f"Wrong size   : {len(wrong_size):,}"
    )

    print(
        f"Wrong mode   : {len(wrong_mode):,}"
    )

    print(
        f"Unreadable   : {len(unreadable):,}"
    )

    return {
        "total": len(image_files),
        "wrong_size": wrong_size,
        "wrong_mode": wrong_mode,
        "unreadable": unreadable
    }


# ============================================================
# 13. VERIFY CLASS STRUCTURE
# ============================================================

def verify_class_structure(
    expected_classes
):

    print_header(
        "VERIFYING CLASS STRUCTURE"
    )

    success = True

    expected = set(
        expected_classes
    )

    for split_name, split_directory in [
        ("train", OUTPUT_TRAIN),
        ("validation", OUTPUT_VAL),
        ("test", OUTPUT_TEST)
    ]:

        actual = {
            item.name
            for item in split_directory.iterdir()
            if item.is_dir()
        }

        missing = (
            expected
            - actual
        )

        extra = (
            actual
            - expected
        )

        print()
        print(
            f"{split_name.upper()}:"
        )

        print(
            f"  Classes found: "
            f"{len(actual)}"
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
# 14. COMPARE IMAGE COUNTS
# ============================================================

def count_images(
    directory
):

    return len(
        find_images(
            directory
        )
    )


def compare_counts(
    original_results,
    processed_results
):

    print_header(
        "COMPARING IMAGE COUNTS"
    )

    success = True

    for split_name in [
        "train",
        "validation",
        "test"
    ]:

        original_count = (
            original_results[
                split_name
            ]["success"]
        )

        processed_count = (
            processed_results[
                split_name
            ]
        )

        print(
            f"{split_name.capitalize():<12}"
            f"Original: "
            f"{original_count:>6,}    "
            f"Preprocessed: "
            f"{processed_count:>6,}"
        )

        if (
            original_count
            != processed_count
        ):

            success = False

            print(
                "  ✗ Count mismatch"
            )

        else:

            print(
                "  ✓ Count matches"
            )

    return success


# ============================================================
# 15. SAVE FINAL REPORT
# ============================================================

def save_final_report(
    original_results,
    processed_counts,
    verification_results
):

    import pandas as pd

    report_directory = (
        PROJECT_ROOT
        / "results"
        / "reports"
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    report_file = (
        report_directory
        / "preprocessing_summary.csv"
    )

    records = []

    for split_name in [
        "train",
        "validation",
        "test"
    ]:

        key = split_name

        records.append(
            {
                "split": split_name,
                "original_images":
                    original_results[
                        key
                    ]["success"],
                "preprocessed_images":
                    processed_counts[
                        key
                    ],
                "wrong_size":
                    len(
                        verification_results[
                            key
                        ]["wrong_size"]
                    ),
                "wrong_mode":
                    len(
                        verification_results[
                            key
                        ]["wrong_mode"]
                    ),
                "unreadable":
                    len(
                        verification_results[
                            key
                        ]["unreadable"]
                    )
            }
        )

    dataframe = pd.DataFrame(
        records
    )

    dataframe.to_csv(
        report_file,
        index=False,
        encoding="utf-8"
    )

    print()
    print(
        "Preprocessing summary saved:"
    )

    print(
        report_file
    )


# ============================================================
# 16. MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Step 1
    # --------------------------------------------------------

    check_input_dataset()

    # --------------------------------------------------------
    # Step 2
    # --------------------------------------------------------

    class_names = get_class_names()

    print()
    print(
        f"Classes detected: "
        f"{len(class_names)}"
    )

    # --------------------------------------------------------
    # Step 3
    # --------------------------------------------------------

    prepare_output_dataset()

    # --------------------------------------------------------
    # Step 4
    # --------------------------------------------------------

    train_result = process_split(
        "train",
        INPUT_TRAIN,
        OUTPUT_TRAIN
    )

    val_result = process_split(
        "validation",
        INPUT_VAL,
        OUTPUT_VAL
    )

    test_result = process_split(
        "test",
        INPUT_TEST,
        OUTPUT_TEST
    )

    original_results = {
        "train": train_result,
        "validation": val_result,
        "test": test_result
    }

    # --------------------------------------------------------
    # Step 5
    # --------------------------------------------------------

    all_failures = []

    for split_name, result in [
        ("train", train_result),
        ("validation", val_result),
        ("test", test_result)
    ]:

        for failure in (
            result["failed_files"]
        ):

            all_failures.append(
                {
                    "split": split_name,
                    "source": failure["source"],
                    "error": failure["error"]
                }
            )

    if all_failures:

        save_failed_report(
            all_failures
        )

    # --------------------------------------------------------
    # Step 6
    # --------------------------------------------------------

    verification_results = {}

    verification_results[
        "train"
    ] = verify_processed_images(
        "train",
        OUTPUT_TRAIN
    )

    verification_results[
        "validation"
    ] = verify_processed_images(
        "validation",
        OUTPUT_VAL
    )

    verification_results[
        "test"
    ] = verify_processed_images(
        "test",
        OUTPUT_TEST
    )

    # --------------------------------------------------------
    # Step 7
    # --------------------------------------------------------

    class_check = verify_class_structure(
        class_names
    )

    # --------------------------------------------------------
    # Step 8
    # --------------------------------------------------------

    processed_counts = {
        "train":
            count_images(
                OUTPUT_TRAIN
            ),

        "validation":
            count_images(
                OUTPUT_VAL
            ),

        "test":
            count_images(
                OUTPUT_TEST
            )
    }

    # --------------------------------------------------------
    # Step 9
    # --------------------------------------------------------

    count_check = compare_counts(
        original_results,
        processed_counts
    )

    # --------------------------------------------------------
    # Step 10
    # --------------------------------------------------------

    save_final_report(
        original_results,
        processed_counts,
        verification_results
    )

    # --------------------------------------------------------
    # Step 11
    # Final verification
    # --------------------------------------------------------

    all_images_valid = True

    for split_name in [
        "train",
        "validation",
        "test"
    ]:

        result = (
            verification_results[
                split_name
            ]
        )

        if (
            result["wrong_size"]
            or result["wrong_mode"]
            or result["unreadable"]
        ):

            all_images_valid = False

    no_failures = (
        len(all_failures) == 0
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print_header(
        "FINAL PREPROCESSING SUMMARY"
    )

    print(
        f"Train images      : "
        f"{processed_counts['train']:,}"
    )

    print(
        f"Validation images : "
        f"{processed_counts['validation']:,}"
    )

    print(
        f"Test images       : "
        f"{processed_counts['test']:,}"
    )

    total_images = (
        processed_counts["train"]
        + processed_counts["validation"]
        + processed_counts["test"]
    )

    print(
        f"Total images      : "
        f"{total_images:,}"
    )

    print()

    print(
        f"Target image size : "
        f"{IMAGE_SIZE[0]} x "
        f"{IMAGE_SIZE[1]}"
    )

    print(
        "Color mode        : RGB"
    )

    print(
        "Augmentation      : NOT APPLIED"
    )

    print()

    print_line("=")

    if (
        class_check
        and count_check
        and all_images_valid
        and no_failures
    ):

        print(
            "IMAGE PREPROCESSING COMPLETED SUCCESSFULLY"
        )

        print_line("=")

        print()
        print(
            "✓ All images converted to RGB"
        )

        print(
            "✓ All images resized to 224 x 224"
        )

        print(
            "✓ All images readable"
        )

        print(
            "✓ All classes preserved"
        )

        print(
            "✓ Image counts preserved"
        )

        print(
            "✓ No augmentation applied"
        )

        print()
        print(
            "Preprocessed dataset:"
        )

        print(
            OUTPUT_DATASET
        )

        print()
        print(
            "The dataset is ready for CNN training."
        )

    else:

        print(
            "IMAGE PREPROCESSING COMPLETED "
            "WITH WARNINGS"
        )

        print_line("=")

        if not class_check:

            print(
                "✗ Class structure problem"
            )

        if not count_check:

            print(
                "✗ Image count mismatch"
            )

        if not all_images_valid:

            print(
                "✗ Image validation problem"
            )

        if not no_failures:

            print(
                "✗ Some images failed preprocessing"
            )

        print()
        print(
            "DO NOT START MODEL TRAINING YET."
        )

    print()


# ============================================================
# 17. ENTRY POINT
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