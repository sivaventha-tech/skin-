"""
============================================================
SKIN DISEASE CLASSIFICATION SYSTEM
IEEE PUBLICATION MASTER RUNNER & CONTROL SCRIPT
============================================================
"""

import sys
import os
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Skin Disease Classification IEEE System Runner")
    parser.add_argument("--train", action="store_true", help="Train proposed MSTA-Net model on GPU")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model against base paper (94.57%)")
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM++ interpretability heatmaps")
    parser.add_argument("--all", action="store_true", help="Run full pipeline: Train -> Evaluate -> GradCAM++")

    args = parser.parse_args()

    if not any([args.train, args.evaluate, args.gradcam, args.all]):
        args.all = True

    print("=" * 80)
    print("SKIN DISEASE CLASSIFICATION SYSTEM - IEEE PIPELINE RUNNER")
    print("=" * 80)

    if args.train or args.all:
        print("\n[STEP 1] Executing GPU Training for Proposed MSTA-Net Architecture...")
        from training.train_msta_net import main as train_main
        train_main()

    if args.evaluate or args.all:
        print("\n[STEP 2] Evaluating Performance Metrics against Base Paper Benchmark...")
        from evaluation.evaluate_msta import evaluate_model
        evaluate_model()

    if args.gradcam or args.all:
        print("\n[STEP 3] Generating Grad-CAM++ Diagnostic Visualizations...")
        from explainability.gradcam import generate_and_save_gradcam
        test_dir = PROJECT_ROOT / "dataset" / "preprocessed" / "test"
        sample_imgs = list(test_dir.glob("*/*.jpeg")) + list(test_dir.glob("*/*.jpg")) + list(test_dir.glob("*/*.png"))
        if sample_imgs:
            generate_and_save_gradcam(sample_imgs[0])

    print("\n" + "=" * 80)
    print("ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
