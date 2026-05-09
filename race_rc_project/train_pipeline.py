"""
Single-entry-point training script for the RACE RC project.

This mirrors the reference project style while reusing the existing
Model A and Model B training functions in src/.

Usage:
  python train_pipeline.py
  python train_pipeline.py --skip-modela
  python train_pipeline.py --skip-modelb
    Local: python train_pipeline.py --base C:/path/to/race_rc_project
    Kaggle: python train_pipeline.py --base /kaggle/working/race_rc_project
"""

import argparse
import os
import pickle
import sys
import time


parser = argparse.ArgumentParser(description="RACE RC Project training pipeline")
parser.add_argument(
    "--base",
    type=str,
    default=None,
    help="Project base directory. Defaults to the folder containing this script.",
)
parser.add_argument(
    "--skip-modela",
    action="store_true",
    help="Skip Model A training.",
)
parser.add_argument(
    "--skip-modelb",
    action="store_true",
    help="Skip Model B training.",
)
parser.add_argument(
    "--preprocess",
    action="store_true",
    help="Run preprocessing before training.",
)
parser.add_argument(
    "--max-train-rows",
    type=int,
    default=None,
    help="Limit training rows loaded by the trainers.",
)
parser.add_argument(
    "--max-val-rows",
    type=int,
    default=None,
    help="Limit validation rows loaded by the trainers.",
)
parser.add_argument(
    "--max-test-rows",
    type=int,
    default=None,
    help="Limit test rows loaded by Model A evaluation.",
)
args = parser.parse_args()


def banner(title):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


# Local: use the repo root. Kaggle: use /kaggle/working/<project_folder> or pass --base.
BASE_DIR = args.base or os.getcwd()
SRC_DIR = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def read_metrics(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle)


def summarize_model_a(model_dir):
    metrics = read_metrics(os.path.join(model_dir, "metrics.pkl"))
    if not metrics:
        print("  No Model A metrics found.")
        return

    print("  Model A metrics:")
    for key in ["lr", "svm", "ensemble_val_acc", "ensemble_val_f1"]:
        value = metrics.get(key)
        if isinstance(value, dict):
            print(f"    {key}: {value}")
        elif value is not None:
            print(f"    {key}: {value}")


def summarize_model_b(model_dir):
    metrics = read_metrics(os.path.join(model_dir, "metrics.pkl"))
    if not metrics:
        print("  No Model B metrics found.")
        return

    print("  Model B metrics:")
    for key in ["distractor", "hint", "distractor_generation", "hint_generation", "distractor_topk", "hint_tier_hits"]:
        value = metrics.get(key)
        if isinstance(value, dict):
            print(f"    {key}: {value}")
        elif value is not None:
            print(f"    {key}: {value}")


def main():
    if args.max_train_rows is not None:
        os.environ["MAX_TRAIN_ROWS"] = str(args.max_train_rows)
    if args.max_val_rows is not None:
        os.environ["MAX_VAL_ROWS"] = str(args.max_val_rows)
    if args.max_test_rows is not None:
        os.environ["MAX_TEST_ROWS"] = str(args.max_test_rows)

    if args.preprocess:
        from preprocessing import run_preprocessing_pipeline

        import preprocessing as prep

        if args.max_train_rows is not None:
            prep.DATASET_ROW_LIMIT = args.max_train_rows

        banner("STAGE 0 — PREPROCESSING")
        preprocess_summary = run_preprocessing_pipeline(os.path.join(BASE_DIR, "data", "raw", "train.csv"))
        print(f"  Preprocessing summary: {preprocess_summary}")

    banner("STAGE 1 — PRE-TRAIN CHECKS")

    processed_dir = os.path.join(BASE_DIR, "data", "processed")
    required_files = [
        os.path.join(processed_dir, "X_train_ohe.npz"),
        os.path.join(processed_dir, "X_val_ohe.npz"),
        os.path.join(processed_dir, "X_test_ohe.npz"),
        os.path.join(processed_dir, "y_train.npy"),
        os.path.join(processed_dir, "y_val.npy"),
        os.path.join(processed_dir, "y_test.npy"),
        os.path.join(processed_dir, "train_clean.csv"),
        os.path.join(processed_dir, "val_clean.csv"),
        os.path.join(processed_dir, "tfidf_vectorizer.pkl"),
        os.path.join(processed_dir, "feature_engineer.pkl"),
    ]

    missing = [path for path in required_files if not os.path.exists(path)]
    if missing:
        print("  Missing processed artifacts:")
        for path in missing:
            print(f"    - {path}")
        print("  Run preprocessing first, then rerun this pipeline.")
        return

    banner("STAGE 2 — MODEL A")
    if args.skip_modela:
        print("  Skipped (--skip-modela)")
    else:
        start = time.time()
        from model_a_train import train_all as train_model_a

        train_model_a()
        print(f"\n  Model A completed in {time.time() - start:.1f}s")

    banner("STAGE 3 — MODEL B")
    if args.skip_modelb:
        print("  Skipped (--skip-modelb)")
    else:
        start = time.time()
        from model_b_train import train_and_save as train_model_b

        train_model_b()
        print(f"\n  Model B completed in {time.time() - start:.1f}s")

    banner("STAGE 4 — FINAL SUMMARY")
    summarize_model_a(os.path.join(BASE_DIR, "models", "model_a", "traditional"))
    summarize_model_b(os.path.join(BASE_DIR, "models", "model_b", "traditional"))


if __name__ == "__main__":
    main()