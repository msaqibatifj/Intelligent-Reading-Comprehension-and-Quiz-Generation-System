"""
transformer_train.py — Training pipelines for transformer-based MCQ models
Optimized for dual-GPU DDP on Kaggle with maximum throughput.

Three sub-pipelines:
  1. TransformerAnswerVerifier  (BERT)   — score P(correct) for (article, question, option)
  2. QuestionGenerator          (T5)     — generate question from (context, answer)
  3. DistractorGenerator        (BART)   — generate distractors from (question, correct)

Launch with torchrun for multi-GPU::

    torchrun --nproc_per_node=N src/transformer_train.py --csv data/raw/train.csv
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist

from ddp_utils import (
    init_ddp,
    destroy_ddp,
    is_main_process,
    is_ddp,
    get_rank,
    get_world_size,
)

# from src.nn_models import (
#     DEVICE,
#     DistractorGenerator,
#     QuestionGenerator,
#     TransformerAnswerVerifier,
#     train_seq2seq,
#     train_transformer,
# )

try:
    _THIS_DIR = Path(__file__).resolve().parent
    _BASE = _THIS_DIR.parent
except NameError:
    _BASE = Path(os.getcwd())
_DATA_RAW = _BASE / "data" / "raw"
_DATA_PROC = _BASE / "data" / "processed"
_MODEL_A_DIR = _BASE / "models" / "model_a" / "transformer"
_MODEL_B_DIR = _BASE / "models" / "model_b" / "transformer"
os.makedirs(_MODEL_A_DIR, exist_ok=True)
os.makedirs(_MODEL_B_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# GPU / hardware setup  (DDP-aware)
# ---------------------------------------------------------------------------

def _setup_gpus() -> Tuple[torch.device, int]:
    """
    Detect available GPUs, initialise DDP (if launched via torchrun),
    and return (primary_device, gpu_count).

    When torchrun is used, LOCAL_RANK and WORLD_SIZE are picked up by
    ``init_ddp()`` and the device is set to ``cuda:LOCAL_RANK``.
    When running without torchrun, we fall back to a single-GPU setup.
    """
    rank, world_size, device = init_ddp()

    if world_size <= 1:
        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            print("[GPU] No CUDA devices found — running on CPU.")
            return torch.device("cpu"), 0
        print(f"[GPU] {n_gpus} CUDA device(s) detected (single-process).")
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"       GPU {i}: {props.name}  |  {props.total_memory // 1024**3} GB VRAM")
    else:
        n_gpus = world_size
        props = torch.cuda.get_device_properties(rank)
        print(f"[GPU] Rank {rank}/{world_size} — {props.name}  |  {props.total_memory // 1024**3} GB VRAM")

    # cuDNN autotuner — big win for fixed input sizes (BERT, T5, BART encoders).
    torch.backends.cudnn.benchmark = True
    # TF32 on Ampere gives ~3× matmul throughput with negligible accuracy loss.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    return device, n_gpus


DEVICE, N_GPUS = _setup_gpus()


def _wrap_ddp(model: nn.Module) -> nn.Module:
    """
    Wrap model with DistributedDataParallel when running under torchrun.
    Uses ``find_unused_parameters=False`` by default for seq2seq models
    where all parameters receive gradient.
    """
    if is_ddp():
        rank = get_rank()
        print(f"[GPU] Rank {rank}: wrapping model with DDP.")
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=False,
        )
    return model.to(DEVICE)


def _try_compile(model: nn.Module) -> nn.Module:
    """
    Attempt torch.compile (PyTorch ≥ 2.0).  Falls back gracefully on older
    versions or unsupported backends (e.g. Windows / CPU-only builds).

    On T4 (no bfloat16 hardware) ``mode="default"`` triggers repeated inductor
    warnings and first-epoch slowdown, so we use ``mode="reduce-overhead"``,
    which applies a lightweight CUDA graph capture instead.
    """
    if hasattr(torch, "compile") and torch.cuda.is_available():
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[Speed] torch.compile() applied (reduce-overhead mode).")
        except Exception as e:
            print(f"[Speed] torch.compile() skipped: {e}")
    return model


# DataLoader performance flags are set directly in nn_models.py
# (train_transformer and train_seq2seq) to avoid monkey-patch issues.


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def _cleanup_old_checkpoints(checkpoint_dir: Path, keep_last: int = 5):
    """
    Remove old epoch checkpoints, keeping only the ``keep_last`` most recent.
    Preserves ``latest.ckpt``, ``best.ckpt``, and non-epoch files.
    """
    if not checkpoint_dir.exists():
        return
    epoch_ckpts = sorted(
        checkpoint_dir.glob("epoch_*.ckpt"),
        key=lambda p: p.stat().st_mtime,
    )
    n_remove = max(0, len(epoch_ckpts) - keep_last)
    for old in epoch_ckpts[:n_remove]:
        old.unlink(missing_ok=True)
        print(f"  checkpoint cleanup: removed {old.name}")


def _save_best_checkpoint(
    checkpoint_dir: Path,
    model: nn.Module,
    val_loss: float,
    epoch: int,
    extra_meta: Optional[dict] = None,
):
    """Save a best-so-far checkpoint that won't be rotated away."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    from nn_models import _unwrap_model
    payload = {
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state_dict": _unwrap_model(model).state_dict(),
        "meta": extra_meta or {},
    }
    best_path = checkpoint_dir / "best.ckpt"
    torch.save(payload, best_path)
    print(f"  best checkpoint updated → {best_path}  (val_loss={val_loss:.4f})")


def _verify_checkpoint(checkpoint_path: Path) -> bool:
    """
    Quick integrity check: can the file be loaded and does it contain
    expected keys?
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        required = {"epoch", "model_state_dict", "optimizer_state_dict", "best_loss"}
        if required.issubset(ckpt.keys()):
            return True
        print(f"  WARNING: {checkpoint_path} is missing keys: {required - ckpt.keys()}")
        return False
    except Exception as e:
        print(f"  WARNING: {checkpoint_path} appears corrupt: {e}")
        return False


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------

def _load_csv(path) -> pd.DataFrame:
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    required = {"article", "question", "A", "B", "C", "D", "answer"}
    if not required.issubset(df.columns):
        print(f"  WARNING: {path} missing columns. Got {list(df.columns)}")
        return pd.DataFrame()
    return df


def prepare_answer_verifier_data(
    df: pd.DataFrame,
    max_rows: Optional[int] = None,
) -> Tuple[List[str], np.ndarray]:
    """
    Convert a MCQ DataFrame into (texts, labels) for TransformerAnswerVerifier.
    Each row → 4 examples, one per option.  Label = 1 for correct option.
    """
    texts, labels = [], []
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=42)
    for _, row in df.iterrows():
        correct_letter = str(row["answer"]).strip().upper()
        for opt_letter in ["A", "B", "C", "D"]:
            option_text = str(row.get(opt_letter, ""))
            text = (
                f"article: {str(row['article'])}\n"
                f"question: {str(row['question'])}\n"
                f"option: {option_text}"
            )
            texts.append(text)
            labels.append(1 if opt_letter == correct_letter else 0)
    return texts, np.array(labels, dtype=np.int32)


def prepare_question_generation_data(
    df: pd.DataFrame,
    max_rows: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """
    Convert a MCQ DataFrame into (inputs, targets) for QuestionGenerator.
    Input:  "context: {article} answer: {correct_option}"
    Target: the question text
    """
    inputs, targets = [], []
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=42)
    for _, row in df.iterrows():
        correct_letter = str(row["answer"]).strip().upper()
        correct_text = str(row.get(correct_letter, ""))
        article = str(row["article"])
        question = str(row.get("question", ""))
        if not question or len(question) < 5:
            continue
        inputs.append(f"context: {article} answer: {correct_text}")
        targets.append(question)
    return inputs, targets


def prepare_distractor_data(
    df: pd.DataFrame,
    max_rows: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """
    Convert MCQ DataFrame into (inputs, targets) for DistractorGenerator.
    Input:  "question: {question} correct: {correct_option}"
    Target: "distractor1 | distractor2 | distractor3"
    """
    inputs, targets = [], []
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=42)
    for _, row in df.iterrows():
        correct_letter = str(row["answer"]).strip().upper()
        correct_text = str(row.get(correct_letter, ""))
        question = str(row.get("question", ""))
        if not question or len(question) < 5:
            continue
        distractors = [
            str(row.get(opt, ""))
            for opt in ["A", "B", "C", "D"]
            if opt != correct_letter
        ]
        distractors = [d for d in distractors if d and d != correct_text]
        if len(distractors) < 3:
            continue
        inputs.append(f"question: {question} correct: {correct_text}")
        targets.append(" | ".join(distractors[:3]))
    return inputs, targets


# ---------------------------------------------------------------------------
# Training wrappers
# ---------------------------------------------------------------------------

def train_answer_verifier_transformer(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    model_name: str = "bert-base-uncased",
    epochs: int = 3,
    # Default batch size scaled up for dual-GPU (2× 16 GB VRAM on T4).
    # DataParallel splits each batch across both cards, so effective per-GPU
    # batch = batch_size / N_GPUS.
    batch_size: int = 128,
    lr: float = 2e-5,
    use_lora: bool = True,
    lora_r: int = 8,
    max_length: int = 256,
    fp16: bool = True,   # AMP handled internally via GradScaler
    max_rows: Optional[int] = None,
    checkpoint_dir: Optional[Path] = None,
    resume_from_checkpoint: Optional[Path] = None,
    max_checkpoints: int = 5,
):
    """Train a BERT-based answer verifier on MCQ data (dual-GPU optimized)."""
    print("[TransformerAV] preparing data …")
    train_texts, train_labels = prepare_answer_verifier_data(train_df, max_rows)

    val_texts, val_labels = None, None
    if val_df is not None and len(val_df) > 0:
        val_texts, val_labels = prepare_answer_verifier_data(val_df, max_rows)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"[TransformerAV] training {model_name}  |  train={len(train_texts):,}  val={len(val_texts) if val_texts else 0:,}")
    model = TransformerAnswerVerifier(
        model_name=model_name,
        num_labels=2,
        use_lora=use_lora,
        lora_r=lora_r,
    )

    model = _try_compile(model)
    model = _wrap_ddp(model)

    ckpt_dir = checkpoint_dir or (_MODEL_A_DIR / "checkpoints")
    if is_main_process():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"[TransformerAV] checkpoints → {ckpt_dir}  (max per epoch: {max_checkpoints})")

    print(f"[TransformerAV] tokenizing {len(train_texts):,} train"
          f" + {len(val_texts) if val_texts else 0:,} val texts "
          f"(max_length={max_length}) …")
    # AMP is handled automatically inside train_transformer via GradScaler.
    model = train_transformer(
        model=model,
        tokenizer=tokenizer,
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=2,
        max_length=max_length,
        checkpoint_dir=ckpt_dir,
        resume_from_checkpoint=resume_from_checkpoint,
    )

    # Post-training: cleanup old checkpoints and verify integrity
    if is_main_process():
        _cleanup_old_checkpoints(ckpt_dir, keep_last=max_checkpoints)
        latest = ckpt_dir / "latest.ckpt"
        if latest.exists():
            ok = _verify_checkpoint(latest)
            print(f"[TransformerAV] latest checkpoint integrity: {'OK' if ok else 'CORRUPT'}")

    # Save — unwrap DDP/compile before persisting state_dict so the checkpoint
    # is portable (loadable on a single-GPU or CPU machine without DDP).
    if is_main_process():
        from nn_models import _unwrap_model
        path = _MODEL_A_DIR / "answer_verifier_transformer.pt"
        state = _unwrap_model(model).state_dict()
        torch.save(state, path)
        with open(_MODEL_A_DIR / "transformer_meta.json", "w") as f:
            json.dump({"model_name": model_name, "type": "answer_verifier"}, f)
        print(f"[TransformerAV] saved → {path}")
    return model


def train_question_generator(
    df: pd.DataFrame,
    model_name: str = "google/flan-t5-base",
    epochs: int = 5,
    # Seq2seq models are decoder-heavy; keep per-GPU batch modest.
    batch_size: int = 12,
    lr: float = 3e-5,
    use_lora: bool = True,
    lora_r: int = 8,
    fp16: bool = True,   # AMP handled internally via GradScaler
    max_rows: Optional[int] = None,
    checkpoint_dir: Optional[Path] = None,
    resume_from_checkpoint: Optional[Path] = None,
    max_checkpoints: int = 5,
):
    """Train a T5/FLAN-T5 question generator (dual-GPU optimized).

    Per-epoch checkpoints are saved to ``checkpoint_dir`` (default:
    ``models/model_b/transformer/checkpoints/``). Resume by passing
    ``resume_from_checkpoint=<path>/latest.ckpt``.
    """
    print("[QG] preparing data …")
    inputs, targets = prepare_question_generation_data(df, max_rows)
    split = int(len(inputs) * 0.9)
    tr_in, tr_tg = inputs[:split], targets[:split]
    val_in, val_tg = (inputs[split:], targets[split:]) if split < len(inputs) else (None, None)

    ckpt_dir = checkpoint_dir or (_MODEL_B_DIR / "checkpoints")
    if is_main_process():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"[QG] per-epoch checkpoints → {ckpt_dir}  (max per epoch: {max_checkpoints})")

    print(f"[QG] training {model_name}  |  train={len(tr_in):,}  val={len(val_in) if val_in else 0:,}")
    torch.cuda.empty_cache()
    model = QuestionGenerator(
        model_name=model_name,
        use_lora=use_lora,
        lora_r=lora_r,
    )

    model = _try_compile(model)
    model = _wrap_ddp(model)

    model = train_seq2seq(
        model=model,
        train_inputs=tr_in,
        train_targets=tr_tg,
        val_inputs=val_in,
        val_targets=val_tg,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=2,
        checkpoint_dir=ckpt_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        accumulation_steps=4,
    )

    # Post-training: cleanup old checkpoints and verify integrity
    if is_main_process():
        _cleanup_old_checkpoints(ckpt_dir, keep_last=max_checkpoints)
        latest = ckpt_dir / "latest.ckpt"
        if latest.exists():
            ok = _verify_checkpoint(latest)
            print(f"[QG] latest checkpoint integrity: {'OK' if ok else 'CORRUPT'}")

    if is_main_process():
        from nn_models import _unwrap_model
        path = _MODEL_B_DIR / "question_generator.pt"
        state = _unwrap_model(model).state_dict()
        torch.save(state, path)
        with open(_MODEL_B_DIR / "qg_meta.json", "w") as f:
            json.dump({"model_name": model_name, "type": "question_generator"}, f)
        print(f"[QG] saved → {path}")
    return model


def train_distractor_generator(
    df: pd.DataFrame,
    model_name: str = "facebook/bart-base",
    epochs: int = 5,
    batch_size: int = 12,
    lr: float = 3e-5,
    use_lora: bool = True,
    lora_r: int = 8,
    fp16: bool = True,   # AMP handled internally via GradScaler
    max_rows: Optional[int] = None,
    checkpoint_dir: Optional[Path] = None,
    resume_from_checkpoint: Optional[Path] = None,
    max_checkpoints: int = 5,
):
    """Train a BART/T5 distractor generator (dual-GPU optimized)."""
    print("[DG] preparing data …")
    inputs, targets = prepare_distractor_data(df, max_rows)
    split = int(len(inputs) * 0.9)
    tr_in, tr_tg = inputs[:split], targets[:split]
    val_in, val_tg = (inputs[split:], targets[split:]) if split < len(inputs) else (None, None)

    ckpt_dir = checkpoint_dir or (_MODEL_B_DIR / "checkpoints")
    if is_main_process():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DG] per-epoch checkpoints → {ckpt_dir}  (max kept: {max_checkpoints})")

    print(f"[DG] training {model_name}  |  train={len(tr_in):,}  val={len(val_in) if val_in else 0:,}")
    torch.cuda.empty_cache()
    model = DistractorGenerator(
        model_name=model_name,
        use_lora=use_lora,
        lora_r=lora_r,
    )

    model = _try_compile(model)
    model = _wrap_ddp(model)

    model = train_seq2seq(
        model=model,
        train_inputs=tr_in,
        train_targets=tr_tg,
        val_inputs=val_in,
        val_targets=val_tg,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=2,
        checkpoint_dir=ckpt_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        accumulation_steps=4,
    )

    # Post-training: cleanup old checkpoints and verify integrity
    if is_main_process():
        _cleanup_old_checkpoints(ckpt_dir, keep_last=max_checkpoints)
        latest = ckpt_dir / "latest.ckpt"
        if latest.exists():
            ok = _verify_checkpoint(latest)
            print(f"[DG] latest checkpoint integrity: {'OK' if ok else 'CORRUPT'}")

    if is_main_process():
        from nn_models import _unwrap_model
        path = _MODEL_B_DIR / "distractor_generator.pt"
        state = _unwrap_model(model).state_dict()
        torch.save(state, path)
        with open(_MODEL_B_DIR / "dg_meta.json", "w") as f:
            json.dump({"model_name": model_name, "type": "distractor_generator"}, f)
        print(f"[DG] saved → {path}")
    return model


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_all(
    csv_path: Optional[str] = None,
    model_a_model: str = "bert-base-uncased",
    qg_model: str = "google/flan-t5-base",
    dg_model: str = "facebook/bart-base",
    max_rows: Optional[int] = None,
    epochs_av: int = 3,
    epochs_qg: int = 5,
    epochs_dg: int = 5,
    fp16: bool = True,
    av_resume_from: Optional[str] = None,
    qg_resume_from: Optional[str] = None,
    dg_resume_from: Optional[str] = None,
    max_checkpoints: int = 5,
):
    """End-to-end training of all three transformer models."""
    # Load data
    path = Path(csv_path) if csv_path else _DATA_RAW / "train.csv"
    print(f"Loading data from {path}")
    df = _load_csv(path)
    if len(df) == 0:
        print("No data loaded. Aborting.")
        return

    split = int(len(df) * 0.85)
    train_df = df.iloc[:split]
    val_df = df.iloc[split:]
    print(f"Data: {len(df):,} rows  |  train={len(train_df):,}  val={len(val_df):,}")

    """1 is done, have the file, skipping this section"""
    # print("\n" + "=" * 60)
    # print("1. Training Transformer AnswerVerifier")
    # print("=" * 60)
    # train_answer_verifier_transformer(
    #     train_df, val_df,
    #     model_name=model_a_model,
    #     epochs=epochs_av,
    #     fp16=fp16,
    #     max_rows=max_rows,
    #     resume_from_checkpoint=Path(av_resume_from) if av_resume_from else None,
    #     max_checkpoints=max_checkpoints,
    # )

    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("2. Training Question Generator (T5/FLAN-T5)")
    print("=" * 60)
    train_question_generator(
        train_df,
        model_name=qg_model,
        epochs=epochs_qg,
        fp16=fp16,
        max_rows=max_rows,
        resume_from_checkpoint=Path(qg_resume_from) if qg_resume_from else None,
        max_checkpoints=max_checkpoints,
    )

    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("3. Training Distractor Generator (BART)")
    print("=" * 60)
    train_distractor_generator(
        train_df,
        model_name=dg_model,
        epochs=epochs_dg,
        fp16=fp16,
        max_rows=max_rows,
        resume_from_checkpoint=Path(dg_resume_from) if dg_resume_from else None,
        max_checkpoints=max_checkpoints,
    )

    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("Done. Models saved to:")
    print(f"  {_MODEL_A_DIR}")
    print(f"  {_MODEL_B_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train transformer models for MCQ pipeline")
    parser.add_argument("--csv", default=None, help="Path to training CSV")
    parser.add_argument("--av-model", default="bert-base-uncased")
    parser.add_argument("--qg-model", default="google/flan-t5-base")
    parser.add_argument("--dg-model", default="facebook/bart-base")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--epochs-av", type=int, default=3)
    parser.add_argument("--epochs-qg", type=int, default=5)
    parser.add_argument("--epochs-dg", type=int, default=5)
    parser.add_argument("--fp16", dest="fp16", action="store_true", default=True,
                        help="Use mixed-precision training (default on)")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false",
                        help="Disable mixed-precision training")
    parser.add_argument("--av-resume-from", default=None,
                        help="Path to a transformer checkpoint to resume answer-verifier training")
    parser.add_argument("--qg-resume-from", default=None,
                        help="Path to a transformer checkpoint to resume question-generator training")
    parser.add_argument("--dg-resume-from", default=None,
                        help="Path to a transformer checkpoint to resume distractor-generator training")
    parser.add_argument("--max-checkpoints", type=int, default=5,
                        help="Maximum number of epoch checkpoints to keep (default: 5)")
    args, _ = parser.parse_known_args()
    run_all(
        csv_path=args.csv,
        model_a_model=args.av_model,
        qg_model=args.qg_model,
        dg_model=args.dg_model,
        max_rows=args.max_rows,
        epochs_av=args.epochs_av,
        epochs_qg=args.epochs_qg,
        epochs_dg=args.epochs_dg,
        fp16=args.fp16,
        av_resume_from=args.av_resume_from,
        qg_resume_from=args.qg_resume_from,
        dg_resume_from=args.dg_resume_from,
        max_checkpoints=args.max_checkpoints,
    )