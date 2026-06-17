"""
nn_models.py — PyTorch neural network models & training utilities.
Replaces scikit-learn classifiers (Logistic Regression, SVM, meta-learner)
with feed-forward networks for the RACE RC project.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import issparse
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import sys as _sys
print(f"[nn_models] device={DEVICE}", file=_sys.stderr)
if torch.cuda.is_available():
    print(f"[nn_models]   GPU: {torch.cuda.get_device_name(0)}", file=_sys.stderr)
    print(f"[nn_models]   CUDA: {torch.version.cuda}", file=_sys.stderr)


# ---------------------------------------------------------------------------
# Model unwrapping utility
# ---------------------------------------------------------------------------

def _unwrap_model(model: nn.Module) -> nn.Module:
    """
    Peel off DataParallel, DistributedDataParallel, and torch.compile
    (OptimizedModule) wrappers to get the original nn.Module back.

    Call this whenever you need to access custom attributes (.tokenizer,
    .max_input_length, etc.) or save a portable state_dict.

    Safe to call on an already-unwrapped model.
    """
    # torch.compile wraps the model in torch._dynamo.eval_frame.OptimizedModule
    # which exposes the original via ._orig_mod
    while hasattr(model, "_orig_mod"):
        model = model._orig_mod
    # DataParallel / DistributedDataParallel expose the original via .module
    while isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model = model.module
    # A compile wrapper may sit on top of a DataParallel, so alternate until stable
    # (the while loops above handle arbitrary nesting already)
    return model


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class SparseDataset(Dataset):
    """Wraps a sparse CSR matrix (or dense array) with binary labels."""

    def __init__(self, X, y):
        self.X = X
        self.y = np.asarray(y, dtype=np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        row = self.X[idx]
        if issparse(row):
            x = row.toarray().flatten().astype(np.float32)
        else:
            x = np.asarray(row).flatten().astype(np.float32)
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Base mixin providing sklearn-compatible predict / predict_proba
# ---------------------------------------------------------------------------

class _BaseNN(nn.Module):
    """Adds .predict() and .predict_proba() so callers need minimal changes."""

    def predict_proba(self, X, batch_size=1024):
        """Return (N, 2) — [P(0), P(1)] matching sklearn's interface."""
        self.eval()
        ds = SparseDataset(X, np.zeros(X.shape[0], dtype=np.float32))
        ld = DataLoader(ds, batch_size=batch_size, shuffle=False)
        all_p1 = []
        with torch.no_grad():
            for bx, _ in ld:
                logits = self.forward(bx.to(DEVICE))
                all_p1.append(torch.sigmoid(logits).cpu().numpy())
        p1 = np.concatenate(all_p1)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X, batch_size=1024):
        """Return binary class predictions (0 / 1)."""
        return np.argmax(self.predict_proba(X, batch_size), axis=1)


# ---------------------------------------------------------------------------
# Model A  —  Answer Verifier (binary classifier on OHE features)
# ---------------------------------------------------------------------------

class AnswerVerifier(_BaseNN):
    """
    Feed-forward network for binary answer verification.

    Input   : OHE / bag-of-words vector (default 10 000 dims)
    Arch    : Linear → BN → ReLU → Dropout → … → Linear(→1)
    Output  : logit (fed into sigmoid in predict_proba)
    """

    def __init__(self, input_dim: int, hidden_dims=None, dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 128]
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Model B  —  Distractor Scorer & Hint Scorer (small NNs on handcrafted feats)
# ---------------------------------------------------------------------------

class DistractorScorer(_BaseNN):
    """Small network for distractor scoring (6 handcrafted features)."""

    def __init__(self, input_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class HintScorer(_BaseNN):
    """Small network for hint scoring (4 handcrafted features)."""

    def __init__(self, input_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total = 0.0
    use_amp = device.type == "cuda" and scaler is not None
    for bx, by in loader:
        bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
        optimizer.zero_grad()
        if use_amp:
            with autocast("cuda"):
                loss = criterion(model(bx), by)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        total += loss.item() * bx.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def _eval_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for bx, by in loader:
        bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
        logits = model(bx)
        total_loss += criterion(logits, by).item() * bx.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        all_preds.append(preds.cpu().numpy())
        all_labels.append(by.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return total_loss / len(loader.dataset), float((preds == labels).mean()), preds


def _checkpoint_payload(
    model: nn.Module,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    scheduler,
    best_loss: float,
    best_state,
    stall: int,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": _unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_loss": best_loss,
        "best_state_dict": best_state,
        "stall": stall,
        "meta": extra_meta or {},
    }


def _save_epoch_checkpoint(
    checkpoint_dir: str | Path,
    model: nn.Module,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    scheduler,
    best_loss: float,
    best_state,
    stall: int,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = _checkpoint_payload(
        model=model,
        epoch=epoch,
        optimizer=optimizer,
        scheduler=scheduler,
        best_loss=best_loss,
        best_state=best_state,
        stall=stall,
        extra_meta=extra_meta,
    )
    epoch_path = checkpoint_dir / f"epoch_{epoch:04d}.ckpt"
    latest_path = checkpoint_dir / "latest.ckpt"
    torch.save(payload, epoch_path)
    torch.save(payload, latest_path)
    print(f"  checkpoint saved → {epoch_path}")
    return latest_path


def _load_training_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return 1, float("inf"), None, 0

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    _unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
    if ckpt.get("optimizer_state_dict"):
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler_state = ckpt.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state:
        scheduler.load_state_dict(scheduler_state)

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_loss = float(ckpt.get("best_loss", float("inf")))
    best_state = ckpt.get("best_state_dict")
    stall = int(ckpt.get("stall", 0))
    return start_epoch, best_loss, best_state, stall


def train_nn(
    model: nn.Module,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    pos_weight: Optional[float] = None,
    patience: int = 5,
    checkpoint_dir: str | Path | None = None,
    verbose: bool = True,
) -> nn.Module:
    """
    Full training loop with early stopping.

    Parameters
    ----------
    model        : a PyTorch module (must have forward returning logits)
    X_train, y_train : training data (sparse or dense)
    X_val, y_val : optional validation data
    epochs       : maximum number of epochs
    batch_size   : mini-batch size
    lr, weight_decay : AdamW hyper-parameters
    pos_weight   : weight for the positive class in BCEWithLogitsLoss
                   (set to neg_count / pos_count for imbalanced data)
    patience     : early stopping patience (validation loss)
    """
    device = DEVICE
    nw = 2 if device.type == "cuda" else 0
    train_ds = SparseDataset(X_train, y_train)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=(nw > 0), num_workers=nw)

    val_ld = None
    if X_val is not None and y_val is not None:
        val_ds = SparseDataset(X_val, y_val)
        val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=(nw > 0), num_workers=nw)

    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    scaler = GradScaler("cuda") if device.type == "cuda" else None
    model = model.to(device)
    best_loss = float("inf")
    best_state = None
    stall = 0

    pbar = tqdm(range(1, epochs + 1), desc="  epochs", unit="ep", disable=not verbose)
    for epoch in pbar:
        tr_loss = _train_epoch(model, train_ld, criterion, optimizer, device, scaler)
        scheduler.step()

        if val_ld is not None:
            val_loss, val_acc, _ = _eval_model(model, val_ld, criterion, device)
            pbar.set_postfix(tr_loss=f"{tr_loss:.4f}", val_loss=f"{val_loss:.4f}", val_acc=f"{val_acc:.4f}")
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                stall = 0
            else:
                stall += 1
                if stall >= patience:
                    if verbose:
                        print(f"  early stopping at epoch {epoch}")
                    break
        else:
            pbar.set_postfix(tr_loss=f"{tr_loss:.4f}")

        if checkpoint_dir is not None:
            _save_epoch_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=model,
                epoch=epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                best_loss=best_loss,
                best_state=best_state,
                stall=stall,
                extra_meta={"trainer": "nn", "epoch": epoch},
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def save_checkpoint(model: nn.Module, path: str, extra_meta: Optional[Dict[str, Any]] = None):
    """Save model state_dict + metadata."""
    base = _unwrap_model(model)
    meta: Dict[str, Any] = {
        "input_dim": (
            base.net[0].in_features
            if hasattr(base, "net") and hasattr(base.net[0], "in_features")
            else None
        )
    }
    if extra_meta:
        meta.update(extra_meta)
    torch.save({"model_state_dict": base.state_dict(), "meta": meta}, path)


def load_checkpoint(model_class, path: str, **model_kwargs):
    """Load a saved checkpoint into a fresh model instance."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    meta = ckpt.get("meta", {})
    input_dim = meta.get("input_dim") or model_kwargs.pop("input_dim", None)
    if input_dim is None and "input_dim" not in model_kwargs:
        raise ValueError(
            "input_dim must be provided via model_kwargs or checkpoint meta"
        )
    model = model_class(input_dim=input_dim, **model_kwargs)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


# ===================================================================
# Transformer-based models  (BERT / T5 / BART  for MCQ pipeline)
# ===================================================================


class TransformerAnswerVerifier(nn.Module):
    """
    BERT/RoBERTa-based answer verifier.
    Encodes (article, question, option) and scores P(correct).
    Supports LoRA via model_name="<path>/lora-adapter" or HF hub.

    NOTE: DataParallel wrapping is handled externally by transformer_train.py.
          Do NOT wrap self.encoder with DataParallel here — the outer wrapper
          in _wrap_dataparallel() already covers it.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        max_length: int = 384,
        use_lora: bool = False,
        lora_r: int = 8,
    ):
        super().__init__()
        from transformers import AutoConfig, AutoModelForSequenceClassification

        self.max_length = max_length
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.encoder = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=self.config
        )

        if use_lora:
            try:
                from peft import LoraConfig, get_peft_model

                lora_cfg = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_r * 2,
                    target_modules=["query", "value"],
                    lora_dropout=0.1,
                    bias="none",
                )
                self.encoder = get_peft_model(self.encoder, lora_cfg)
                self.encoder.print_trainable_parameters()
            except ImportError:
                print("  WARNING: peft not installed. Training full model.")

        # ── Removed internal DataParallel wrap ──────────────────────────────
        # transformer_train.py calls _wrap_dataparallel() on the whole model
        # object after construction, which is the correct place to do it.
        # Wrapping self.encoder here AND then wrapping the whole model in
        # transformer_train.py produces nested DataParallel which breaks
        # attribute access and doubles GPU memory overhead.

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return out


class TextDataset(Dataset):
    """Simple text-pair dataset for transformer training.

    Tokenises in chunks (default 10 000) with a progress bar so that
    large corpora don't appear to "hang" and don't OOM.
    """

    _CHUNK = 10_000  # texts per tokenizer call

    def __init__(self, texts, labels, tokenizer, max_length=384):
        self.labels = torch.tensor(labels, dtype=torch.long)

        all_ids, all_mask = [], []
        n = len(texts)
        chunks = range(0, n, self._CHUNK)
        for start in tqdm(chunks, desc="  tokenizing", unit="chunk"):
            end = min(start + self._CHUNK, n)
            enc = tokenizer(
                texts[start:end],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            all_ids.append(enc["input_ids"])
            all_mask.append(enc["attention_mask"])

        self.input_ids = torch.cat(all_ids, dim=0)
        self.attention_mask = torch.cat(all_mask, dim=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def train_transformer(
    model: nn.Module,
    tokenizer,
    train_texts,
    train_labels,
    val_texts=None,
    val_labels=None,
    epochs: int = 5,
    batch_size: int = 48,
    lr: float = 2e-5,
    max_length: int = 384,
    patience: int = 2,
    checkpoint_dir: str | Path = "./checkpoints",
    resume_from_checkpoint: str | Path | None = None,
    verbose: bool = True,
):
    """Training loop for TransformerAnswerVerifier with early stopping."""
    device = DEVICE
    model = model.to(device)
    n_gpus = max(torch.cuda.device_count(), 1)
    nw = min(4 * n_gpus, os.cpu_count() or 1, 8)

    train_ds = TextDataset(train_texts, train_labels, tokenizer, max_length)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          pin_memory=True, num_workers=nw,
                          prefetch_factor=2, persistent_workers=True)

    val_ld = None
    if val_texts is not None and val_labels is not None:
        val_ds = TextDataset(val_texts, val_labels, tokenizer, max_length)
        val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            pin_memory=True, num_workers=nw,
                            prefetch_factor=2, persistent_workers=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler("cuda") if device.type == "cuda" else None
    use_amp = scaler is not None

    start_epoch = 1
    best_loss = float("inf")
    best_state = None
    stall = 0
    n_batches = len(train_ld)

    if resume_from_checkpoint is not None:
        start_epoch, best_loss, best_state, stall = _load_training_checkpoint(
            resume_from_checkpoint,
            model,
            optimizer,
            scheduler,
        )
        if verbose:
            tqdm.write(f"  resumed from epoch {start_epoch - 1} using {resume_from_checkpoint}")

    if start_epoch > epochs:
        if verbose:
            tqdm.write(f"  checkpoint is already at epoch {start_epoch - 1}; nothing to do")
        if best_state is not None:
            _unwrap_model(model).load_state_dict(best_state)
        return model

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        tr_loss = 0.0
        pbar = tqdm(train_ld, total=n_batches, desc=f"  epoch {epoch}/{epochs}", unit="batch", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad()
            if use_amp:
                with autocast("cuda"):
                    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                scaler.scale(out.loss.mean()).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                out.loss.mean().backward()
                optimizer.step()
            tr_loss += out.loss.mean().item() * input_ids.size(0)
            pbar.set_postfix(loss=f"{out.loss.mean().item():.4f}")
        scheduler.step()
        tr_loss /= len(train_ld.dataset)

        if val_ld is not None:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_ld:
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                    labels = batch["labels"].to(device, non_blocking=True)
                    if use_amp:
                        with autocast("cuda"):
                            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    else:
                        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss += out.loss.mean().item() * input_ids.size(0)
                    preds = out.logits.argmax(-1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            val_loss /= len(val_ld.dataset)
            val_acc = correct / total if total > 0 else 0.0

            if verbose:
                tqdm.write(f"  epoch {epoch:2d}/{epochs}  tr_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                stall = 0
            else:
                stall += 1
        else:
            if verbose:
                tqdm.write(f"  epoch {epoch:2d}/{epochs}  tr_loss={tr_loss:.4f}")

        if checkpoint_dir is not None:
            _save_epoch_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=model,
                epoch=epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                best_loss=best_loss,
                best_state=best_state,
                stall=stall,
                extra_meta={"trainer": "transformer", "epoch": epoch},
            )

        if val_ld is not None and stall >= patience:
            if verbose:
                tqdm.write(f"  early stopping at epoch {epoch}")
            break

    if best_state is not None:
        _unwrap_model(model).load_state_dict(best_state)
    return model


# ===================================================================
# Seq2Seq models for MCQ generation (T5 / BART / FLAN-T5)
# ===================================================================


class QuestionGenerator(nn.Module):
    """
    T5/FLAN-T5 based question generator.
    Input  : "context: ... answer: ..."
    Output : generated question text
    Supports LoRA.

    NOTE: DataParallel wrapping is handled externally by transformer_train.py.
          Do NOT wrap self.model with DataParallel here.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        max_input_length: int = 384,
        max_output_length: int = 64,
        use_lora: bool = False,
        lora_r: int = 8,
    ):
        super().__init__()
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        if use_lora:
            try:
                from peft import LoraConfig, get_peft_model

                lora_cfg = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_r * 2,
                    target_modules=["q", "v"],
                    lora_dropout=0.1,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_cfg)
                self.model.print_trainable_parameters()
            except ImportError:
                print("  WARNING: peft not installed. Training full model.")

        # ── Removed internal DataParallel wrap ──────────────────────────────
        # See TransformerAnswerVerifier note above.

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def generate_question(self, context: str, answer: str) -> str:
        # Always unwrap to get the raw HF model for .generate()
        base = _unwrap_model(self)
        prompt = f"context: {context} answer: {answer}"
        inputs = base.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=base.max_input_length,
        ).to(next(base.model.parameters()).device)
        outputs = base.model.generate(
            **inputs,
            max_new_tokens=base.max_output_length,
            num_beams=4,
            temperature=0.7,
        )
        return base.tokenizer.decode(outputs[0], skip_special_tokens=True)


class DistractorGenerator(nn.Module):
    """
    T5/BART based distractor generator.
    Input  : "question: ... correct: ..."
    Output : distractor texts separated by |
    Supports LoRA.

    NOTE: DataParallel wrapping is handled externally by transformer_train.py.
          Do NOT wrap self.model with DataParallel here.
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-base",
        max_input_length: int = 256,
        max_output_length: int = 96,
        use_lora: bool = False,
        lora_r: int = 8,
    ):
        super().__init__()
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        if use_lora:
            try:
                from peft import LoraConfig, get_peft_model

                lora_cfg = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_r * 2,
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=0.1,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_cfg)
                self.model.print_trainable_parameters()
            except ImportError:
                print("  WARNING: peft not installed. Training full model.")

        # ── Removed internal DataParallel wrap ──────────────────────────────
        # See TransformerAnswerVerifier note above.

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def generate_distractors(self, question: str, correct_answer: str, n: int = 3) -> list:
        # Always unwrap to get the raw HF model for .generate()
        base = _unwrap_model(self)
        prompt = f"question: {question} correct: {correct_answer}"
        inputs = base.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=base.max_input_length,
        ).to(next(base.model.parameters()).device)
        outputs = base.model.generate(
            **inputs,
            max_new_tokens=base.max_output_length,
            num_beams=4,
            temperature=0.8,
        )
        text = base.tokenizer.decode(outputs[0], skip_special_tokens=True)
        distractors = [d.strip() for d in text.split("|") if d.strip()]
        while len(distractors) < n:
            distractors.append(f"option {len(distractors) + 1}")
        return distractors[:n]


def train_seq2seq(
    model: nn.Module,
    train_inputs: list,
    train_targets: list,
    val_inputs: list = None,
    val_targets: list = None,
    epochs: int = 10,
    batch_size: int = 24,
    lr: float = 3e-5,
    patience: int = 2,
    checkpoint_dir: str | Path = "./checkpoints",
    resume_from_checkpoint: str | Path | None = None,
    verbose: bool = True,
):
    """
    Training loop for seq2seq models (QuestionGenerator / DistractorGenerator).
    Each input is raw text; each target is the expected output text.

    `model` may be wrapped with DataParallel and/or torch.compile by the caller
    (transformer_train.py).  We unwrap here to safely access .tokenizer,
    .max_input_length, and .max_output_length before the training loop begins.
    """
    device = DEVICE
    model = model.to(device)

    # ── Unwrap to access custom attributes safely ───────────────────────────
    # After _wrap_dataparallel() and _try_compile() in transformer_train.py,
    # `model` may be:
    #   OptimizedModule(_orig_mod=DataParallel(module=QuestionGenerator(...)))
    # _unwrap_model() peels all layers to reach the original QuestionGenerator.
    base_model = _unwrap_model(model)
    tokenizer = base_model.tokenizer
    max_in = base_model.max_input_length
    max_out = base_model.max_output_length

    n_gpus = max(torch.cuda.device_count(), 1)
    nw = min(4 * n_gpus, os.cpu_count() or 2, 8) if device.type == "cuda" else 0

    def _encode(texts, max_len):
        return tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_len, return_tensors="pt",
        )

    train_enc = _encode(train_inputs, max_in)
    train_labels = _encode(train_targets, max_out).input_ids
    train_labels[train_labels == tokenizer.pad_token_id] = -100

    val_data = None
    if val_inputs and val_targets:
        val_enc = _encode(val_inputs, max_in)
        val_labels = _encode(val_targets, max_out).input_ids
        val_labels[val_labels == tokenizer.pad_token_id] = -100
        val_data = (val_enc, val_labels)

    class _Seq2SeqDataset(Dataset):
        def __init__(self, enc, labels):
            self.enc = enc
            self.labels = labels

        def __len__(self):
            return self.enc.input_ids.size(0)

        def __getitem__(self, idx):
            return {
                "input_ids": self.enc.input_ids[idx],
                "attention_mask": self.enc.attention_mask[idx],
                "labels": self.labels[idx],
            }

    train_ds = _Seq2SeqDataset(train_enc, train_labels)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          pin_memory=(nw > 0), num_workers=nw,
                          prefetch_factor=2, persistent_workers=True) if nw > 0 else \
               DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          pin_memory=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler("cuda") if device.type == "cuda" else None
    use_amp = scaler is not None

    start_epoch = 1
    best_loss = float("inf")
    best_state = None
    stall = 0
    n_batches = len(train_ld)

    if resume_from_checkpoint is not None:
        start_epoch, best_loss, best_state, stall = _load_training_checkpoint(
            resume_from_checkpoint,
            model,
            optimizer,
            scheduler,
        )
        if verbose:
            tqdm.write(f"  resumed from epoch {start_epoch - 1} using {resume_from_checkpoint}")

    if start_epoch > epochs:
        if verbose:
            tqdm.write(f"  checkpoint is already at epoch {start_epoch - 1}; nothing to do")
        if best_state is not None:
            _unwrap_model(model).load_state_dict(best_state)
        return model

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        tr_loss = 0.0
        pbar = tqdm(train_ld, total=n_batches, desc=f"  epoch {epoch}/{epochs}", unit="batch", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad()
            if use_amp:
                with autocast("cuda"):
                    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                scaler.scale(out.loss.mean()).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                out.loss.mean().backward()
                optimizer.step()
            tr_loss += out.loss.mean().item() * input_ids.size(0)
            pbar.set_postfix(loss=f"{out.loss.mean().item():.4f}")
        tr_loss /= len(train_ld.dataset)
        scheduler.step()

        if val_data is not None:
            model.eval()
            val_enc, val_lbl = val_data
            val_loss = 0.0
            with torch.no_grad():
                if use_amp:
                    with autocast("cuda"):
                        out = model(
                            input_ids=val_enc.input_ids.to(device, non_blocking=True),
                            attention_mask=val_enc.attention_mask.to(device, non_blocking=True),
                            labels=val_lbl.to(device, non_blocking=True),
                        )
                else:
                    out = model(
                        input_ids=val_enc.input_ids.to(device, non_blocking=True),
                        attention_mask=val_enc.attention_mask.to(device, non_blocking=True),
                        labels=val_lbl.to(device, non_blocking=True),
                    )
                val_loss = out.loss.mean().item()
            if verbose:
                tqdm.write(f"  epoch {epoch:2d}/{epochs}  tr_loss={tr_loss:.4f}  val_loss={val_loss:.4f}")
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                stall = 0
            else:
                stall += 1
        else:
            if verbose:
                tqdm.write(f"  epoch {epoch:2d}/{epochs}  tr_loss={tr_loss:.4f}")

        if checkpoint_dir is not None:
            _save_epoch_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=model,
                epoch=epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                best_loss=best_loss,
                best_state=best_state,
                stall=stall,
                extra_meta={"trainer": "seq2seq", "epoch": epoch},
            )

        if val_data is not None and stall >= patience:
            if verbose:
                tqdm.write(f"  early stopping at epoch {epoch}")
            break

    if best_state is not None:
        _unwrap_model(model).load_state_dict(best_state)
    return model