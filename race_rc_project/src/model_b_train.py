"""
scorer_b.py  —  distractor & hint scorers (batched TF-IDF, logistic regression)
"""
from __future__ import annotations

import os, pickle, sys
from collections import Counter
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# from preprocessing import (
#     ARTIFACT_DIR,
#     sentence_fragments,
#     word_tokens,
#     normalize,
# )

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

# Local development paths (commented)
# _THIS_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "/kaggle/working/data"
# ARTIFACT_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "data", "processed"))
# _SCORER_OUT = os.path.normpath(os.path.join(_THIS_DIR, "..", "models", "model_b", "traditional"))

# Kaggle paths (active)
_THIS_DIR = "/kaggle/working/src"
ARTIFACT_DIR = "/kaggle/working/data/processed"
_SCORER_OUT = "/kaggle/working/models/model_b/traditional"

sys.path.insert(0, _THIS_DIR)
os.makedirs(_SCORER_OUT, exist_ok=True)

with open(os.path.join(ARTIFACT_DIR, "tfidf_vectorizer.pkl"), "rb") as _f:
    _TFIDF = pickle.load(_f)

# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def _batch_row_cosine(mat_a, mat_b) -> np.ndarray:
    """Row-wise cosine similarity between two same-shape sparse matrices."""
    dot    = np.array(mat_a.multiply(mat_b).sum(axis=1)).flatten()
    norm_a = np.sqrt(np.array(mat_a.power(2).sum(axis=1)).flatten())
    norm_b = np.sqrt(np.array(mat_b.power(2).sum(axis=1)).flatten())
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(norm_a * norm_b > 0, dot / (norm_a * norm_b), 0.0)

def _batch_transform(*lists) -> list:
    """Transform multiple string lists in one tqdm loop; returns list of sparse matrices."""
    mats, names = [], ["cand/sent", "gold/question", "passage"]
    for i, lst in enumerate(tqdm(lists, desc="  transform", unit="matrix")):
        mats.append(_TFIDF.transform(lst))
    return mats

# ---------------------------------------------------------------------------
# Keyword / candidate helpers
# ---------------------------------------------------------------------------

_OPTIONS, _NEG_PER_ITEM, _POOL_SIZE = ["A", "B", "C", "D"], 3, 15

def top_keywords(passage: str, k: int = 10) -> List[str]:
    viable = [t for t in word_tokens(passage) if len(t) >= 4]
    return [w for w, _ in Counter(viable).most_common(k)]

def _candidate_pool(passage: str) -> List[str]:
    return [w for w in top_keywords(passage, k=_POOL_SIZE) if w]

def _phrase_relevance(phrase: str, passage: str, gold: str) -> float:
    return passage.lower().count(phrase.lower()) * 0.7 + len(set(phrase.split()) & set(gold.split())) * 0.3

# ---------------------------------------------------------------------------
# Feature vectors  (pre-computed similarities passed in as scalars)
# ---------------------------------------------------------------------------

def _dist_feats(cand: str, gold: str, passage: str, sim_cg: float, sim_cp: float) -> List[float]:
    n = len(word_tokens(passage))
    return [
        sim_cg,
        sim_cp,
        len(set(cand) & set(gold)) / (len(gold) + 1e-9),
        passage.lower().count(cand.lower()) / (n + 1e-9),
        min(len(cand) / 20.0, 1.0),
        _phrase_relevance(cand, passage, gold),
    ]

def _hint_feats(sent: str, question: str, idx: int, total: int, sim: float) -> List[float]:
    q_toks, s_toks = set(word_tokens(question)), set(word_tokens(sent))
    return [len(q_toks & s_toks) / (len(q_toks) + 1e-9), idx / max(total - 1, 1), len(s_toks) / 50.0, sim]

# ---------------------------------------------------------------------------
# Dataset builders  — BATCHED
# ---------------------------------------------------------------------------

def _build_distractor_rows(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    cands, golds, passages, labels = [], [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  distractor collect", unit="row"):
        passage   = str(row["article"])
        gold_key  = str(row["answer"])
        gold_text = str(row[gold_key])
        wrong     = [str(row[o]) for o in _OPTIONS if o != gold_key]

        for opt in wrong:
            cands.append(opt); golds.append(gold_text); passages.append(passage); labels.append(1)

        kws = [kw for kw in _candidate_pool(passage) if 3 <= len(kw) <= 15 and kw not in wrong]
        for kw in kws[:_NEG_PER_ITEM]:
            cands.append(kw); golds.append(gold_text); passages.append(passage); labels.append(0)

    mat_c, mat_g, mat_p = _batch_transform(cands, golds, passages)
    sim_cg = _batch_row_cosine(mat_c, mat_g)
    sim_cp = _batch_row_cosine(mat_c, mat_p)

    X = np.array(
        [_dist_feats(cands[i], golds[i], passages[i], sim_cg[i], sim_cp[i])
         for i in tqdm(range(len(cands)), desc="  distractor features", unit="ex")],
        dtype=np.float32,
    )
    return X, np.array(labels, dtype=np.int32)


def _build_hint_rows(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    sentences, questions, sent_indices, total_counts = [], [], [], []
    group_starts, group_sizes = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  hint collect", unit="row"):
        sents = sentence_fragments(str(row["article"]))
        if not sents:
            continue
        q = str(row["question"])
        start = len(sentences)
        for idx, s in enumerate(sents):
            sentences.append(s); questions.append(q)
            sent_indices.append(idx); total_counts.append(len(sents))
        group_starts.append(start); group_sizes.append(len(sents))

    mat_s, mat_q = _batch_transform(sentences, questions)
    sims = _batch_row_cosine(mat_s, mat_q)

    labels = np.zeros(len(sentences), dtype=np.int32)
    for start, size in zip(group_starts, group_sizes):
        labels[start + int(np.argmax(sims[start:start + size]))] = 1

    X = np.array(
        [_hint_feats(sentences[i], questions[i], sent_indices[i], total_counts[i], float(sims[i]))
         for i in tqdm(range(len(sentences)), desc="  hint features", unit="ex")],
        dtype=np.float32,
    )
    return X, labels

# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def _fit(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(max_iter=1_500, C=2.0, class_weight="balanced")
    return clf.fit(X, y)

fit_distractor_scorer = fit_hint_scorer = _fit   # same hyperparams for both

def _eval(clf, X, y) -> dict:
    p = clf.predict(X)
    return {
        "accuracy":  accuracy_score(y, p),
        "f1":        f1_score(y, p, zero_division=0),
        "precision": precision_score(y, p, zero_division=0),
        "recall":    recall_score(y, p, zero_division=0),
    }

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

_DEFAULT_N     = 3
_FALLBACK_HINTS = ["Look for the main detail", "Scan for key terms", "Use the passage context"]


def pick_distractors(passage: str, correct_answer: str, scorer: LogisticRegression, n: int = _DEFAULT_N) -> List[str]:
    pool = _candidate_pool(passage)
    if not pool:
        return []
    m = len(pool)
    mat_p, mat_g, mat_a = _batch_transform(pool, [correct_answer] * m, [passage] * m)
    X = np.array([_dist_feats(pool[i], correct_answer, passage, float(_batch_row_cosine(mat_p, mat_g)[i]),
                               float(_batch_row_cosine(mat_p, mat_a)[i])) for i in range(m)], dtype=np.float32)
    ranked = sorted(zip(scorer.predict_proba(X)[:, 1], pool), reverse=True)
    selected, seen = [], set()
    for _, cand in ranked:
        if cand[:4] not in seen:
            seen.add(cand[:4]); selected.append(cand)
        if len(selected) >= n:
            break
    return selected


def pick_hints(passage: str, question: str, scorer: LogisticRegression, n: int = _DEFAULT_N) -> List[str]:
    sents = sentence_fragments(passage)
    if not sents:
        return _FALLBACK_HINTS[:]
    mat_s, mat_q = _batch_transform(sents, [question] * len(sents))
    sims = _batch_row_cosine(mat_s, mat_q)
    X = np.array([_hint_feats(sents[i], question, i, len(sents), float(sims[i])) for i in range(len(sents))], dtype=np.float32)
    top = [s for _, s in sorted(zip(scorer.predict_proba(X)[:, 1], sents), reverse=True)[:n]]
    hints = []
    if top:       hints.append(f"Hint 1: Think about {' '.join(top_keywords(top[-1], k=4))}")
    if len(top)>1: hints.append(f"Hint 2: {top[1][:120]}")
    if len(top)>2: hints.append(f"Hint 3: {top[0][:150]}")
    return hints

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run():
    SEP = "=" * 42
    print(f"[scorer_b] starting\n{SEP}")

    # Local paths (commented)
    # train_df = pd.read_csv("../data/processed/train_clean.csv")
    # val_df   = pd.read_csv("../data/processed/val_clean.csv")
    
    # Kaggle paths (active)
    train_df = pd.read_csv("/kaggle/working/data/processed/train_clean.csv")
    val_df   = pd.read_csv("/kaggle/working/data/processed/val_clean.csv")
    print(f"train: {len(train_df):,}  |  val: {len(val_df):,}")

    print("\n--- distractor ---")
    X_d, y_d = _build_distractor_rows(train_df)
    print(f"  matrix {X_d.shape}  positives: {y_d.sum():,}")

    print("\n--- hint ---")
    X_h, y_h = _build_hint_rows(train_df)
    print(f"  matrix {X_h.shape}  positives: {y_h.sum():,}")

    print("\n--- fit ---")
    dist_clf = _fit(X_d, y_d)
    hint_clf = _fit(X_h, y_h)

    # eval with metrics collection
    metrics_rows = []
    for split, build in [("train", lambda: (X_d, y_d, X_h, y_h)),
                          ("val",   lambda: (*_build_distractor_rows(val_df), *_build_hint_rows(val_df)))]:
        Xd, yd, Xh, yh = build()
        for name, clf, X, y in [("distractor", dist_clf, Xd, yd), ("hint", hint_clf, Xh, yh)]:
            m = _eval(clf, X, y)
            metrics_rows.append({
                "model": name,
                "split": split,
                "accuracy": m["accuracy"],
                "f1": m["f1"],
                "precision": m["precision"],
                "recall": m["recall"]
            })
            print(f"  [{name} {split}]  " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))

    joblib.dump(dist_clf, os.path.join(_SCORER_OUT, "distractor.pkl"))
    joblib.dump(hint_clf, os.path.join(_SCORER_OUT, "hint.pkl"))
    
    # Export metrics to CSV
    _export_metrics_to_csv(metrics_rows)
    
    print(f"\n[scorer_b] done — saved to {_SCORER_OUT}\n{SEP}")


def _export_metrics_to_csv(metrics_rows: list):
    """Export Model-B metrics to CSV."""
    import os
    # Local path (commented)
    # reports_dir = os.path.normpath(os.path.join(_THIS_DIR, "..", "data", "processed", "reports"))
    
    # Kaggle path (active)
    reports_dir = "/kaggle/working/data/processed/reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    df = pd.DataFrame(metrics_rows)
    csv_path = os.path.join(reports_dir, "model_b_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"    → {csv_path}")


if __name__ == "__main__":
    run()