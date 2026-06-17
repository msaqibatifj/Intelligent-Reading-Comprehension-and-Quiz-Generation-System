"""
model_a_train.py
================
Answer Verification + MCQ Generation  (Model A)

Models:
  AnswerVerifier (feed-forward NN on OHE features)
  TransformerAnswerVerifier (BERT/RoBERTa + LoRA)  — optional

MCQ generation pipeline:
  Phase 1 — candidate sentence extraction (keyword-overlap scoring)
  Phase 2 — Wh-word template instantiation
  Phase 3 — ML / heuristic question ranking
  Phase 4 — distractor assembly from article sentences

Evaluation:
  Binary classification: Accuracy, Precision, Recall, Macro-F1
  4-way MCQ accuracy
  Cosine-sim accuracy (TF-IDF)
  Text generation: BLEU, ROUGE-1/2/L, METEOR
"""

from __future__ import annotations

import math
import os
import pickle
import random
import re
import sys
import warnings
from collections import Counter
from itertools import chain
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.sparse import load_npz, hstack as sparse_hstack
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity as _sk_cos
from tqdm import tqdm

from src.nn_models import (
    AnswerVerifier,
    DistractorScorer,
    HintScorer,
    DEVICE,
    load_checkpoint,
    save_checkpoint,
    train_nn,
)
from src.preprocessing import clean_text

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

# Local development path (commented)
# _THIS_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "/kaggle/working/data"
# ARTIFACT_DIR = os.path.join(_THIS_DIR, "..", "data", "processed")

# Local paths
try:
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_PROJ  = os.path.normpath(os.path.join(_THIS_DIR, ".."))
except NameError:
    BASE_PROJ = os.getcwd()
ARTIFACT_DIR = os.path.join(BASE_PROJ, "data", "processed")
PROCESSED_DIR = ARTIFACT_DIR

sys.path.insert(0, _THIS_DIR)
# try:
#     from preprocessing import (
#         ARTIFACT_DIR,
#         normalize,
#         word_tokens,
#         sentence_fragments,
#         dot_cosine,
#         vec_cosine,
#         build_model_a_dataset,
#         apply_vectorizer,
#         make_sample_string,
#     )
#     PROCESSED_DIR = ARTIFACT_DIR
# except ImportError:
#     # Minimal stubs so the module loads without preprocessing.py
#     ARTIFACT_DIR  = os.path.join(_THIS_DIR, "processed")
#     PROCESSED_DIR = ARTIFACT_DIR
_SW = frozenset({
    "a", "an", "the", "is", "it", "in", "of", "to", "and", "or", "for",
    "on", "with", "as", "at", "by", "be", "was", "are", "were", "this",
    "that", "from", "but", "not", "have", "has", "had", "he", "she",
    "they", "we", "you", "i", "do", "did", "will", "its", "their",
})
def normalize(raw) -> str:
    s = str(raw).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()
def word_tokens(text: str) -> List[str]:
    return [t for t in normalize(text).split() if t not in _SW and len(t) > 1]
def sentence_fragments(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", str(text).strip())
    return [p.strip() for p in parts if len(p.strip()) > 10]
def dot_cosine(a: str, b: str) -> float:
    ba, bb = Counter(word_tokens(a)), Counter(word_tokens(b))
    if not ba or not bb:
        return 0.0
    dot  = sum(ba[k] * bb[k] for k in ba if k in bb)
    norm = math.sqrt(sum(v ** 2 for v in ba.values())) * \
           math.sqrt(sum(v ** 2 for v in bb.values()))
    return dot / (norm + 1e-9)
def vec_cosine(a, b, vec) -> float:
    mats = vec.transform([str(a), str(b)])
    return float(_sk_cos(mats[0], mats[1])[0][0])
def build_model_a_dataset(*_, **__):
    raise NotImplementedError("Provide preprocessing.py")
def apply_vectorizer(texts, vec):
    return vec.transform(texts)
def make_sample_string(article, question, option):
        return normalize(f"{article} {question} {option}")

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------

# Local paths (commented)
# _MODEL_DEST   = os.path.join(_THIS_DIR, "..", "models", "model_a", "traditional")
# _REPORTS_DEST = os.path.join(ARTIFACT_DIR, "reports")

# Kaggle paths (active)
_MODEL_DEST   = os.path.join(BASE_PROJ, "models", "model_a", "neural")
_REPORTS_DEST = os.path.join(ARTIFACT_DIR, "reports")
os.makedirs(_MODEL_DEST,   exist_ok=True)
os.makedirs(_REPORTS_DEST, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHOICES    = ["A", "B", "C", "D"]
_RNG_SEED   = 42

# ---------------------------------------------------------------------------
# SECTION 1 — Data loading
# ---------------------------------------------------------------------------

def _fetch_arrays():
    """Load OHE feature matrices, labels, and handcrafted features."""
    d = ARTIFACT_DIR
    files = [
        "X_train_ohe.npz", "X_val_ohe.npz", "X_test_ohe.npz",
        "y_train.npy", "y_val.npy", "y_test.npy",
        "hc_train.npy", "hc_val.npy", "hc_test.npy",
    ]
    loaders = [load_npz if f.endswith(".npz") else np.load for f in files]
    results = []
    for fname, loader in tqdm(zip(files, loaders), total=len(files), desc="loading arrays", unit="file"):
        results.append(loader(os.path.join(d, fname)))
    return tuple(results)


# keep old name available
load_processed_data = _fetch_arrays


def _integrity_check(X_tr, y_tr, X_va, y_va) -> dict:
    assert X_tr.shape[0] == len(y_tr), "Train X/y mismatch"
    assert X_va.shape[0] == len(y_va), "Val X/y mismatch"
    return {
        "train_rows":      X_tr.shape[0],
        "val_rows":        X_va.shape[0],
        "train_pos_frac":  float(np.mean(y_tr)),
        "val_pos_frac":    float(np.mean(y_va)),
    }


# ---------------------------------------------------------------------------
# SECTION 2 — Internal matrix helpers
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# SECTION 3 — Neural network classifier (replaces LR + SVM)
# ---------------------------------------------------------------------------

def _stack_hc(X, hc):
    """Stack handcrafted dense features onto sparse OHE matrix."""
    if hc is None or hc.shape[1] == 0:
        return X
    return sparse_hstack([X, hc.astype(np.float32)], format="csr")


def fit_answer_verifier(X_tr, y_tr, X_va=None, y_va=None, hc_tr=None, hc_va=None, **kw) -> AnswerVerifier:
    """
    Train a feed-forward neural network on OHE + handcrafted features.
    Loss is weighted by the inverse class frequency to handle
    the ≈3:1 wrong-to-correct ratio.  Early stopping on validation loss.
    """
    print("[A] NN", end="", flush=True)
    X_tr = _stack_hc(X_tr, hc_tr)
    if X_va is not None:
        X_va = _stack_hc(X_va, hc_va)
    input_dim = X_tr.shape[1]
    pos = y_tr.sum()
    neg = len(y_tr) - pos
    pos_weight = neg / max(pos, 1)          # ≈3.0 for the RACE dataset

    model = AnswerVerifier(
        input_dim=input_dim,
        hidden_dims=kw.get("hidden_dims", [512, 128]),
        dropout=kw.get("dropout", 0.3),
    )
    model = train_nn(
        model,
        X_tr, y_tr,
        X_val=X_va, y_val=y_va,
        epochs=kw.get("epochs", 30),
        batch_size=kw.get("batch_size", 256),
        lr=kw.get("lr", 1e-3),
        weight_decay=kw.get("weight_decay", 1e-4),
        pos_weight=pos_weight,
        patience=kw.get("patience", 5),
        verbose=True,
    )
    print(" ✓")
    return model

# compat alias
train_answer_verifier = fit_answer_verifier





# ---------------------------------------------------------------------------
# SECTION 7 — Cosine-similarity accuracy
# ---------------------------------------------------------------------------

def _sentence_max_cosine(article: str, option_text: str, vec) -> float:
    """Max TF-IDF cosine over all article sentences vs. option_text."""
    sents = sentence_fragments(str(article))
    if not sents:
        return vec_cosine(article, option_text, vec)
    s_vecs = vec.transform(sents)
    o_vec  = vec.transform([str(option_text)])
    return float(_sk_cos(s_vecs, o_vec).flatten().max())


def retrieval_accuracy(
    tfidf_vec,
    df: pd.DataFrame,
    n_rows: Optional[int] = None,
    ohe_vec=None,
    sentence_level: bool = True,
    alpha: float = 0.7,
) -> dict:
    """
    For each row select the option with the highest article–option similarity;
    report accuracy, average correct-option sim, average wrong-option sim, gap.
    """
    if n_rows and len(df) > n_rows:
        df = df.sample(n_rows, random_state=_RNG_SEED)

    n_hit, correct_sims, wrong_sims = 0, [], []

    for _, row in df.iterrows():
        article    = str(row["article"])
        gold       = str(row["answer"]).strip().upper()
        options    = {o: str(row[o]) for o in _CHOICES}
        per_option: Dict[str, float] = {}

        for key, txt in options.items():
            if tfidf_vec is not None:
                t_score = (
                    _sentence_max_cosine(article, txt, tfidf_vec)
                    if sentence_level
                    else vec_cosine(article, txt, tfidf_vec)
                )
            else:
                t_score = 0.0

            if ohe_vec is not None:
                o_score = (
                    _sentence_max_cosine(article, txt, ohe_vec)
                    if sentence_level
                    else vec_cosine(article, txt, ohe_vec)
                )
                per_option[key] = alpha * t_score + (1.0 - alpha) * o_score
            else:
                per_option[key] = t_score

        predicted = max(per_option, key=per_option.get)
        if predicted == gold:
            n_hit += 1
        correct_sims.append(per_option[gold])
        wrong_sims.extend(v for k, v in per_option.items() if k != gold)

    total = len(df)
    return {
        "accuracy":        n_hit / total if total else 0.0,
        "avg_correct_sim": float(np.mean(correct_sims)) if correct_sims else 0.0,
        "avg_wrong_sim":   float(np.mean(wrong_sims))   if wrong_sims   else 0.0,
        "sim_gap": float(np.mean(correct_sims) - np.mean(wrong_sims))
                   if (correct_sims and wrong_sims) else 0.0,
    }

# compat alias
cosine_similarity_accuracy = retrieval_accuracy


def sweep_retrieval_params(tfidf_vec, ohe_vec, eval_df: pd.DataFrame) -> dict:
    """Grid-search sentence_level × alpha; return best-accuracy configuration."""
    grid = [(sl, a) for sl in (True, False) for a in (0.55, 0.70, 0.85)]
    champion: dict = {"accuracy": -1.0}
    for sent_lvl, a in grid:
        result = retrieval_accuracy(
            tfidf_vec, eval_df,
            ohe_vec=ohe_vec,
            sentence_level=sent_lvl,
            alpha=a,
        )
        if result["accuracy"] > champion["accuracy"]:
            champion = {**result, "use_sentence_max": sent_lvl, "alpha": a}
    return champion

# compat alias
tune_cosine_similarity = sweep_retrieval_params


def domain_overlap(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    tfidf_vec,
    n_sample: int = 200,
) -> float:
    """
    Average max-cosine similarity from each test article to the nearest
    training article — a proxy for train/test domain overlap.
    """
    tr_corpus = train_df["article"].dropna().sample(
        min(n_sample, len(train_df)), random_state=_RNG_SEED).tolist()
    te_corpus = test_df["article"].dropna().sample(
        min(n_sample, len(test_df)),  random_state=_RNG_SEED).tolist()

    tr_vecs = tfidf_vec.transform(tr_corpus)
    te_vecs = tfidf_vec.transform(te_corpus)

    block, max_sims = 50, []
    for i in tqdm(range(0, len(te_corpus), block), desc="domain overlap", unit="chunk"):
        chunk   = te_vecs[i:i + block]
        sim_mat = _sk_cos(chunk, tr_vecs)
        max_sims.extend(sim_mat.max(axis=1).tolist())

    return float(np.mean(max_sims))

# compat alias
compute_train_test_domain_similarity = domain_overlap


# ---------------------------------------------------------------------------
# SECTION 8 — 4-way MCQ accuracy
# ---------------------------------------------------------------------------

def _build_hc_row(article, question, opt_text, tfidf_vec, idf_dict=None, vocab=None):
    """Compute the 12 handcrafted features for one (article, question, option) triple."""
    if tfidf_vec is not None and len(opt_text) > 0:
        art_vec = tfidf_vec.transform([article])
        q_vec = tfidf_vec.transform([question])
        opt_vec = tfidf_vec.transform([opt_text])
        sim_ao = float(_sk_cos(art_vec, opt_vec)[0, 0])
        sim_qo = float(_sk_cos(q_vec, opt_vec)[0, 0])
        opt_tfidf_max = float(opt_vec.max())
    else:
        sim_ao = sim_qo = 0.0
        opt_tfidf_max = 0.0

    art_tokens = set(article.split())
    opt_tokens = set(opt_text.split())
    q_tokens = set(question.split())
    overlap_ao = len(opt_tokens & art_tokens) / (len(opt_tokens) + 1e-9)
    overlap_qo = len(opt_tokens & q_tokens) / (len(opt_tokens) + 1e-9)
    opt_len_ratio = min(len(opt_text) / 200.0, 1.0)
    overlap_oa = len(opt_tokens & art_tokens) / (len(art_tokens) + 1e-9)
    overlap_oq = len(opt_tokens & q_tokens) / (len(q_tokens) + 1e-9)

    opt_idf_mean = 0.0
    if idf_dict and vocab and opt_tokens:
        idfs = [idf_dict.get(w, 0.0) for w in opt_tokens if w in vocab]
        opt_idf_mean = sum(idfs) / len(idfs) if idfs else 0.0

    has_digit = 1.0 if re.search(r"\d", opt_text) else 0.0

    art_bigrams = set(
        " ".join(article.split()[i:i+2])
        for i in range(len(article.split()) - 1)
    )
    opt_bigrams = set(
        " ".join(opt_text.split()[i:i+2])
        for i in range(len(opt_text.split()) - 1)
    )
    shared_bigrams = len(art_bigrams & opt_bigrams) / (len(opt_bigrams) + 1e-9) if opt_bigrams else 0.0

    starts_cap = 1.0 if opt_text and opt_text[0].isupper() else 0.0

    return [sim_ao, sim_qo, overlap_ao, overlap_qo, opt_len_ratio,
            overlap_oa, overlap_oq, opt_idf_mean, has_digit,
            shared_bigrams, opt_tfidf_max, starts_cap]


def mcq_accuracy(clf, ohe_vec, df: pd.DataFrame, n_rows: Optional[int] = None, tfidf_vec=None) -> float:
    """
    For each row score all four options; pick the one with highest P(correct).
    Returns fraction of rows where the predicted option matches the gold label.
    Uses handcrafted features when tfidf_vec is provided.
    """
    if n_rows and len(df) > n_rows:
        df = df.sample(n_rows, random_state=_RNG_SEED)
    hits = total = 0
    use_hc = tfidf_vec is not None
    idf_dict = dict(zip(tfidf_vec.get_feature_names_out(), tfidf_vec.idf_)) if use_hc else None
    vocab = set(tfidf_vec.get_feature_names_out()) if use_hc else None
    for _, row in df.iterrows():
        article = clean_text(row["article"])
        question = clean_text(row["question"])
        encoded = []
        hc_rows = []
        for o in _CHOICES:
            opt_text = clean_text(str(row[o]))
            encoded.append(make_sample_string(article, question, opt_text))
            if use_hc:
                hc_rows.append(
                    _build_hc_row(article, question, opt_text, tfidf_vec, idf_dict, vocab)
                )
        X = ohe_vec.transform(encoded)
        if use_hc and hc_rows:
            hc = np.array(hc_rows, dtype=np.float32)
            X = _stack_hc(X, hc)
        probs = clf.predict_proba(X)[:, 1]
        best = _CHOICES[int(np.argmax(probs))]
        if best == str(row["answer"]).strip().upper():
            hits += 1
        total += 1
    return hits / total if total else 0.0

# compat alias
compute_4way_accuracy = mcq_accuracy


# ---------------------------------------------------------------------------
# SECTION 9 — Binary evaluation
# ---------------------------------------------------------------------------

def score_classifier(clf, X, y, tag: str = "") -> dict:
    preds = clf.predict(X)
    acc   = accuracy_score(y, preds)
    prec  = precision_score(y, preds, average="macro", zero_division=0)
    rec   = recall_score(y,    preds, average="macro", zero_division=0)
    f1    = f1_score(y,        preds, average="macro", zero_division=0)
    em    = float(np.mean([str(p) == str(t) for p, t in zip(preds, y)]))
    if tag:
        print(f"\n  {tag}")
        for name, val in [("acc", acc), ("prec", prec), ("rec", rec), ("f1", f1), ("em", em)]:
            print(f"    {name}={val:.4f}")
    return {"accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "exact_match": em, "predictions": preds}

# compat alias
evaluate_binary = score_classifier


# ---------------------------------------------------------------------------
# SECTION 10 — MCQ generation
# ---------------------------------------------------------------------------

_WH_BANK: Dict[str, List[str]] = {
    "what": [
        "What does the passage say about {topic}?",
        "What is {topic} according to the passage?",
        "What role does {topic} play in the passage?",
    ],
    "how": [
        "How is {topic} described in the passage?",
        "How does {topic} relate to the main idea?",
    ],
    "why": [
        "Why is {topic} important according to the passage?",
        "Why is {topic} mentioned in the passage?",
    ],
    "where": ["Where does {topic} occur according to the passage?"],
    "when":  ["When is {topic} relevant in the context of the passage?"],
    "who":   ["Who is associated with {topic} in the passage?"],
}

_ALL_TEMPLATES: List[str] = list(chain.from_iterable(_WH_BANK.values()))

_GEN_STOP = frozenset({
    "a","an","the","is","it","in","of","to","and","or","for","on","with","as",
    "at","by","be","was","are","were","this","that","from","but","not","have",
    "has","had","he","she","they","we","you","i","do","did","will","its",
    "their","which","who","what","how","when","where","there","these","those",
    "can","could","would","should","also","been","being",
})


def _content_words(text: str) -> List[str]:
    return [t for t in word_tokens(text) if t not in _GEN_STOP and len(t) >= 4]


def _sent_score(sentence: str, answer_vocab: set) -> float:
    s_vocab = set(_content_words(sentence))
    return len(answer_vocab & s_vocab) + len(sentence.split()) / 100.0


def _drop_short_sents(sents: List[str], min_len: int = 10) -> List[str]:
    return [s for s in sents if len(s) >= min_len]


def extract_candidate_sentences(
    article: str, anchor: str, top_k: int = 5
) -> List[Tuple[str, float]]:
    """Score each article sentence by keyword overlap with anchor text."""
    raw   = sentence_fragments(article)
    sents = _drop_short_sents(raw)
    if not sents:
        return [(article[:200], 1.0)]
    a_vocab = set(_content_words(anchor))
    ranked  = [(s, _sent_score(s, a_vocab)) for s in sents]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def _pick_topic_word(sentence: str, used: set) -> str:
    pool = [t for t in _content_words(sentence) if t not in used]
    if not pool:
        pool = _content_words(sentence)
    return max(pool, key=len) if pool else "this topic"


def _extract_answer_span(sentence: str, question: str, max_words: int = 8) -> str:
    words  = str(sentence).split()
    if not words:
        return sentence[:40]
    q_set  = set(normalize(question).split())
    anchor = next((i for i, w in enumerate(words) if normalize(w) in q_set), None)
    if anchor is None:
        cands = [
            (sum(1 for w in words[s:s + max_words] if w.lower() not in _GEN_STOP), s)
            for s in range(max(0, len(words) - max_words + 1))
        ]
        _, best_start = max(cands) if cands else (0, 0)
        return " ".join(words[best_start:best_start + max_words])
    start = max(0, anchor - max_words // 2)
    end   = min(len(words), start + max_words)
    start = max(0, end - max_words)
    return " ".join(words[start:end])


# -- Question ranker features --

def _ranker_features(question: str, source_sent: str, article: str) -> np.ndarray:
    q_tok = word_tokens(question)
    s_tok = word_tokens(source_sent)
    a_tok = word_tokens(article)
    q_len = len(q_tok)
    return np.array([
        len(set(q_tok) & set(s_tok)) / (len(set(q_tok)) + 1e-9),
        len(set(q_tok) & set(a_tok)) / (len(set(q_tok)) + 1e-9),
        q_len,
        float(question.split()[0].lower() in {"what","who","where","when","why","how"})
            if question else 0.0,
        float(question.strip().endswith("?")),
        float(any(t.endswith(("ed","ing","es","tion")) for t in q_tok)),
        len(set(q_tok)) / (q_len + 1e-9),
        len(_content_words(question)) / (q_len + 1e-9),
    ], dtype=np.float32)


def _heuristic_rank(pairs: List[Tuple[str, str]], article: str) -> List[Tuple[str, str]]:
    art_vocab = set(word_tokens(article))
    def score(q, _s):
        return len(set(word_tokens(q)) & art_vocab) / (len(word_tokens(q)) + 1e-9)
    return sorted(pairs, key=lambda p: score(*p), reverse=True)


# -- Distractor assembly --

def _build_distractors(article: str, correct: str, n: int = 3) -> List[str]:
    """
    Build plausible distractors in priority order:
      1. Article sentence fragments with moderate article-answer similarity
      2. Keyword-substituted answer variants
    """
    sents       = sentence_fragments(article)
    ans_vocab   = set(word_tokens(correct))
    art_content = _content_words(article)

    candidates: List[Tuple[float, str]] = []
    for sent in sents:
        phrase = _extract_answer_span(sent, correct, max_words=6)
        if set(word_tokens(phrase)) == ans_vocab or len(phrase) <= 5:
            continue
        sim = dot_cosine(phrase, correct)
        candidates.append((sim, phrase))

    candidates.sort(reverse=True)
    chosen = [ph for sim, ph in candidates if 0.08 < sim < 0.85][:n]

    swap_pool  = list(set(art_content) - ans_vocab)
    ans_words  = correct.split()
    if len(swap_pool) >= 2 and len(ans_words) >= 2:
        for _ in range(n + 1):
            variant = ans_words[:]
            variant[random.randrange(len(variant))] = random.choice(swap_pool)
            chosen.append(" ".join(variant))

    seen, unique = set(), []
    for d in chosen:
        key = normalize(d)
        if key not in seen and key != normalize(correct):
            seen.add(key)
            unique.append(d)
        if len(unique) >= n:
            break

    while len(unique) < n:
        unique.append(f"none of the above ({len(unique) + 1})")

    return unique[:n]

# compat alias
generate_distractors = _build_distractors


def compose_questions(
    article: str,
    count: int = 5,
    ranker=None,
) -> List[dict]:
    """
    Three-phase MCQ generation:
      Phase 1 — extract candidate sentences by keyword-overlap scoring
      Phase 2 — apply Wh-word templates over topic keywords
      Phase 3 — rank with ML ranker (if available) or heuristic fallback

    Each returned dict contains:
      question, answer, correct_letter, distractors, source_sentence, options
    """
    sents = sentence_fragments(article)
    if not sents:
        sents = [article[:200]]

    richest  = max(sents, key=lambda s: len(_content_words(s)))
    a_vocab  = set(_content_words(richest))

    candidates = extract_candidate_sentences(
        article, richest, top_k=max(count * 2, 10)
    )

    templates = _ALL_TEMPLATES[:]
    random.shuffle(templates)

    raw_pairs: List[Tuple[str, str]] = []
    for i, (sent, _) in enumerate(candidates):
        topic    = _pick_topic_word(sent, a_vocab)
        template = templates[i % len(templates)]
        raw_pairs.append((template.format(topic=topic), sent))

    if ranker is not None:
        feats = np.array([_ranker_features(q, s, article) for q, s in raw_pairs])
        try:
            scores = ranker.predict_proba(feats)[:, 1]
            ranked = [p for _, p in sorted(zip(scores, raw_pairs), reverse=True)]
        except Exception:
            ranked = _heuristic_rank(raw_pairs, article)
    else:
        ranked = _heuristic_rank(raw_pairs, article)

    results, seen_keys = [], set()
    for question, src_sent in ranked:
        answer  = _extract_answer_span(src_sent, question, max_words=8)
        key     = normalize(answer)[:20]
        if key in seen_keys:
            continue
        seen_keys.add(key)

        distractors = _build_distractors(article, answer, n=3)
        opts_list   = [answer] + distractors
        random.shuffle(opts_list)
        opts_dict   = dict(zip(_CHOICES, opts_list))
        correct_ltr = next(k for k, v in opts_dict.items() if v == answer)

        results.append({
            "question":        question,
            "answer":          answer,
            "correct_letter":  correct_ltr,
            "distractors":     distractors,
            "source_sentence": src_sent,
            "options":         opts_dict,
        })
        if len(results) >= count:
            break

    if not results:
        q = "What is the main idea of the passage?"
        a = _extract_answer_span(richest, q, max_words=8)
        d = _build_distractors(article, a, n=3)
        o = dict(zip(_CHOICES, [a] + d))
        results.append({
            "question": q, "answer": a, "correct_letter": "A",
            "distractors": d, "source_sentence": richest, "options": o,
        })

    return results

# compat alias
generate_questions_from_passage = compose_questions


# ---------------------------------------------------------------------------
# SECTION 11 — Generation metrics (BLEU / ROUGE / METEOR)
# ---------------------------------------------------------------------------

def _lex_tokens(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _ngram_freq(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """
    Sentence-level BLEU with modified n-gram precision + brevity penalty.
    Near-zero precisions are smoothed with 1e-9 to avoid -inf log.
    """
    ref = _lex_tokens(reference)
    hyp = _lex_tokens(hypothesis)
    if not ref or not hyp:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        h_ng = _ngram_freq(hyp, n)
        r_ng = _ngram_freq(ref, n)
        if not h_ng:
            precisions.append(0.0)
            continue
        clip = sum(min(cnt, r_ng[ng]) for ng, cnt in h_ng.items())
        precisions.append(clip / sum(h_ng.values()))
    bp       = 1.0 if len(hyp) >= len(ref) else math.exp(1 - len(ref) / len(hyp))
    smoothed = [p if p > 0 else 1e-9 for p in precisions]
    return bp * math.exp(sum(math.log(p) for p in smoothed) / max_n)

# compat alias
sentence_bleu_score = bleu


def rouge(reference: str, hypothesis: str) -> dict:
    """ROUGE-1, ROUGE-2, and ROUGE-L F1 scores."""
    ref = _lex_tokens(reference)
    hyp = _lex_tokens(hypothesis)

    def _f1_ngram(r, h, n):
        rng = _ngram_freq(r, n)
        hng = _ngram_freq(h, n)
        if not rng or not hng:
            return 0.0
        shared = sum(min(rng[k], hng[k]) for k in rng if k in hng)
        prec   = shared / sum(hng.values())
        rec    = shared / sum(rng.values())
        return 2 * prec * rec / (prec + rec + 1e-9)

    def _lcs_len(a, b):
        m, n = len(a), len(b)
        dp   = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = (
                    dp[i-1][j-1] + 1
                    if a[i-1] == b[j-1]
                    else max(dp[i-1][j], dp[i][j-1])
                )
        return dp[m][n]

    r1  = _f1_ngram(ref, hyp, 1)
    r2  = _f1_ngram(ref, hyp, 2)
    lcs = _lcs_len(ref, hyp)
    if ref and hyp:
        p_l = lcs / len(hyp)
        r_l = lcs / len(ref)
        rl  = 2 * p_l * r_l / (p_l + r_l + 1e-9)
    else:
        rl  = 0.0
    return {"rouge1_f": r1, "rouge2_f": r2, "rougeL_f": rl}

# compat alias
rouge_scores = rouge


def meteor(reference: str, hypothesis: str) -> float:
    """Simplified METEOR: precision/recall harmonic mean with fragmentation penalty."""
    ref = _lex_tokens(reference)
    hyp = _lex_tokens(hypothesis)
    if not ref or not hyp:
        return 0.0
    rc  = Counter(ref)
    hc  = Counter(hyp)
    m   = sum(min(rc[t], hc[t]) for t in rc if t in hc)
    if m == 0:
        return 0.0
    prec   = m / len(hyp)
    rec    = m / len(ref)
    fmean  = (10 * prec * rec) / (9 * prec + rec + 1e-9)
    return fmean * (1 - 0.5 / max(m, 1))

# compat alias
meteor_score = meteor


def generation_metrics(
    generated: List[dict],
    df: pd.DataFrame,
    n_sample: int = 300,
) -> dict:
    """Compare generated answers to source sentences; return aggregate scores."""
    if not generated:
        return {}
    if len(df) > n_sample:
        df = df.sample(n_sample, random_state=_RNG_SEED)

    b_vals, r1_vals, r2_vals, rl_vals, m_vals = [], [], [], [], []
    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="generation metrics", unit="row")):
        sample  = generated[i % len(generated)]
        ref     = sample["source_sentence"]
        hyp     = sample["answer"]
        b_vals.append(bleu(ref, hyp))
        rg = rouge(ref, hyp)
        r1_vals.append(rg["rouge1_f"])
        r2_vals.append(rg["rouge2_f"])
        rl_vals.append(rg["rougeL_f"])
        m_vals.append(meteor(ref, hyp))

    return {
        "bleu":      float(np.mean(b_vals)),
        "rouge1_f":  float(np.mean(r1_vals)),
        "rouge2_f":  float(np.mean(r2_vals)),
        "rougeL_f":  float(np.mean(rl_vals)),
        "meteor":    float(np.mean(m_vals)),
        "n_samples": len(b_vals),
    }

# compat alias
compute_generation_metrics = generation_metrics


# ---------------------------------------------------------------------------
# SECTION 12 — Persistence
# ---------------------------------------------------------------------------

def persist_models(nn_model, metrics: dict):
    """Write all Model-A artifacts to _MODEL_DEST."""
    save_checkpoint(nn_model, os.path.join(_MODEL_DEST, "answer_verifier.pt"))
    joblib.dump(metrics, os.path.join(_MODEL_DEST, "metrics.pkl"))
    print(f"\n  artifacts → {_MODEL_DEST}")
    _export_metrics_to_csv(metrics)


def _export_metrics_to_csv(metrics: dict):
    """Export all metrics to CSV files in the reports directory."""
    import os
    os.makedirs(_REPORTS_DEST, exist_ok=True)

    # 1. NN binary classification metrics
    if "nn" in metrics and metrics["nn"]:
        nn_data = [{"model": "NN"}]
        nn_data[0].update(
            {k: v for k, v in metrics["nn"].items() if k != "predictions"}
        )
        df_nn = pd.DataFrame(nn_data)
        csv_path = os.path.join(_REPORTS_DEST, "model_a_binary_metrics.csv")
        df_nn.to_csv(csv_path, index=False)
        print(f"    → {csv_path}")

    # 2. Cosine retrieval metrics
    if "cosine_retrieval" in metrics and metrics["cosine_retrieval"]:
        df_cosine = pd.DataFrame([metrics["cosine_retrieval"]])
        csv_path = os.path.join(_REPORTS_DEST, "model_a_cosine_retrieval_metrics.csv")
        df_cosine.to_csv(csv_path, index=False)
        print(f"    → {csv_path}")

    # 3. Text generation metrics
    if "text_generation" in metrics and metrics["text_generation"]:
        df_gen = pd.DataFrame([metrics["text_generation"]])
        csv_path = os.path.join(_REPORTS_DEST, "model_a_text_generation_metrics.csv")
        df_gen.to_csv(csv_path, index=False)
        print(f"    → {csv_path}")

# compat alias
save_models = persist_models


def _load_artifact(fname: str):
    path = os.path.join(_MODEL_DEST, fname)
    return joblib.load(path) if os.path.exists(path) else None


# ---------------------------------------------------------------------------
# SECTION 13 — Main training pipeline
# ---------------------------------------------------------------------------

def _train_transformer_av(train_csv, val_csv):
    """Optional transformer answer verifier training if transformers is installed."""
    try:
        from src.transformer_train import train_answer_verifier_transformer
    except ImportError:
        print("  transformers not installed — skipping")
        return
    if not os.path.exists(train_csv):
        print(f"  {train_csv} not found — skipping transformer AV")
        return
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv) if os.path.exists(val_csv) else None
    try:
        train_answer_verifier_transformer(train_df, val_df, epochs=5, use_lora=True)
    except Exception as e:
        print(f"  transformer AV failed: {e}")


def train_all():
    """
    End-to-end Model-A pipeline (neural network + transformer):
      1  Load OHE feature arrays
      2  Train NN answer verifier
      3  Optional: K-Means / Label Propagation analysis
      4  Evaluate on validation: binary metrics, 4-way MCQ, cosine accuracy
      5  Generate MCQs and score BLEU / ROUGE / METEOR
      6  Train transformer answer verifier (optional)
      7  Final evaluation on held-out test split
      8  Persist all artifacts
    """
    BAR = "─" * 58
    print(BAR)
    print("  verifier_a  ::  neural network training run")
    print(BAR)

    # -- 1. Load arrays --
    print("\n(1) loading feature arrays")
    X_tr, X_va, X_te, y_tr, y_va, y_te, hc_tr, hc_va, hc_te = _fetch_arrays()
    info = _integrity_check(X_tr, y_tr, X_va, y_va)
    print(f"    train={X_tr.shape[0]:,}  val={X_va.shape[0]:,}  test={X_te.shape[0]:,}")
    print(f"    features={X_tr.shape[1]:,}")
    print(f"    pos-rate — train={info['train_pos_frac']:.3f}  val={info['val_pos_frac']:.3f}")

    # -- 2. Train neural network answer verifier --
    print("\n(2) training neural network answer verifier")
    nn_model = fit_answer_verifier(X_tr, y_tr, X_va, y_va, hc_tr=hc_tr, hc_va=hc_va, epochs=30)

    # -- 3. Validation snapshot --
    print("\n(3) validation snapshot")
    X_va_stacked = _stack_hc(X_va, hc_va)
    nn_res = score_classifier(nn_model, X_va_stacked, y_va, "NN (val)")

    # Data / vectorizer paths
    ohe_path   = os.path.join(ARTIFACT_DIR, "ohe_vectorizer.pkl")
    tfidf_path = os.path.join(ARTIFACT_DIR, "tfidf_vectorizer.pkl")
    val_csv    = os.path.join(ARTIFACT_DIR, "val_clean.csv")
    train_csv  = os.path.join(ARTIFACT_DIR, "train_clean.csv")
    test_csv   = os.path.join(ARTIFACT_DIR, "test_clean.csv")

    cos_metrics, nn_4w = {}, 0.0
    ohe_vec = tfidf_vec = None

    if all(os.path.exists(p) for p in [ohe_path, tfidf_path, val_csv]):
        with open(ohe_path,   "rb") as fh: ohe_vec   = pickle.load(fh)
        with open(tfidf_path, "rb") as fh: tfidf_vec = pickle.load(fh)

        val_df  = pd.read_csv(val_csv)
        eval_df = val_df.sample(min(1_000, len(val_df)), random_state=_RNG_SEED)

        print("\n  4-way MCQ accuracy (val):")
        nn_4w = mcq_accuracy(nn_model, ohe_vec, eval_df, tfidf_vec=tfidf_vec)
        print(f"    NN : {nn_4w:.4f}")

        print("\n  cosine-similarity retrieval accuracy (val):")
        best = sweep_retrieval_params(tfidf_vec, ohe_vec, eval_df)
        print(f"    sent_level={best['use_sentence_max']}  alpha={best['alpha']}")
        print(f"    acc={best['accuracy']:.4f}  "
              f"avg_correct={best['avg_correct_sim']:.4f}  "
              f"avg_wrong={best['avg_wrong_sim']:.4f}  "
              f"gap={best['sim_gap']:.4f}")
        cos_metrics = dict(best)

        if os.path.exists(train_csv) and os.path.exists(test_csv):
            tr_df = pd.read_csv(train_csv)
            te_df = pd.read_csv(test_csv)
            dom   = domain_overlap(tr_df, te_df, tfidf_vec, n_sample=400)
            print(f"\n  domain overlap (train↔test): {dom:.4f}")
            cos_metrics["domain_similarity"] = dom

    # -- 4. Question generation + text metrics --
    print("\n(4) MCQ generation checks")
    gen_m: dict = {}
    if os.path.exists(val_csv):
        gen_df  = pd.read_csv(val_csv)
        sub_df  = gen_df.sample(min(400, len(gen_df)), random_state=_RNG_SEED)
        all_gen = []
        for _, row in tqdm(sub_df.iterrows(), total=len(sub_df), desc="generating MCQs", unit="article"):
            art = str(row.get("article", ""))
            if len(art) >= 50:
                all_gen.extend(compose_questions(art, count=3))
        if all_gen:
            gen_m = generation_metrics(all_gen, sub_df)
            print(f"    generated {len(all_gen)} question-answer pairs")
            for k in ("bleu", "rouge1_f", "rouge2_f", "rougeL_f", "meteor"):
                print(f"    {k.upper():<12}: {gen_m.get(k, 0):.4f}")
            report = os.path.join(_REPORTS_DEST, "generation_metrics.pkl")
            joblib.dump(gen_m, report)
            print(f"    metrics → {report}")
        else:
            print("    (no usable articles — skipping generation eval)")
    else:
        print("    (val_clean.csv not found — skipping generation eval)")

    # -- 5. Transformer answer verifier (optional) --
    print("\n(5) transformer answer verifier")
    _train_transformer_av(train_csv, val_csv)

    # -- 6. Test evaluation --
    print("\n(6) held-out test split")
    X_te_stacked = _stack_hc(X_te, hc_te)
    score_classifier(nn_model, X_te_stacked, y_te, "NN (test)")
    preds = nn_model.predict(X_te_stacked)
    te_a  = accuracy_score(y_te, preds)
    te_f  = f1_score(y_te, preds, average="macro", zero_division=0)
    te_e  = float(np.mean([str(p) == str(t) for p, t in zip(preds, y_te)]))
    print(f"\n  NN (test)\n    acc={te_a:.4f}  f1={te_f:.4f}  em={te_e:.4f}")

    # -- Aggregate metrics dict --
    metrics = {
        "nn":               {**(nn_res or {}), "4way_acc": nn_4w},
        "cosine_retrieval":  cos_metrics,
        "text_generation":   gen_m,
    }

    persist_models(nn_model, metrics)
    print("\n" + "=" * 65)
    print("  verifier_a training complete")
    print(BAR)
    return nn_model, metrics


# ---------------------------------------------------------------------------
# Public inference API
# ---------------------------------------------------------------------------

def load_model_a() -> dict:
    """Load the NN answer verifier and metrics from disk."""
    nn_path = os.path.join(_MODEL_DEST, "answer_verifier.pt")
    if not os.path.exists(nn_path):
        print(f"  WARNING: NN checkpoint not found at {nn_path}")
        return {"model": None, "metrics": None}
    input_dim = (  # try to infer from saved meta
        _load_checkpoint_meta(nn_path, "input_dim") or 10_000
    )
    model = AnswerVerifier(input_dim=input_dim)
    ckpt = torch.load(nn_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    metrics = _load_artifact("metrics.pkl")
    return {"model": model, "metrics": metrics}


def _load_checkpoint_meta(path: str, key: str):
    """Read a single meta key from a saved NN checkpoint."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        return ckpt.get("meta", {}).get(key)
    except Exception:
        return None


def verify_answer(
    article: str,
    question: str,
    option: str,
    models: dict,
    ohe_vec,
    tfidf_vec=None,
) -> dict:
    """
    Predict whether `option` is the correct answer using the neural network
    with handcrafted features. Returns prediction and probabilities.
    """
    idf_dict = dict(zip(tfidf_vec.get_feature_names_out(), tfidf_vec.idf_)) if tfidf_vec is not None else None
    vocab = set(tfidf_vec.get_feature_names_out()) if tfidf_vec is not None else None
    article_clean = clean_text(article)
    question_clean = clean_text(question)
    opt_clean = clean_text(option)

    sample = ohe_vec.transform([make_sample_string(article_clean, question_clean, opt_clean)])
    if tfidf_vec is not None:
        hc = np.array([_build_hc_row(article_clean, question_clean, opt_clean, tfidf_vec, idf_dict, vocab)], dtype=np.float32)
        sample = _stack_hc(sample, hc)

    nn_prob = float(models["model"].predict_proba(sample)[0, 1])

    return {
        "prediction":     int(nn_prob > 0.5),
        "probability":    nn_prob,
        "lr_proba":       nn_prob,
        "svm_proba":      nn_prob,
        "soft_ensemble":  nn_prob,
        "stack_ensemble": nn_prob,
        "nn_proba":       nn_prob,
    }


def generate_mcq(article: str, n_questions: int = 5) -> List[dict]:
    """Public API: generate n_questions MCQs from a passage."""
    return compose_questions(str(article), count=n_questions)


# ---------------------------------------------------------------------------
# Optional: transformer-based model loading (only if transformers is installed)
# ---------------------------------------------------------------------------

def _try_import_transformers():
    try:
        import transformers as _tf
        import peft as _peft
        return True
    except ImportError:
        return False


def load_transformer_answer_verifier():
    """Load a trained TransformerAnswerVerifier checkpoint + tokenizer."""
    if not _try_import_transformers():
        print("  transformers/peft not installed. Skipping transformer model.")
        return None, None

    from transformers import AutoTokenizer
    from src.nn_models import TransformerAnswerVerifier

    meta_path = os.path.join(_MODEL_DEST.replace("neural", "transformer"), "transformer_meta.json")
    ckpt_path = os.path.join(_MODEL_DEST.replace("neural", "transformer"), "answer_verifier_transformer.pt")

    if not os.path.exists(ckpt_path):
        return None, None

    with open(meta_path) as f:
        import json
        meta = json.load(f)

    model_name = meta.get("model_name", "bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = TransformerAnswerVerifier(model_name=model_name, num_labels=2)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()
    print(f"  loaded transformer answer verifier ({model_name})")
    return model, tokenizer


def verify_answer_transformer(
    article: str,
    question: str,
    option: str,
    model,
    tokenizer,
) -> dict:
    """Predict P(correct) using the transformer answer verifier."""
    text = f"article: {article}\nquestion: {question}\noption: {option}"
    enc = tokenizer(text, truncation=True, padding="max_length", max_length=384, return_tensors="pt")
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        probs = torch.softmax(out.logits, dim=-1)
        prob_correct = float(probs[0, 1])
    return {
        "prediction": int(prob_correct > 0.5),
        "probability": prob_correct,
        "nn_proba": prob_correct,
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_all()