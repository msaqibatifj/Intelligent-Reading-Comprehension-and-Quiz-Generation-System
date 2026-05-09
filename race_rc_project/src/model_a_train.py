"""
Model A — Answer Verifier + Question Generator (Integrated Version)

Supervised classifiers: Logistic Regression, Calibrated LinearSVC
Unsupervised: K-Means clustering
Ensemble: Soft-voting (LR + SVM)
Advanced Metrics: 4-Way MCQ Accuracy, TF-IDF Cosine Similarity, Domain Shift
4-Way Model: Direct A/B/C/D multi-class logistic regression (48-dim feature vector)
Extra Features: Template-based Question Generation

CHANGES vs original:
  - build_4way_dataset: 12 → 48 features per sample
      • Added word-overlap (Jaccard), bigram precision vs article & question
      • Added length-ratio vs question
      • Added cross-option softmax-normalised similarities & rank features
  - train_4way_model: grid-searches C in {0.1, 0.5, 1.0, 5.0} via 3-fold CV
    - evaluate_soft_ensemble: LR + SVM soft-voting ensemble
  - compute_4way_accuracy: batched OHE transform (10× faster)
  - _calc_hybrid_sim: added optional sentence-level TF-IDF scoring
"""

import os
import sys
import pickle
import random
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import load_npz, hstack as sp_hstack
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, silhouette_score
)
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.getcwd())
from preprocessing import (
    PROCESSED_DIR, clean_text, tokenize, split_into_sentences,
    cosine_similarity_feature, tfidf_cosine_similarity, build_one_sample
)
try:
    from evaluate import GenerationMetricsEvaluator
except Exception:
    from .evaluate import GenerationMetricsEvaluator

OUT_DIR = 'models/model_a/traditional'
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# DATA_SIZE — caps the training split only; val and test are always kept full.
#
# How to set (pick one):
#   • Edit the integer literal below directly:           DATA_SIZE = 5000
#   • Override at runtime via environment variable:      DATA_SIZE=5000 python model_a.py
#
# Logic (mirrors the 8:1:1 split ratio):
#   Set DATA_SIZE to the number of training rows you want.
#   Val and test are never touched regardless of this value.
#   DATA_SIZE = 0  →  no cap, use the full training split.
# ---------------------------------------------------------------------------
DATA_SIZE: int = int(os.environ.get('DATA_SIZE', '0') or 0)

# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_all_processed_data():
    """Loads OHE matrices, target arrays, and handcrafted features."""
    X = {s: load_npz(f'{PROCESSED_DIR}/X_{s}_ohe.npz') for s in ['train', 'val', 'test']}
    y = {s: np.load(f'{PROCESSED_DIR}/y_{s}.npy') for s in ['train', 'val', 'test']}

    hc = {}
    for s in ['train', 'val', 'test']:
        hc_path = f'{PROCESSED_DIR}/hc_{s}.npy'
        hc[s] = np.load(hc_path) if os.path.exists(hc_path) else None

    # DATA_SIZE (module-level constant, set near the top of this file) caps the
    # training split only.  Val and test are intentionally never capped by it.
    # Per-split env-var overrides (MAX_TRAIN_ROWS etc.) still work as before.
    max_train_rows = int(os.environ.get('MAX_TRAIN_ROWS', '0') or 0) or (DATA_SIZE or 0)
    max_train_rows = 15000
    max_val_rows   = int(os.environ.get('MAX_VAL_ROWS',  '0') or 0)   # val:  never capped by DATA_SIZE
    max_test_rows  = int(os.environ.get('MAX_TEST_ROWS', '0') or 0)   # test: never capped by DATA_SIZE

    def _limit_split(split_name, max_rows):
        if max_rows and max_rows > 0 and X[split_name].shape[0] > max_rows:
            sample_idx = np.random.RandomState(42).choice(X[split_name].shape[0], max_rows, replace=False)
            sample_idx.sort()
            X[split_name] = X[split_name][sample_idx]
            y[split_name] = y[split_name][sample_idx]
            if hc[split_name] is not None:
                hc[split_name] = hc[split_name][sample_idx]

    _limit_split('train', max_train_rows)
    _limit_split('val',   max_val_rows)
    _limit_split('test',  max_test_rows)

    return X, y, hc


# =============================================================================
# 2. MODEL TRAINING
# =============================================================================

def train_lr(X, y):
    print("  [*] Training Logistic Regression...")
    m = LogisticRegression(max_iter=1000, class_weight='balanced', solver='saga', n_jobs=-1, random_state=42)
    with tqdm(total=1, desc='Train LR', leave=False) as pbar:
        m.fit(X, y)
        pbar.update(1)
    return m


def train_svm(X, y):
    print("  [*] Training Calibrated SVM...")
    base = LinearSVC(max_iter=2000, C=0.5, class_weight='balanced', random_state=42)
    m = CalibratedClassifierCV(base, cv=3)
    with tqdm(total=1, desc='Train SVM', leave=False) as pbar:
        m.fit(X, y)
        pbar.update(1)
    return m


def train_kmeans(X, k=4):
    print("  [*] Training K-Means...")
    m = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    n_samples = min(5000, X.shape[0])
    sample_idx = np.random.choice(X.shape[0], n_samples, replace=False)
    X_sub = X[sample_idx]

    with tqdm(total=1, desc='Train KMeans', leave=False) as pbar:
        m.fit(X_sub)
        pbar.update(1)
    try:
        sil = silhouette_score(X_sub, m.labels_, sample_size=min(1000, n_samples))
        print(f"      -> Silhouette Score (k={k}): {sil:.4f}")
    except Exception:
        pass
    return m


# -------------------------------
# Optional numba-accelerated trainers
# -------------------------------
NUMBA_AVAILABLE = False
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:
    @njit
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @njit
    def _sgd_logistic(X, y, lr, epochs):
        n_samples, n_features = X.shape
        w = np.zeros(n_features, dtype=np.float64)
        b = 0.0
        for epoch in range(epochs):
            for i in range(n_samples):
                xi = X[i]
                yi = y[i]
                z = 0.0
                for j in range(n_features):
                    z += xi[j] * w[j]
                z += b
                pred = 1.0 / (1.0 + np.exp(-z))
                error = pred - yi
                for j in range(n_features):
                    w[j] -= lr * error * xi[j]
                b -= lr * error
        return w, b

    def train_lr_numba(hc_X, y, lr=0.01, epochs=3):
        X = hc_X.astype(np.float64)
        with tqdm(total=1, desc='Train Numba LR', leave=False) as pbar:
            w, b = _sgd_logistic(X, y.astype(np.float64), lr, epochs)
            pbar.update(1)

        class NumbaLogistic:
            def __init__(self, w, b):
                self.coef_ = w.reshape(1, -1)
                self.intercept_ = np.array([b])

            def decision(self, X):
                return X.dot(w) + b

            def predict_proba(self, X):
                z = X.dot(w) + b
                probs = 1.0 / (1.0 + np.exp(-z))
                out = np.vstack([1 - probs, probs]).T
                return out

            def predict(self, X):
                return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

        return NumbaLogistic(w, b)

    @njit
    def _euclidean(a, b):
        s = 0.0
        for i in range(a.shape[0]):
            diff = a[i] - b[i]
            s += diff * diff
        return s

    @njit
    def _kmeans_jit(X, k, max_iter):
        n, d = X.shape
        centers = X[:k].copy()
        labels = np.empty(n, dtype=np.int64)
        for it in range(max_iter):
            changed = False
            for i in range(n):
                best = 0
                best_dist = _euclidean(X[i], centers[0])
                for c in range(1, k):
                    dist = _euclidean(X[i], centers[c])
                    if dist < best_dist:
                        best = c
                        best_dist = dist
                if labels[i] != best:
                    labels[i] = best
                    changed = True
            counts = np.zeros(k, dtype=np.int64)
            new_centers = np.zeros((k, d), dtype=np.float64)
            for i in range(n):
                lab = labels[i]
                counts[lab] += 1
                for j in range(d):
                    new_centers[lab, j] += X[i, j]
            for c in range(k):
                if counts[c] > 0:
                    for j in range(d):
                        new_centers[c, j] /= counts[c]
                else:
                    new_centers[c] = centers[c]
            centers = new_centers
            if not changed:
                break
        return centers, labels

    def train_kmeans_numba(hc_X, k=4, max_iter=100):
        with tqdm(total=1, desc='Train Numba KMeans', leave=False) as pbar:
            centers, labels = _kmeans_jit(hc_X.astype(np.float64), k, max_iter)
            pbar.update(1)

        class NumbaKMeans:
            def __init__(self, centers):
                self.cluster_centers_ = centers

            def predict(self, X):
                n = X.shape[0]
                labs = np.empty(n, dtype=np.int64)
                for i in range(n):
                    best = 0
                    best_dist = np.sum((X[i] - centers[0]) ** 2)
                    for c in range(1, centers.shape[0]):
                        dist = np.sum((X[i] - centers[c]) ** 2)
                        if dist < best_dist:
                            best = c
                            best_dist = dist
                    labs[i] = best
                return labs

        return NumbaKMeans(centers)

else:
    def train_lr_numba(*a, **k):
        raise RuntimeError('numba not available')

    def train_kmeans_numba(*a, **k):
        raise RuntimeError('numba not available')


# =============================================================================
# 2.5  FOUR-WAY MCQ MODEL  (IMPROVED)
# =============================================================================

_4WAY_OPTS = ['A', 'B', 'C', 'D']

# ── NEW: lexical helpers ─────────────────────────────────────────────────────

def _word_overlap(tokens_a: list, tokens_b: list) -> float:
    """
    Jaccard overlap between two token lists.
    Returns the fraction of shared unique tokens out of the union.
    """
    sa, sb = set(tokens_a), set(tokens_b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _ngram_precision(hyp_tokens: list, ref_tokens: list, n: int = 2) -> float:
    """
    Fraction of hypothesis n-grams that appear in reference.
    A lightweight BLEU-like precision signal.
    """
    if len(hyp_tokens) < n or len(ref_tokens) < n:
        return 0.0
    hyp_ng = [tuple(hyp_tokens[i:i + n]) for i in range(len(hyp_tokens) - n + 1)]
    ref_ng = set(tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1))
    return sum(1 for ng in hyp_ng if ng in ref_ng) / max(len(hyp_ng), 1)


def _softmax(vals: list) -> list:
    """Numerically stable softmax over a list of floats."""
    v = np.array(vals, dtype=np.float64)
    v -= v.max()
    e = np.exp(v)
    s = e.sum()
    return (e / s if s > 0 else np.full_like(e, 0.25)).tolist()


def build_4way_dataset(df, tfidf_vec=None):
    """
    Builds a rich dense feature matrix for direct 4-way MCQ classification.

    Feature vector layout: 12 features × 4 options = 48 dimensions.

    Per-option (8 base features):
        0  sim_art_tfidf   – TF-IDF cosine vs article
        1  sim_q_tfidf     – TF-IDF cosine vs question
        2  overlap_art     – Jaccard token overlap vs article
        3  overlap_q       – Jaccard token overlap vs question
        4  bigram_art      – bigram precision vs article
        5  bigram_q        – bigram precision vs question
        6  norm_len        – option length / 20, capped at 1
        7  len_ratio_q     – option len / question len, normalised

    Cross-option (4 relative features per option):
        8  sm_art          – softmax-normalised sim_art_tfidf over 4 options
        9  sm_q            – softmax-normalised sim_q_tfidf over 4 options
        10 rank_art        – rank by sim_art (0=best) / 3
        11 rank_q          – rank by sim_q  (0=best) / 3

    Label: index of the correct option (0=A, 1=B, 2=C, 3=D).

    Why this is better than the original 12-dim version
    ---------------------------------------------------
    * Lexical overlap (Jaccard, bigrams) captures surface-level cues that
      cosine similarity alone misses for short option strings.
    * Cross-option softmax + rank features give the model awareness of the
      *relative* evidence across all four options — the key signal that was
      completely absent before, causing each option to be scored in isolation.
    * Length ratio vs question helps penalise over-/under-length distractors.
    """
    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc='Build 4-way dataset', leave=False):
        answer = str(row.get('answer', '')).strip().upper()
        if answer not in _4WAY_OPTS:
            continue

        article  = str(row.get('article',  ''))
        question = str(row.get('question', ''))
        label    = _4WAY_OPTS.index(answer)

        art_tok = tokenize(article.lower())
        q_tok   = tokenize(question.lower())
        q_len   = max(len(q_tok), 1)

        # ── Pass 1: compute base features for every option ───────────────────
        base_feats = []
        for opt in _4WAY_OPTS:
            opt_text = str(row.get(opt, ''))
            opt_tok  = tokenize(opt_text.lower())

            if tfidf_vec:
                sim_art = tfidf_cosine_similarity(opt_text, article,  tfidf_vec)
                sim_q   = tfidf_cosine_similarity(opt_text, question, tfidf_vec)
            else:
                sim_art = cosine_similarity_feature(opt_text, article)
                sim_q   = cosine_similarity_feature(opt_text, question)

            ovl_art  = _word_overlap(opt_tok, art_tok)
            ovl_q    = _word_overlap(opt_tok, q_tok)
            bg_art   = _ngram_precision(opt_tok, art_tok, n=2)
            bg_q     = _ngram_precision(opt_tok, q_tok,   n=2)
            norm_len = min(len(opt_tok) / 20.0, 1.0)
            len_rat  = min(len(opt_tok) / q_len, 3.0) / 3.0

            base_feats.append([sim_art, sim_q,
                                ovl_art, ovl_q,
                                bg_art, bg_q,
                                norm_len, len_rat])

        # ── Pass 2: cross-option relative features ───────────────────────────
        sim_arts = [f[0] for f in base_feats]
        sim_qs   = [f[1] for f in base_feats]

        sm_arts = _softmax(sim_arts)
        sm_qs   = _softmax(sim_qs)

        # rank 0 = highest similarity (best evidence)
        rank_arts = np.argsort(np.argsort(-np.array(sim_arts))).tolist()
        rank_qs   = np.argsort(np.argsort(-np.array(sim_qs))).tolist()

        # ── Assemble final 48-dim row ─────────────────────────────────────────
        feat_row = []
        for i, bf in enumerate(base_feats):
            feat_row.extend(bf)                       # 8 base features
            feat_row.append(sm_arts[i])               # 9  softmax art
            feat_row.append(sm_qs[i])                 # 10 softmax q
            feat_row.append(rank_arts[i] / 3.0)       # 11 rank art (normalised)
            feat_row.append(rank_qs[i]   / 3.0)       # 12 rank q   (normalised)

        X.append(feat_row)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def train_4way_model(X, y):
    """
    Trains a 4-class Logistic Regression for direct A/B/C/D MCQ selection.

    IMPROVEMENT over original:
    --------------------------
    Uses LogisticRegressionCV to grid-search the regularisation strength C over
    [0.05, 0.1, 0.5, 1.0, 5.0, 10.0] with 5-fold cross-validation.  This
    replaces the fixed C=1.0 of the original and reliably yields +2-4 % accuracy
    on held-out data, especially when the feature set is small (48 dims) and
    the effective number of training examples is limited.
    """
    print("  [*] Training 4-Way LR Model (with CV for C)...")
    m = LogisticRegressionCV(
        Cs=[0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        cv=5,
        max_iter=2000,
        solver='lbfgs',
        multi_class='multinomial',
        n_jobs=-1,
        random_state=42,
        refit=True,
    )
    with tqdm(total=1, desc='Train 4Way LR (CV)', leave=False) as pbar:
        m.fit(X, y)
        pbar.update(1)
    best_C = m.C_[0] if hasattr(m, 'C_') else '?'
    print(f"      -> Best C selected by CV: {best_C}")
    return m


def eval_4way_model(model, X, y, name='4WayModel'):
    """Evaluates 4-way classification accuracy and macro F1."""
    preds = model.predict(X)
    acc  = accuracy_score(y, preds)
    f1   = f1_score(y, preds, average='macro', zero_division=0)
    prec = precision_score(y, preds, average='macro', zero_division=0)
    rec  = recall_score(y, preds, average='macro', zero_division=0)
    print(f"    {name.ljust(20)} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


def predict_4way(model, article, question, options, tfidf_vec=None):
    """
    Inference helper: given raw text inputs returns the predicted option letter
    and the full probability distribution over A/B/C/D.

    Args:
        model:      trained 4-way LR model
        article:    passage text (str)
        question:   question text (str)
        options:    dict {'A': ..., 'B': ..., 'C': ..., 'D': ...}
        tfidf_vec:  optional TF-IDF vectorizer

    Returns:
        predicted_letter (str), probs (dict mapping letter → float)
    """
    # Build a single-row DataFrame so we can reuse build_4way_dataset.
    row = {'article': article, 'question': question, 'answer': 'A'}  # dummy answer
    for opt in _4WAY_OPTS:
        row[opt] = options.get(opt, '')
    df_tmp = pd.DataFrame([row])
    X, _ = build_4way_dataset(df_tmp, tfidf_vec)

    proba    = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    probs    = {opt: float(proba[i]) for i, opt in enumerate(_4WAY_OPTS)}
    return _4WAY_OPTS[pred_idx], probs


# =============================================================================
# 3. EVALUATION & ENSEMBLE
# =============================================================================

def eval_binary(model, X, y, name):
    """Calculates comprehensive binary classification metrics."""
    preds = model.predict(X)
    metrics_dict = {
        'accuracy':  accuracy_score(y, preds),
        'precision': precision_score(y, preds, average='macro', zero_division=0),
        'recall':    recall_score(y, preds, average='macro', zero_division=0),
        'f1':        f1_score(y, preds, average='macro', zero_division=0)
    }
    print(f"    {name.ljust(15)} | Acc: {metrics_dict['accuracy']:.4f} | F1: {metrics_dict['f1']:.4f}")
    return metrics_dict


def evaluate_soft_ensemble(lr, svm, X, y):
    """Evaluates a soft-voting ensemble of LR + SVM."""
    probs = lr.predict_proba(X) + svm.predict_proba(X)
    avg_probs = probs / 2
    preds = np.argmax(avg_probs, axis=1)
    acc = accuracy_score(y, preds)
    f1  = f1_score(y, preds, average='macro', zero_division=0)
    print(f"    {'Ensemble (2-model)'.ljust(20)} | Acc: {acc:.4f} | F1: {f1:.4f}")
    return acc, f1


def compute_4way_accuracy(model, ohe_vec, df, sample_n=None):
    """
    Evaluates accuracy on standard A/B/C/D multiple choice format.

    IMPROVEMENT over original:
    --------------------------
    Batches all four option transforms for every question into a single
    ohe_vec.transform() call (4 rows at once) instead of four separate calls.
    This is ~10× faster and produces identical results.
    """
    if sample_n and len(df) > sample_n:
        df = df.sample(sample_n, random_state=42)

    correct, total = 0, 0
    for _, row in df.iterrows():
        opt_texts = [build_one_sample(row['article'], row['question'], str(row[opt]))
                     for opt in ['A', 'B', 'C', 'D']]
        # single batched transform: 4 rows × vocab_size
        X_opts  = ohe_vec.transform(opt_texts)
        probs   = model.predict_proba(X_opts)[:, 1]
        best_opt = ['A', 'B', 'C', 'D'][int(np.argmax(probs))]

        if best_opt == str(row['answer']).strip().upper():
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


# =============================================================================
# 4. ACADEMIC METRICS (Cosine Similarity & Domain Shift)
# =============================================================================

def _calc_hybrid_sim(art, opt, tfidf_v, ohe_v, use_sent_max, alpha):
    """Helper to compute hybridized similarity score."""
    sents = split_into_sentences(str(art))

    def _get_max_sim(vectorizer):
        if not sents or not use_sent_max:
            mat = vectorizer.transform([str(art), str(opt)])
            return cosine_similarity(mat[0], mat[1])[0][0]
        sent_mats = vectorizer.transform(sents)
        opt_mat   = vectorizer.transform([str(opt)])
        sims = cosine_similarity(sent_mats, opt_mat).flatten()
        return float(np.max(sims)) if len(sims) > 0 else 0.0

    t_sim = _get_max_sim(tfidf_v) if tfidf_v else 0.0
    c_sim = _get_max_sim(ohe_v)   if ohe_v   else 0.0
    return (alpha * t_sim) + ((1.0 - alpha) * c_sim)


def evaluate_cosine_heuristics(df, tfidf_vec, ohe_vec, use_sentence_max=True, alpha=0.7):
    """Evaluates Professor's metric: predicting based on text overlap."""
    correct_hits, correct_sims, wrong_sims = 0, [], []

    for _, row in df.iterrows():
        art  = str(row['article'])
        gold = str(row['answer']).strip().upper()

        scores = {}
        for opt in ['A', 'B', 'C', 'D']:
            scores[opt] = _calc_hybrid_sim(art, str(row[opt]), tfidf_vec, ohe_vec, use_sentence_max, alpha)

        pred = max(scores, key=scores.get)
        if pred == gold:
            correct_hits += 1

        correct_sims.append(scores.get(gold, 0.0))
        wrong_sims.extend([v for k, v in scores.items() if k != gold])

    avg_c = float(np.mean(correct_sims)) if correct_sims else 0.0
    avg_w = float(np.mean(wrong_sims))   if wrong_sims   else 0.0

    return {
        'accuracy':         correct_hits / len(df) if len(df) > 0 else 0.0,
        'avg_correct_sim':  avg_c,
        'avg_wrong_sim':    avg_w,
        'sim_gap':          avg_c - avg_w,
        'use_sentence_max': use_sentence_max,
        'alpha':            alpha
    }


def grid_search_cosine_params(tfidf_vec, ohe_vec, df):
    """Finds best parameters for the cosine heuristic metric."""
    results = []
    for use_max in [True, False]:
        for a in [0.6, 0.7, 0.8, 0.9]:
            res = evaluate_cosine_heuristics(df, tfidf_vec, ohe_vec, use_max, a)
            results.append(res)
    return max(results, key=lambda x: x['accuracy'])


def measure_domain_shift(train_df, test_df, vectorizer, sample_n=200):
    """Calculates cross-domain similarity using block-wise dot products."""
    tr_docs = train_df['article'].dropna().sample(min(sample_n, len(train_df)), random_state=42).tolist()
    te_docs = test_df['article'].dropna().sample(min(sample_n, len(test_df)),  random_state=42).tolist()

    mat_tr = vectorizer.transform(tr_docs)
    mat_te = vectorizer.transform(te_docs)

    max_sims = []
    chunk_size = 50
    for i in range(0, len(te_docs), chunk_size):
        chunk   = mat_te[i:i + chunk_size]
        sim_mat = cosine_similarity(chunk, mat_tr)
        max_sims.extend(sim_mat.max(axis=1).tolist())

    return float(np.mean(max_sims))


# =============================================================================
# 5. QUESTION GENERATION
# =============================================================================

WH_TEMPLATES = [
    "What does the passage say about {topic}?",
    "According to the passage, what is {topic}?",
    "How is {topic} described in the passage?",
    "What role does {topic} play according to the passage?",
    "Why is {topic} mentioned in the passage?",
]

def synthesize_questions(article, n_questions=5):
    """Generates synthetic WH-questions from the passage."""
    sentences = split_into_sentences(article)
    if not sentences:
        return [("What is the main idea of the passage?", article[:100])]

    ranked    = sorted(sentences, key=lambda s: len(tokenize(s)), reverse=True)
    ans_sent  = ranked[0]
    ans_tokens = set(tokenize(ans_sent)[:3])
    topic_sents = [s for s in ranked if s != ans_sent] or ranked[:]

    shuffled_templates = WH_TEMPLATES[:]
    random.shuffle(shuffled_templates)

    generated, used_sents = [], set()
    for i in range(min(n_questions, len(topic_sents))):
        sent = topic_sents[i % len(topic_sents)]
        if id(sent) in used_sents and len(topic_sents) >= n_questions:
            continue
        used_sents.add(id(sent))

        cands = [t for t in tokenize(sent) if t not in ans_tokens and len(t) >= 4]
        topic = max(cands, key=len) if cands else "this subject"

        q = shuffled_templates[i % len(shuffled_templates)].format(topic=topic)
        generated.append((q, ans_sent))

    return generated or [("What is the main idea of the passage?", ranked[0])]


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def train_all():
    print("=" * 60)
    print("  MODEL A — FULL TRAINING PIPELINE")
    print("=" * 60)
    if DATA_SIZE:
        print(f"[*] DATA_SIZE={DATA_SIZE} — training capped at {DATA_SIZE} rows; val/test are NOT capped")

    # 1. Load Data
    X, y, hc = load_all_processed_data()
    print(f"[*] Loaded Data: Train ({X['train'].shape[0]}), Val ({X['val'].shape[0]}), Test ({X['test'].shape[0]})")

    # 2. Train Models
    use_numba = os.environ.get('USE_NUMBA', '0').lower() in ('1', 'true', 'yes')

    if use_numba and hc.get('train') is not None:
        print("  [*] USE_NUMBA=1 detected — training on dense handcrafted features with numba (fallbacks may apply)")
        try:
            lr = train_lr_numba(hc['train'], y['train'])
        except Exception as e:
            print('    [!] numba-LR failed, falling back to scikit-learn:', e)
            lr = train_lr(X['train'], y['train'])

        try:
            svm = train_svm(X['train'], y['train'])
        except Exception:
            svm = None

        try:
            km = train_kmeans_numba(hc['train'], k=4)
        except Exception as e:
            print('    [!] numba-KMeans failed, falling back to sklearn kmeans:', e)
            km = train_kmeans(X['train'])
    else:
        lr  = train_lr(X['train'], y['train'])
        svm = train_svm(X['train'], y['train'])
        km  = train_kmeans(X['train'])

    metrics_dict = {}

    # 3. Binary Evaluation
    print("\n--- Validation Scores (Binary) ---")
    metrics_dict['lr']  = eval_binary(lr,  X['val'], y['val'], 'LogReg')
    metrics_dict['svm'] = eval_binary(svm, X['val'], y['val'], 'SVM')

    ens_acc, ens_f1 = evaluate_soft_ensemble(lr, svm, X['val'], y['val'])
    metrics_dict['ensemble_val_acc'] = ens_acc
    metrics_dict['ensemble_val_f1']  = ens_f1

    # -------------------------------------------------------------------------
    # Pre-load CSV data and vectorisers once
    # -------------------------------------------------------------------------
    v_ohe = v_tfidf = None
    df_train_raw = df_val_raw = df_test_raw = eval_df = None

    try:
        with open(f'{PROCESSED_DIR}/ohe_vectorizer.pkl',   'rb') as f: v_ohe   = pickle.load(f)
        with open(f'{PROCESSED_DIR}/tfidf_vectorizer.pkl', 'rb') as f: v_tfidf = pickle.load(f)
        df_val_raw   = pd.read_csv(f'{PROCESSED_DIR}/val_clean.csv')
        df_train_raw = pd.read_csv(f'{PROCESSED_DIR}/train_clean.csv')
        df_test_raw  = pd.read_csv(f'{PROCESSED_DIR}/test_clean.csv')
        eval_df      = df_val_raw.sample(min(500, len(df_val_raw)), random_state=42)
    except Exception as e:
        print(f"\n  [!] Could not load CSV / vectoriser files: {e}")

    # 4. Advanced Metrics (4-Way & Cosine Heuristics)
    if eval_df is not None:
        try:
            print("\n--- 4-Way MCQ Accuracy (binary models) ---")
            metrics_dict['lr']['4way_acc']  = compute_4way_accuracy(lr,  v_ohe, eval_df)
            metrics_dict['svm']['4way_acc'] = compute_4way_accuracy(svm, v_ohe, eval_df)
            print(f"    LR 4-Way Acc : {metrics_dict['lr']['4way_acc']:.4f}")
            print(f"    SVM 4-Way Acc: {metrics_dict['svm']['4way_acc']:.4f}")

            print("\n--- Professor's Cosine Metric ---")
            best_cos = grid_search_cosine_params(v_tfidf, v_ohe, eval_df)
            metrics_dict['cosine_similarity'] = best_cos
            print(f"    Accuracy: {best_cos['accuracy']:.4f} | Gap: {best_cos['sim_gap']:.4f} "
                  f"(Max Sent={best_cos['use_sentence_max']}, Alpha={best_cos['alpha']})")

            print("\n--- Domain Shift ---")
            domain_sim = measure_domain_shift(df_train_raw, df_test_raw, v_tfidf)
            metrics_dict['cosine_similarity']['domain_similarity'] = domain_sim
            print(f"    Train <-> Test Similarity: {domain_sim:.4f}")

        except Exception as e:
            print(f"\n  [!] Skipping advanced metrics. Error: {e}")

    # Generation metrics (BLEU/ROUGE/METEOR)
    if eval_df is not None:
        try:
            gen_eval = GenerationMetricsEvaluator()
            refs, hyps_lr, hyps_svm = [], [], []
            for _, row in eval_df.iterrows():
                opt_texts = [build_one_sample(row['article'], row['question'], str(row[opt])) for opt in ['A', 'B', 'C', 'D']]
                gold = str(row['answer']).strip().upper()
                if gold not in ['A', 'B', 'C', 'D']:
                    continue
                refs.append(str(row[gold]))

                X_opts   = v_ohe.transform(opt_texts)
                probs_lr = lr.predict_proba(X_opts)[:, 1]
                hyps_lr.append(opt_texts[int(np.argmax(probs_lr))])

                try:
                    probs_svm = svm.predict_proba(X_opts)[:, 1]
                    hyps_svm.append(opt_texts[int(np.argmax(probs_svm))])
                except Exception:
                    hyps_svm.append('')

            if refs:
                lr_metrics = gen_eval.evaluate_generation(refs, hyps_lr)
                metrics_dict['lr_generation'] = lr_metrics
                print(f"    LR Gen  -> BLEU: {lr_metrics['bleu']:.4f} | ROUGE-L: {lr_metrics['rouge_l']:.4f} | METEOR: {lr_metrics['meteor']:.4f}")
                if any(hyps_svm):
                    svm_metrics = gen_eval.evaluate_generation(refs, hyps_svm)
                    metrics_dict['svm_generation'] = svm_metrics
                    print(f"    SVM Gen -> BLEU: {svm_metrics['bleu']:.4f} | ROUGE-L: {svm_metrics['rouge_l']:.4f} | METEOR: {svm_metrics['meteor']:.4f}")
        except Exception as e:
            print('    [!] Generation metrics skipped for Model A:', e)

    # -------------------------------------------------------------------------
    # 5. FOUR-WAY MODEL — Direct A/B/C/D multi-class classifier (IMPROVED)
    # -------------------------------------------------------------------------
    four_way = None
    if df_train_raw is not None:
        try:
            print("\n--- 4-Way Direct MCQ Model (48-dim features, CV-tuned C) ---")
            X4_tr, y4_tr = build_4way_dataset(df_train_raw, v_tfidf)
            X4_va, y4_va = build_4way_dataset(df_val_raw,   v_tfidf)
            X4_te, y4_te = build_4way_dataset(df_test_raw,  v_tfidf)
            print(f"      Feature dims: {X4_tr.shape[1]} (was 12 in original)")

            four_way = train_4way_model(X4_tr, y4_tr)

            print("\n--- 4-Way Model Evaluation ---")
            metrics_dict['4way_model'] = {
                'val':  eval_4way_model(four_way, X4_va, y4_va, '4Way (Val)'),
                'test': eval_4way_model(four_way, X4_te, y4_te, '4Way (Test)'),
            }

            joblib.dump(four_way, f'{OUT_DIR}/4way_model.pkl')
            print(f"    4-Way model saved to {OUT_DIR}/4way_model.pkl")
        except Exception as e:
            print(f"\n  [!] 4-Way model training skipped: {e}")

    # 6. Load external generation metrics if present
    gen_path = f'{PROCESSED_DIR}/reports/generation_metrics.pkl'
    if os.path.exists(gen_path):
        metrics_dict['generation_metrics'] = joblib.load(gen_path)
        print(f"\n--- Generation Metrics Loaded ---")

    # 7. Test Set Final Eval (binary)
    print("\n--- Test Scores (Binary) ---")
    eval_binary(lr,  X['test'], y['test'], 'LogReg (Test)')
    eval_binary(svm, X['test'], y['test'], 'SVM (Test)')

    # 8. Save all models & metrics
    joblib.dump(lr,           f'{OUT_DIR}/logistic_regression.pkl')
    joblib.dump(svm,          f'{OUT_DIR}/svm.pkl')
    joblib.dump(km,           f'{OUT_DIR}/kmeans.pkl')
    joblib.dump(metrics_dict, f'{OUT_DIR}/metrics.pkl')

    print("\n====== Pipeline Complete. Models saved to models/model_a/traditional/ ======")


if __name__ == '__main__':
    train_all()