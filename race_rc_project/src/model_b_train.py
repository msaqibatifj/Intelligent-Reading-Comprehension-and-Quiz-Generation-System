"""
Model B — Distractor Ranker & Hint Generator (Integrated Version)

Pipeline:
  1. Candidate Extraction (Frequency-based filtering)
  2. Feature Engineering (TF-IDF Cosine / Jaccard, Character Overlap, Position, Length)
  3. Distractor Ranker (Logistic Regression with prefix-diversity filtering)
  4. Hint Scorer (Logistic Regression outputting 3-tier graduated hints)
"""

import os
import sys
import joblib
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

sys.path.insert(0, os.getcwd())
# from preprocessing import (
#     PROCESSED_DIR, tokenize, clean_text, split_into_sentences,
#     cosine_similarity_feature, tfidf_cosine_similarity
# )

OUT_DIR = os.path.join(os.getcwd(), 'models', 'model_b', 'traditional')
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# DATA_SIZE — caps the training split only; val and test are always kept full.
#
# How to set (pick one):
#   • Edit the integer literal below directly:           DATA_SIZE = 5000
#   • Override at runtime via environment variable:      DATA_SIZE=5000 python model_b.py
#
# Logic (mirrors the 8:1:1 split ratio):
#   Set DATA_SIZE to the number of training rows you want.
#   Val and test are never touched regardless of this value.
#   DATA_SIZE = 0  →  no cap, use the full training split.
# ---------------------------------------------------------------------------
DATA_SIZE: int = int(os.environ.get('DATA_SIZE', '0') or 0)


# Optional numba acceleration for dense logistic training
NUMBA_AVAILABLE = False
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:
    @njit
    def _sgd_logistic_numba(X, y, lr, epochs):
        n_samples, n_features = X.shape
        w = np.zeros(n_features, dtype=np.float64)
        b = 0.0
        for ep in range(epochs):
            for i in range(n_samples):
                z = 0.0
                for j in range(n_features):
                    z += X[i, j] * w[j]
                z += b
                pred = 1.0 / (1.0 + np.exp(-z))
                err = pred - y[i]
                for j in range(n_features):
                    w[j] -= lr * err * X[i, j]
                b -= lr * err
        return w, b

    def train_lr_numba(X, y, lr=0.01, epochs=5):
        Xd = X.astype(np.float64)
        yd = y.astype(np.float64)
        w, b = _sgd_logistic_numba(Xd, yd, lr, epochs)

        class NumbaLR:
            def __init__(self, w, b):
                self.coef_ = w.reshape(1, -1)
                self.intercept_ = np.array([b])

            def predict_proba(self, X):
                z = X.dot(w) + b
                probs = 1.0 / (1.0 + np.exp(-z))
                return np.vstack([1 - probs, probs]).T

            def predict(self, X):
                return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

        return NumbaLR(w, b)

else:
    def train_lr_numba(*a, **k):
        raise RuntimeError('numba not available')


# =============================================================================
# 1. DISTRACTOR LOGIC (Extraction & Features)
# =============================================================================

def get_candidates(article, answer, top_n=40):
    """Extract frequent article words that are not part of the correct answer."""
    ans_tokens = set(tokenize(answer))
    valid_toks = [t for t in tokenize(article) if t not in ans_tokens and len(t) > 3]
    return [w for w, _ in Counter(valid_toks).most_common(top_n)]


def get_candidates_cached(article_tokens, answer_tokens, top_n=40):
    """Cached version of candidate extraction for repeated dataset passes."""
    valid_toks = [t for t in article_tokens if t not in answer_tokens and len(t) > 3]
    return [w for w, _ in Counter(valid_toks).most_common(top_n)]


def compute_distractor_vector(cand, answer, article, tfidf_vec=None):
    """
    Computes 5 features for distractor quality:
    [Sim to Answer, Sim to Article, Char Overlap Ratio, Norm Freq, Norm Length]
    """
    if tfidf_vec:
        sim_ans = tfidf_cosine_similarity(cand, answer, tfidf_vec)
        sim_art = tfidf_cosine_similarity(cand, article, tfidf_vec)
    else:
        sim_ans = cosine_similarity_feature(cand, answer)
        sim_art = cosine_similarity_feature(cand, article)

    ans_chars = set(answer)
    char_ratio = len(set(cand) & ans_chars) / (len(ans_chars) + 1e-9)

    total_art_len = max(len(tokenize(article)), 1)
    norm_freq = article.lower().count(cand.lower()) / total_art_len
    norm_len = min(len(cand) / 20.0, 1.0)

    return [sim_ans, sim_art, char_ratio, norm_freq, norm_len]


def compute_distractor_vector_cached(cand, answer, article_tokens, article_text, tfidf_vec=None):
    """Cached version of distractor features to avoid repeated tokenization."""
    if tfidf_vec:
        sim_ans = tfidf_cosine_similarity(cand, answer, tfidf_vec)
        sim_art = tfidf_cosine_similarity(cand, article_text, tfidf_vec)
    else:
        sim_ans = cosine_similarity_feature(cand, answer)
        sim_art = cosine_similarity_feature(cand, article_text)

    ans_chars = set(answer)
    char_ratio = len(set(cand) & ans_chars) / (len(ans_chars) + 1e-9)

    total_art_len = max(len(article_tokens), 1)
    norm_freq = article_text.lower().count(cand.lower()) / total_art_len
    norm_len = min(len(cand) / 20.0, 1.0)

    return [sim_ans, sim_art, char_ratio, norm_freq, norm_len]


def build_dist_dataset(df, tfidf_vec=None):
    """Labels real RACE wrong options as 1, and random article words as 0."""
    X, y = [], []
    for _, r in df.iterrows():
        article = str(r['article'])
        if 'answer' not in r or str(r['answer']) not in r:
            continue

        correct = str(r[str(r['answer'])])
        wrongs = [str(r[opt]) for opt in ['A', 'B', 'C', 'D'] if opt != r['answer']]

        for w in wrongs:
            X.append(compute_distractor_vector(w, correct, article, tfidf_vec))
            y.append(1)

        wrong_prefixes = {tokenize(w)[0][:4] if tokenize(w) else '' for w in wrongs}
        neg_added = 0
        for cand in get_candidates(article, correct, top_n=20):
            if neg_added >= 3:
                break
            if cand[:4] not in wrong_prefixes:
                X.append(compute_distractor_vector(cand, correct, article, tfidf_vec))
                y.append(0)
                neg_added += 1

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def build_dist_dataset_cached(row_cache, tfidf_vec=None):
    """Build distractor data from pre-tokenized cached rows."""
    X, y = [], []
    for row in tqdm(row_cache, desc='Build distractor data', leave=False):
        article = row['article']
        article_tokens = row['article_tokens']
        answer_label = row['answer']
        if answer_label not in ['A', 'B', 'C', 'D']:
            continue

        correct = row['options'][answer_label]
        wrongs = [row['options'][opt] for opt in ['A', 'B', 'C', 'D'] if opt != answer_label]
        answer_tokens = row['answer_tokens']

        for w in wrongs:
            X.append(compute_distractor_vector_cached(w, correct, article_tokens, article, tfidf_vec))
            y.append(1)

        wrong_prefixes = {tok[0][:4] if tok else '' for tok in row['wrong_tokens']}
        neg_added = 0
        for cand in get_candidates_cached(article_tokens, answer_tokens, top_n=20):
            if neg_added >= 3:
                break
            if cand[:4] not in wrong_prefixes:
                X.append(compute_distractor_vector_cached(cand, correct, article_tokens, article, tfidf_vec))
                y.append(0)
                neg_added += 1

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# =============================================================================
# 2. HINT LOGIC (Features & Dataset)
# =============================================================================

def compute_hint_vector(sent, question, pos, total_sents, tfidf_vec=None):
    """
    Computes 4 features for hint quality:
    [Token Overlap, Norm Position, Norm Length, TF-IDF/Jaccard Sim]
    """
    q_toks, s_toks = set(tokenize(question)), set(tokenize(sent))
    overlap = len(q_toks & s_toks) / (len(q_toks) + 1e-9)
    norm_pos = pos / max(total_sents - 1, 1)
    norm_len = min(len(s_toks) / 50.0, 1.0)
    sim = (tfidf_cosine_similarity(question, sent, tfidf_vec)
           if tfidf_vec else cosine_similarity_feature(question, sent))
    return [overlap, norm_pos, norm_len, sim]


def compute_hint_vector_cached(sent, question_tokens, question_text, pos, total_sents, tfidf_vec=None):
    """Cached version of hint features using pre-tokenized question text."""
    s_toks = set(tokenize(sent))
    overlap = len(question_tokens & s_toks) / (len(question_tokens) + 1e-9)
    norm_pos = pos / max(total_sents - 1, 1)
    norm_len = min(len(s_toks) / 50.0, 1.0)
    sim = (tfidf_cosine_similarity(question_text, sent, tfidf_vec)
           if tfidf_vec else cosine_similarity_feature(question_text, sent))
    return [overlap, norm_pos, norm_len, sim]


def build_hint_dataset(df, tfidf_vec=None):
    """Labels the most question-similar sentence as 1, the rest as 0."""
    X, y = [], []
    for _, r in df.iterrows():
        article, question = str(r['article']), str(r['question'])
        sents = split_into_sentences(article)
        if not sents:
            continue

        sims = ([tfidf_cosine_similarity(question, s, tfidf_vec) for s in sents]
                if tfidf_vec else [cosine_similarity_feature(question, s) for s in sents])
        best_idx = int(np.argmax(sims))

        for i, s in enumerate(sents):
            X.append(compute_hint_vector(s, question, i, len(sents), tfidf_vec))
            y.append(1 if i == best_idx else 0)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def build_hint_dataset_cached(row_cache, tfidf_vec=None):
    """Build hint data from pre-tokenized cached rows."""
    X, y = [], []
    for row in tqdm(row_cache, desc='Build hint data', leave=False):
        sents = row['sentences']
        if not sents:
            continue

        question = row['question']
        question_tokens = row['question_tokens']
        sims = ([tfidf_cosine_similarity(question, s, tfidf_vec) for s in sents]
                if tfidf_vec else [cosine_similarity_feature(question, s) for s in sents])
        best_idx = int(np.argmax(sims))
        total_sents = len(sents)

        for i, s in enumerate(sents):
            X.append(compute_hint_vector_cached(s, question_tokens, question, i, total_sents, tfidf_vec))
            y.append(1 if i == best_idx else 0)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def build_row_cache(df):
    """Precompute repeated token/sentence work for the full dataset build."""
    cached_rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc='Cache rows', leave=False):
        article      = str(r.get('article', ''))
        question     = str(r.get('question', ''))
        answer_label = str(r.get('answer', '')).strip().upper()
        options      = {opt: str(r.get(opt, '')) for opt in ['A', 'B', 'C', 'D']}
        article_tokens  = tokenize(article)
        question_tokens = set(tokenize(question))
        sentences    = split_into_sentences(article)
        answer_text  = options.get(answer_label, '') if answer_label in options else ''
        wrong_tokens = [tokenize(options[opt]) for opt in ['A', 'B', 'C', 'D'] if opt != answer_label]

        cached_rows.append({
            'article':         article,
            'question':        question,
            'answer':          answer_label,
            'options':         options,
            'article_tokens':  article_tokens,
            'question_tokens': question_tokens,
            'sentences':       sentences,
            'answer_text':     answer_text,
            'answer_tokens':   set(tokenize(answer_text)),
            'wrong_tokens':    wrong_tokens,
        })

    return cached_rows


# =============================================================================
# 3. GENERATORS (Inference)
# =============================================================================

def generate_distractors(ranker, article, answer, tfidf_vec=None, n=3):
    """Scores candidates and enforces diversity (no matching 4-char prefixes)."""
    cands = get_candidates(article, answer, top_n=50)
    if not cands:
        return ['Option A', 'Option B', 'Option C']

    scored = sorted(
        [(ranker.predict_proba([compute_distractor_vector(c, answer, article, tfidf_vec)])[0][1], c)
         for c in cands],
        reverse=True,
    )

    chosen, seen_prefixes = [], set()
    ans_tokens = set(tokenize(answer))
    for _, word in scored:
        prefix = word[:4].lower()
        if prefix in seen_prefixes or word.lower() in ans_tokens:
            continue
        chosen.append(word)
        seen_prefixes.add(prefix)
        if len(chosen) >= n:
            break

    fallbacks = ['an alternative', 'none of the above', 'another option']
    while len(chosen) < n:
        chosen.append(fallbacks[len(chosen) % len(fallbacks)])
    return chosen[:n]


def generate_graduated_hints(scorer, article, question, tfidf_vec=None):
    """Extracts top sentences and formats them into a 3-tier hint structure."""
    sents = split_into_sentences(article)
    if not sents:
        return ["Look closely.", "Re-read the text.", "Check the main body."]

    scored = sorted(
        [(scorer.predict_proba([compute_hint_vector(s, question, i, len(sents), tfidf_vec)])[0][1], s)
         for i, s in enumerate(sents)],
        reverse=True,
    )

    tops  = [s for _, s in scored[:3]]
    hints = []
    if len(tops) > 0:
        topic = ' '.join(tokenize(tops[-1])[:4]) or 'the passage theme'
        hints.append(f"Hint 1 (General): Think about {topic}.")
    if len(tops) > 1:
        hints.append(f"Hint 2 (Medium): {tops[1][:90]}...")
    if len(tops) > 2:
        hints.append(f"Hint 3 (Specific): {tops[0][:130]}...")

    while len(hints) < 3:
        hints.append("Re-read the passage for more clues.")
    return hints[:3]


# =============================================================================
# 4. MAIN PIPELINE
# =============================================================================

def train_and_save():
    print("=" * 60)
    print("  MODEL B — DISTRACTOR & HINT RANKERS (Integrated)")
    print("=" * 60)

    # ── Row-count controls ────────────────────────────────────────────────────
    # DATA_SIZE caps the training split only.
    # Val is always kept at its full original size — DATA_SIZE never shrinks it.
    # The legacy MAX_TRAIN_ROWS env-var still overrides DATA_SIZE when set.
    max_train_rows = int(os.environ.get('MAX_TRAIN_ROWS', '0') or 0) or DATA_SIZE
    max_train_rows = 15000
    max_val_rows   = int(os.environ.get('MAX_VAL_ROWS',  '0') or 0)  # val: NOT capped by DATA_SIZE

    if DATA_SIZE:
        print(f"[*] DATA_SIZE={DATA_SIZE} — training capped at {DATA_SIZE} rows; val is NOT capped")

    train_csv  = f'{PROCESSED_DIR}/train_clean.csv'
    val_csv    = f'{PROCESSED_DIR}/val_clean.csv'
    tfidf_path = f'{PROCESSED_DIR}/tfidf_vectorizer.pkl'

    if not os.path.exists(train_csv):
        print("  [!] train_clean.csv missing. Run preprocessing first.")
        return

    df_train = pd.read_csv(train_csv)
    df_val   = pd.read_csv(val_csv) if os.path.exists(val_csv) else df_train.iloc[:200]

    # Cap train only ──────────────────────────────────────────────────────────
    if max_train_rows and len(df_train) > max_train_rows:
        df_train = df_train.sample(max_train_rows, random_state=42).reset_index(drop=True)
        print(f"  [*] Training split capped at {len(df_train)} rows "
              f"(val kept at full {len(df_val)} rows)")

    # Cap val only when the legacy MAX_VAL_ROWS env-var was explicitly set
    if max_val_rows and len(df_val) > max_val_rows:
        df_val = df_val.sample(max_val_rows, random_state=42).reset_index(drop=True)
        print(f"  [*] Val split capped at {len(df_val)} rows (MAX_VAL_ROWS env-var)")

    v_tfidf = None
    if os.path.exists(tfidf_path):
        with open(tfidf_path, 'rb') as f:
            v_tfidf = pickle.load(f)
        print("  [*] Loaded TF-IDF vectorizer for advanced Cosine Similarity.")
    else:
        print("  [*] No TF-IDF vectorizer found. Using Jaccard fallback.")

    print("\n  [1/3] Building Datasets...")
    print("    Caching train rows...")
    train_cache = build_row_cache(df_train)
    print(f"    Cached {len(train_cache)} train rows")

    print("    Caching val rows...")
    val_cache = build_row_cache(df_val)
    print(f"    Cached {len(val_cache)} val rows")

    Xd_tr, yd_tr = build_dist_dataset_cached(train_cache, v_tfidf)
    Xd_va, yd_va = build_dist_dataset_cached(val_cache,   v_tfidf)
    Xh_tr, yh_tr = build_hint_dataset_cached(train_cache, v_tfidf)
    Xh_va, yh_va = build_hint_dataset_cached(val_cache,   v_tfidf)

    print("\n  [2/3] Training Logistic Rankers...")
    use_numba = os.environ.get('USE_NUMBA', '0').lower() in ('1', 'true', 'yes')

    if use_numba and NUMBA_AVAILABLE:
        print('    [*] USE_NUMBA=1 detected — using numba trainers for dense rankers')
        try:
            with tqdm(total=1, desc='Train Distractor (numba)', leave=False) as pbar:
                dist_clf = train_lr_numba(Xd_tr, yd_tr); pbar.update(1)
        except Exception as e:
            print('    [!] numba distractor trainer failed, falling back to sklearn:', e)
            dist_clf = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced',
                                          solver='lbfgs', random_state=42)
            if len(yd_tr) > 0:
                with tqdm(total=1, desc='Train Distractor (sklearn)', leave=False) as pbar:
                    dist_clf.fit(Xd_tr, yd_tr); pbar.update(1)

        try:
            with tqdm(total=1, desc='Train Hint (numba)', leave=False) as pbar:
                hint_clf = train_lr_numba(Xh_tr, yh_tr); pbar.update(1)
        except Exception as e:
            print('    [!] numba hint trainer failed, falling back to sklearn:', e)
            hint_clf = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced',
                                          solver='lbfgs', random_state=42)
            if len(yh_tr) > 0:
                with tqdm(total=1, desc='Train Hint (sklearn)', leave=False) as pbar:
                    hint_clf.fit(Xh_tr, yh_tr); pbar.update(1)
    else:
        dist_clf = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced',
                                      solver='lbfgs', random_state=42)
        hint_clf = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced',
                                      solver='lbfgs', random_state=42)
        if len(yd_tr) > 0:
            with tqdm(total=1, desc='Train Distractor', leave=False) as pbar:
                dist_clf.fit(Xd_tr, yd_tr); pbar.update(1)
        if len(yh_tr) > 0:
            with tqdm(total=1, desc='Train Hint', leave=False) as pbar:
                hint_clf.fit(Xh_tr, yh_tr); pbar.update(1)

    print("\n  [3/3] Validation Metrics...")
    metrics_dict = {}

    if len(yd_va) > 0:
        d_preds = dist_clf.predict(Xd_va)
        acc  = accuracy_score(yd_va, d_preds)
        prec = precision_score(yd_va, d_preds, zero_division=0)
        rec  = recall_score(yd_va, d_preds, zero_division=0)
        f1   = f1_score(yd_va, d_preds, zero_division=0)
        metrics_dict['distractor'] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
        try:
            metrics_dict['distractor']['confusion_matrix'] = confusion_matrix(yd_va, d_preds).tolist()
        except Exception:
            metrics_dict['distractor']['confusion_matrix'] = None
        print(f"    Distractor Ranker | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

    if len(yh_va) > 0:
        h_preds = hint_clf.predict(Xh_va)
        h_acc = accuracy_score(yh_va, h_preds)
        h_prec = precision_score(yh_va, h_preds, zero_division=0)
        h_rec = recall_score(yh_va, h_preds, zero_division=0)
        h_f1 = f1_score(yh_va, h_preds, zero_division=0)
        metrics_dict['hint'] = {
            'accuracy': h_acc,
            'precision': h_prec,
            'recall': h_rec,
            'f1': h_f1,
        }
        try:
            metrics_dict['hint']['confusion_matrix'] = confusion_matrix(yh_va, h_preds).tolist()
        except Exception:
            metrics_dict['hint']['confusion_matrix'] = None
        print(f"    Hint Scorer       | Acc: {h_acc:.4f} | Prec: {h_prec:.4f} | Rec: {h_rec:.4f} | F1: {h_f1:.4f}")

    # Generation metrics (BLEU/ROUGE/METEOR) on full val set
    try:
        # from evaluate import GenerationMetricsEvaluator
        gen_eval = GenerationMetricsEvaluator()
        distractor_refs, distractor_hyps, hint_refs, hint_hyps = [], [], [], []
        topk_stats      = {1: {'hits': 0, 'total': 0}, 3: {'hits': 0, 'total': 0}}
        hint_level_hits = {1: 0, 2: 0, 3: 0}
        hint_total      = 0

        for _, r in df_val.iterrows():
            article   = str(r.get('article', ''))
            ans_label = str(r.get('answer', '')).strip()
            if ans_label not in ['A', 'B', 'C', 'D']:
                continue
            correct = str(r.get(ans_label, ''))
            wrongs  = [str(r.get(opt, '')) for opt in ['A', 'B', 'C', 'D'] if opt != ans_label]
            gen = generate_distractors(dist_clf, article, correct, v_tfidf, n=3)
            distractor_refs.append(' || '.join([w for w in wrongs if w]))
            distractor_hyps.append(' || '.join(gen))

            sents = split_into_sentences(article)
            if not sents:
                continue
            try:
                question = r.get('question', '')
                sims = ([tfidf_cosine_similarity(question, s, v_tfidf) for s in sents]
                        if v_tfidf else [cosine_similarity_feature(question, s) for s in sents])
                best_idx = int(np.argmax(sims))
                hint_refs.append(sents[best_idx])
                tops = generate_graduated_hints(hint_clf, article, question, v_tfidf)
                hint_hyps.append(tops[0] if tops else '')

                gold_set = {w.lower() for w in wrongs if w}
                if gold_set:
                    for k in [1, 3]:
                        hits = sum(1 for p in [x.lower() for x in gen[:k]] if p in gold_set)
                        topk_stats[k]['hits']  += hits
                        topk_stats[k]['total'] += k

                scored_idx = sorted(
                    [(hint_clf.predict_proba([compute_hint_vector(s, question, i, len(sents), v_tfidf)])[0][1], i)
                     for i, s in enumerate(sents)],
                    reverse=True,
                )
                top_indices = [idx for _, idx in scored_idx[:3]]
                hint_total += 1
                for n_top in [1, 2, 3]:
                    if best_idx in top_indices[:n_top]:
                        hint_level_hits[n_top] += 1
            except Exception:
                continue

        if distractor_refs and distractor_hyps:
            d_m = gen_eval.evaluate_generation(distractor_refs, distractor_hyps)
            metrics_dict['distractor_generation'] = d_m
            print(f"    Distractor Gen -> BLEU: {d_m['bleu']:.4f} | ROUGE-L: {d_m['rouge_l']:.4f} | METEOR: {d_m['meteor']:.4f}")

            pk = {}
            for k, stats in topk_stats.items():
                p_k = stats['hits'] / max(stats['total'], 1)
                r_k = stats['hits'] / (len(df_val) * 3) if len(df_val) > 0 else 0.0
                f_k = 2 * p_k * r_k / (p_k + r_k) if p_k + r_k > 0 else 0.0
                pk[k] = {'precision_at_k': p_k, 'recall_at_k': r_k, 'f1_at_k': f_k}
            metrics_dict['distractor_topk'] = pk
            print(f"    Distractor Top-K -> P@1: {pk[1]['precision_at_k']:.4f} P@3: {pk[3]['precision_at_k']:.4f} | "
                  f"R@1: {pk[1]['recall_at_k']:.4f} R@3: {pk[3]['recall_at_k']:.4f}")

        if hint_refs and hint_hyps:
            h_m = gen_eval.evaluate_generation(hint_refs, hint_hyps)
            metrics_dict['hint_generation'] = h_m
            print(f"    Hint Gen -> BLEU: {h_m['bleu']:.4f} | ROUGE-L: {h_m['rouge_l']:.4f} | METEOR: {h_m['meteor']:.4f}")

            hint_rates = ({n: hint_level_hits[n] / hint_total for n in [1, 2, 3]}
                          if hint_total > 0 else {1: 0.0, 2: 0.0, 3: 0.0})
            metrics_dict['hint_tier_hits'] = hint_rates
            print(f"    Hint Tier Hits -> Top1: {hint_rates[1]:.4f} | Top2: {hint_rates[2]:.4f} | Top3: {hint_rates[3]:.4f}")
    except Exception as e:
        print('    [!] Generation metrics skipped:', e)

    joblib.dump(dist_clf,     f'{OUT_DIR}/distractor_ranker.pkl')
    joblib.dump(hint_clf,     f'{OUT_DIR}/hint_scorer.pkl')
    joblib.dump(metrics_dict, f'{OUT_DIR}/metrics.pkl')
    print(f"\n====== Pipeline Complete. Models saved to {OUT_DIR} ======")


if __name__ == '__main__':
    train_and_save()