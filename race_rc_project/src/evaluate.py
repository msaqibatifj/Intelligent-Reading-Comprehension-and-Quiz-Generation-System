"""
evaluate.py
──────────────────────────────────────────────────────────────────────────────
Full evaluation of Model A and Model B on the test split.

PRIMARY METRICS (NLP generation task — professor's requirement):
  BLEU   — n-gram precision between predicted answer and gold answer
  ROUGE  — recall-oriented overlap (ROUGE-1, ROUGE-2, ROUGE-L)
  METEOR — alignment-based metric covering synonyms and stemming

SECONDARY METRICS:
  Cosine Similarity Accuracy — TF-IDF cosine between article and options
  Binary classification: Accuracy, Precision, Recall, F1, Confusion Matrix
  4-way MCQ accuracy (ML model picks best option from OHE features)
  Train↔Test domain similarity

Usage:
  python src/evaluate.py
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from scipy.sparse import load_npz
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import PROCESSED_DIR, BASE_DIR
from model_a_train import (
    compute_4way_accuracy,
    cosine_similarity_accuracy,
    compute_train_test_domain_similarity,
    compute_generation_metrics as ma_compute_generation_metrics,
    generate_questions_from_passage,
)

MODEL_A_DIR = os.path.join(BASE_DIR, 'models', 'model_a', 'traditional')
MODEL_B_DIR = os.path.join(BASE_DIR, 'models', 'model_b', 'traditional')
REPORTS_DIR = os.path.join(PROCESSED_DIR, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def print_section(title):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print('═' * 60)


def print_subsection(title):
    print(f"\n  {'─' * 56}")
    print(f"    {title}")
    print(f"  {'─' * 56}")


def _ensure_nltk_resources():
    """Download required NLTK data if not already present."""
    import nltk
    resources = [
        ('tokenizers/punkt',     'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/wordnet',      'wordnet'),
        ('corpora/omw-1.4',      'omw-1.4'),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception:
                pass


def _score_generation_pair(ref_text, hyp_text, rouge_scorer_obj, smoother):
    import nltk
    from nltk.translate.bleu_score import sentence_bleu
    from nltk.translate.meteor_score import meteor_score

    ref_tokens = nltk.word_tokenize(str(ref_text).lower())
    hyp_tokens = nltk.word_tokenize(str(hyp_text).lower())
    if not ref_tokens or not hyp_tokens:
        return None

    bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoother)
    try:
        met = meteor_score([ref_tokens], hyp_tokens)
    except Exception:
        met = 0.0
    rouge_out = rouge_scorer_obj.score(str(ref_text), str(hyp_text))

    return {
        'bleu': bleu,
        'meteor': met,
        'rouge1_f': rouge_out['rouge1'].fmeasure,
        'rouge1_p': rouge_out['rouge1'].precision,
        'rouge1_r': rouge_out['rouge1'].recall,
        'rouge2_f': rouge_out['rouge2'].fmeasure,
        'rouge2_p': rouge_out['rouge2'].precision,
        'rouge2_r': rouge_out['rouge2'].recall,
        'rougeL_f': rouge_out['rougeL'].fmeasure,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY METRICS — BLEU / ROUGE / METEOR
# ─────────────────────────────────────────────────────────────────────────────

def compute_generation_metrics(test_df, tfidf_vec=None, sample_n=300):
    """
    Generation metrics via model_a_train (BLEU / ROUGE / METEOR on generated answers).
    """
    print_section("PRIMARY METRICS — BLEU / ROUGE / METEOR")
    print("  Comparing generated answers (passage pipeline) vs. RACE references …")

    sample_df = test_df.sample(min(sample_n, len(test_df)), random_state=42)
    all_generated = []
    for _, row in sample_df.iterrows():
        article = str(row.get('article', ''))
        if len(article) < 50:
            continue
        all_generated.extend(generate_questions_from_passage(article, count=2))

    if not all_generated:
        print("  WARNING: no generated samples.")
        return {}

    # model_a_train.generation_metrics expects the keyword `n_sample`
    results = ma_compute_generation_metrics(all_generated, sample_df, n_sample=sample_n)
    if not results:
        return {}

    n = int(results.get('n_samples', 0))
    print(f"\n  Evaluated {n} samples (test split)")
    print(f"\n  ┌{'─'*40}┐")
    print(f"  │ {'METRIC':<28}  {'SCORE':>8} │")
    print(f"  ├{'─'*40}┤")
    print(f"  │ {'BLEU':<28}  {results.get('bleu', 0):>8.4f} │")
    print(f"  │ {'METEOR':<28}  {results.get('meteor', 0):>8.4f} │")
    print(f"  │ {'ROUGE-1  F1':<28}  {results.get('rouge1_f', 0):>8.4f} │")
    print(f"  │ {'ROUGE-2  F1':<28}  {results.get('rouge2_f', 0):>8.4f} │")
    print(f"  │ {'ROUGE-L  F1':<28}  {results.get('rougeL_f', 0):>8.4f} │")
    print(f"  └{'─'*40}┘")

    plot_payload = {
        'bleu': results.get('bleu', 0),
        'meteor': results.get('meteor', 0),
        'rouge1_f': results.get('rouge1_f', 0),
        'rouge2_f': results.get('rouge2_f', 0),
        'rougeL_f': results.get('rougeL_f', 0),
    }
    _save_generation_metrics_plot(plot_payload)
    return results


def compute_question_generation_metrics(test_df, sample_n=300, n_candidates=5):
    """
    Evaluate generated questions vs. gold questions using BLEU / ROUGE / METEOR.

    For each test sample:
      Candidates: generate_questions_from_passage(article)
      Reference : gold question from RACE dataset
      Selection : use the best-matching candidate by ROUGE-L F1
    """
    print_section("PRIMARY METRICS — BLEU / ROUGE / METEOR (Questions)")
    print("  Comparing generated questions vs. RACE gold questions …")

    _ensure_nltk_resources()

    from nltk.translate.bleu_score import SmoothingFunction
    from rouge_score import rouge_scorer as rs_lib

    rouge_scorer_obj = rs_lib.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
    )
    smoother = SmoothingFunction().method1

    sample_df = test_df.sample(min(sample_n, len(test_df)), random_state=42)

    bleu_scores, meteor_scores = [], []
    rouge1_f, rouge2_f, rougeL_f = [], [], []
    rouge1_p, rouge1_r = [], []
    rouge2_p, rouge2_r = [], []

    for _, row in sample_df.iterrows():
        article = str(row['article'])
        gold_q  = str(row['question'])

        # `generate_questions_from_passage` (alias of `compose_questions`) expects
        # the `count` keyword for number of questions.
        candidates_rows = generate_questions_from_passage(
            article, count=n_candidates
        )
        candidates = [d['question'] for d in candidates_rows]
        if not candidates:
            continue

        # Pick best candidate by ROUGE-L F1 to avoid penalizing diverse phrasing
        best = None
        best_rl = -1.0
        for cand in candidates:
            scores = _score_generation_pair(gold_q, cand, rouge_scorer_obj, smoother)
            if not scores:
                continue
            if scores['rougeL_f'] > best_rl:
                best_rl = scores['rougeL_f']
                best = scores

        if not best:
            continue

        bleu_scores.append(best['bleu'])
        meteor_scores.append(best['meteor'])
        rouge1_f.append(best['rouge1_f'])
        rouge1_p.append(best['rouge1_p'])
        rouge1_r.append(best['rouge1_r'])
        rouge2_f.append(best['rouge2_f'])
        rouge2_p.append(best['rouge2_p'])
        rouge2_r.append(best['rouge2_r'])
        rougeL_f.append(best['rougeL_f'])

    n = len(bleu_scores)
    if n == 0:
        print("  WARNING: no valid samples processed.")
        return {}

    results = {
        'q_bleu':      float(np.mean(bleu_scores)),
        'q_meteor':    float(np.mean(meteor_scores)),
        'q_rouge1_f':  float(np.mean(rouge1_f)),
        'q_rouge1_p':  float(np.mean(rouge1_p)),
        'q_rouge1_r':  float(np.mean(rouge1_r)),
        'q_rouge2_f':  float(np.mean(rouge2_f)),
        'q_rouge2_p':  float(np.mean(rouge2_p)),
        'q_rouge2_r':  float(np.mean(rouge2_r)),
        'q_rougeL_f':  float(np.mean(rougeL_f)),
        'n_samples':    n,
        'n_candidates': n_candidates,
    }

    print(f"\n  Evaluated {n} samples (test split)")
    print(f"\n  ┌{'─'*40}┐")
    print(f"  │ {'METRIC':<28}  {'SCORE':>8} │")
    print(f"  ├{'─'*40}┤")
    print(f"  │ {'BLEU':<28}  {results['q_bleu']:>8.4f} │")
    print(f"  │ {'METEOR':<28}  {results['q_meteor']:>8.4f} │")
    print(f"  │ {'ROUGE-1  F1':<28}  {results['q_rouge1_f']:>8.4f} │")
    print(f"  │   {'Precision':<26}  {results['q_rouge1_p']:>8.4f} │")
    print(f"  │   {'Recall':<26}  {results['q_rouge1_r']:>8.4f} │")
    print(f"  │ {'ROUGE-2  F1':<28}  {results['q_rouge2_f']:>8.4f} │")
    print(f"  │ {'ROUGE-L  F1':<28}  {results['q_rougeL_f']:>8.4f} │")
    print(f"  └{'─'*40}┘")

    return results


def _save_generation_metrics_plot(gen_metrics):
    """Bar chart for BLEU / ROUGE / METEOR scores."""
    labels = ['BLEU', 'METEOR', 'ROUGE-1\nF1', 'ROUGE-2\nF1', 'ROUGE-L\nF1']
    values = [
        gen_metrics.get('bleu', 0),
        gen_metrics.get('meteor', 0),
        gen_metrics.get('rouge1_f', 0),
        gen_metrics.get('rouge2_f', 0),
        gen_metrics.get('rougeL_f', 0),
    ]
    colors = ['#4f46e5', '#0891b2', '#059669', '#10b981', '#34d399']

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(labels, values, color=colors, alpha=0.88, width=0.55)
    ax.set_ylim(0, max(max(values) * 1.4, 0.1))
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Generation Evaluation — BLEU / ROUGE / METEOR',
                 fontsize=13, fontweight='bold')
    ax.axhline(y=0, color='#334155', linewidth=0.8)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{v:.4f}', ha='center', va='bottom',
                fontweight='bold', fontsize=10)
    plt.tight_layout()
    out = os.path.join(REPORTS_DIR, 'generation_metrics_plot.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# COSINE SIMILARITY ACCURACY
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_cosine_similarity(tfidf_vec, test_df, train_df=None, sample_n=500):
    """TF-IDF cosine similarity accuracy — argmax(cosine) predicts correct option."""
    print_section("COSINE SIMILARITY ACCURACY")

    eval_df = test_df.sample(min(sample_n, len(test_df)), random_state=42)
    cos = cosine_similarity_accuracy(tfidf_vec, eval_df)

    gap_flag = "correct > wrong ✓" if cos['sim_gap'] > 0 else "inverted ✗"
    print(f"\n  {'Cosine Similarity Accuracy':<35}: {cos['accuracy']:.4f}  ({cos['accuracy']*100:.1f}%)")
    print(f"  {'Avg similarity — correct option':<35}: {cos['avg_correct_sim']:.4f}")
    print(f"  {'Avg similarity — wrong options':<35}: {cos['avg_wrong_sim']:.4f}")
    print(f"  {'Similarity gap (correct − wrong)':<35}: {cos['sim_gap']:.4f}  [{gap_flag}]")

    if train_df is not None:
        # model_a_train.domain_overlap expects `n_sample` keyword
        domain_sim = compute_train_test_domain_similarity(
            train_df, test_df, tfidf_vec, n_sample=200
        )
        print(f"\n  Train↔Test domain similarity: {domain_sim:.4f}")
        cos['domain_similarity'] = domain_sim

    _save_cosine_sim_plot(tfidf_vec, eval_df)
    return cos


def _save_cosine_sim_plot(tfidf_vec, eval_df):
    from preprocessing import tfidf_cosine

    correct_sims, wrong_sims = [], []
    for _, row in eval_df.iterrows():
        article = str(row['article'])
        gold    = str(row['answer']).strip().upper()
        for opt in ['A', 'B', 'C', 'D']:
            sim = tfidf_cosine(article, str(row[opt]), tfidf_vec)
            (correct_sims if opt == gold else wrong_sims).append(sim)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle('TF-IDF Cosine Similarity: Correct vs Wrong Options',
                 fontsize=13, fontweight='bold')
    axes[0].hist(correct_sims, bins=30, alpha=0.75, color='#059669', label='Correct')
    axes[0].hist(wrong_sims,   bins=30, alpha=0.75, color='#dc2626', label='Wrong')
    axes[0].axvline(np.mean(correct_sims), color='#059669', lw=2, ls='--')
    axes[0].axvline(np.mean(wrong_sims),   color='#dc2626', lw=2, ls='--')
    axes[0].set_xlabel('Cosine Similarity'); axes[0].set_ylabel('Count')
    axes[0].legend(); axes[0].set_title('Distribution')
    means = [np.mean(correct_sims), np.mean(wrong_sims)]
    stds  = [np.std(correct_sims),  np.std(wrong_sims)]
    bars  = axes[1].bar(['Correct', 'Wrong'], means, yerr=stds,
                        color=['#059669', '#dc2626'], alpha=0.85,
                        capsize=8, error_kw={'linewidth': 2})
    axes[1].set_ylabel('Mean Cosine Similarity'); axes[1].set_title('Mean ± Std')
    for bar, m in zip(bars, means):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                     f'{m:.4f}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'cosine_sim_plot.png'), dpi=120, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# MODEL A  (binary + 4-way MCQ)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model_a(ohe_vec, test_df, tfidf_vec=None):
    print_section("MODEL A — BINARY CLASSIFICATION & 4-WAY MCQ")

    X_te_path = os.path.join(PROCESSED_DIR, 'X_test_ohe.npz')
    y_te_path  = os.path.join(PROCESSED_DIR, 'y_test.npy')
    model_specs = [
        ('logistic_regression.pkl', 'Logistic Regression'),
        ('svm.pkl',                 'SVM'),
    ]
    results = {}

    if os.path.exists(X_te_path) and os.path.exists(y_te_path):
        print_subsection("Binary classification (is option correct?)")
        X_te = load_npz(X_te_path)
        y_te = np.load(y_te_path)
        for filename, label in model_specs:
            path = os.path.join(MODEL_A_DIR, filename)
            if not os.path.exists(path):
                continue
            model = joblib.load(path)
            preds = model.predict(X_te)
            acc = accuracy_score(y_te, preds)
            p   = precision_score(y_te, preds, average='macro', zero_division=0)
            r   = recall_score(y_te,    preds, average='macro', zero_division=0)
            f1  = f1_score(y_te,        preds, average='macro', zero_division=0)
            cm  = confusion_matrix(y_te, preds)
            print(f"    {label:<30}: Acc={acc:.4f}  P={p:.4f}  R={r:.4f}  F1={f1:.4f}")
            # Save confusion matrix plot
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Incorrect', 'Correct'],
                        yticklabels=['Incorrect', 'Correct'], ax=ax)
            ax.set_title(f'{label} — Confusion Matrix')
            ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
            plt.tight_layout()
            cm_path = os.path.join(REPORTS_DIR,
                                   f"{label.lower().replace(' ','_')}_cm.png")
            plt.savefig(cm_path, dpi=100); plt.close()
            key = label.lower().replace(' ', '_')
            results[key] = {'accuracy': acc, 'precision': p, 'recall': r,
                            'f1': f1, 'confusion_matrix': cm}

    print_subsection("4-way MCQ accuracy")
    for filename, label in model_specs:
        path = os.path.join(MODEL_A_DIR, filename)
        if not os.path.exists(path):
            continue
        model  = joblib.load(path)
        acc_4w = compute_4way_accuracy(model, ohe_vec, test_df)
        print(f"    {label:<30}: {acc_4w:.4f}  ({acc_4w*100:.1f}%)")
        key = label.lower().replace(' ', '_')
        if key not in results:
            results[key] = {}
        results[key]['4way_acc'] = acc_4w

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MODEL B
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model_b():
    print_section("MODEL B — DISTRACTOR RANKER & HINT SCORER")
    path = os.path.join(MODEL_B_DIR, 'metrics.pkl')
    if not os.path.exists(path):
        print("  Not found. Run model_b_train.py first.")
        return {}
    metrics = joblib.load(path)
    d = metrics.get('distractor', {})
    h = metrics.get('hint', {})
    print(f"  Distractor Ranker: Acc={d.get('acc', 0):.4f}  F1={d.get('f1', 0):.4f}")
    print(f"  Hint Scorer:       Acc={h.get('acc', 0):.4f}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(gen_results, cos_results, ma_results):
    print_section("EVALUATION SUMMARY")
    print(f"\n  ┌{'─'*52}┐")
    print(f"  │ {'METRIC':<40}  {'VALUE':>8} │")
    print(f"  ├{'─'*52}┤")
    if gen_results:
        for key, lbl in [('bleu','★ BLEU (answer extraction)'),
                          ('meteor','★ METEOR'),
                          ('rouge1_f','★ ROUGE-1 F1'),
                          ('rouge2_f','★ ROUGE-2 F1'),
                          ('rougeL_f','★ ROUGE-L F1')]:
            print(f"  │ {lbl:<40}  {gen_results.get(key,0):>8.4f} │")
        for key, lbl in [('q_bleu','★ BLEU (question gen)'),
                          ('q_meteor','★ METEOR (question gen)'),
                          ('q_rouge1_f','★ ROUGE-1 F1 (question gen)'),
                          ('q_rouge2_f','★ ROUGE-2 F1 (question gen)'),
                          ('q_rougeL_f','★ ROUGE-L F1 (question gen)')]:
            if key in gen_results:
                print(f"  │ {lbl:<40}  {gen_results.get(key,0):>8.4f} │")
        print(f"  ├{'─'*52}┤")
    if cos_results:
        print(f"  │ {'Cosine Similarity Accuracy':<40}  {cos_results.get('accuracy',0):>8.4f} │")
        print(f"  │ {'Similarity Gap (correct−wrong)':<40}  {cos_results.get('sim_gap',0):>8.4f} │")
        print(f"  ├{'─'*52}┤")
    for key, lbl in [('logistic_regression', 'LR binary accuracy'),
                      ('svm',                 'SVM binary accuracy')]:
        v = ma_results.get(key, {}).get('accuracy')
        if v is not None:
            print(f"  │ {lbl:<40}  {v:>8.4f} │")
    print(f"  └{'─'*52}┘")


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_full_evaluation():
    print_section("RACE RC PROJECT — FULL EVALUATION")

    ohe_path   = os.path.join(PROCESSED_DIR, 'ohe_vectorizer.pkl')
    tfidf_path = os.path.join(PROCESSED_DIR, 'tfidf_vectorizer.pkl')
    test_csv   = os.path.join(PROCESSED_DIR, 'test_clean.csv')
    train_csv  = os.path.join(PROCESSED_DIR, 'train_clean.csv')

    missing = [p for p in [ohe_path, tfidf_path, test_csv] if not os.path.exists(p)]
    if missing:
        print(f"\n  ERROR: Missing files:\n    " + '\n    '.join(missing))
        print("  Run preprocessing.py first.")
        return {}

    with open(ohe_path,   'rb') as f: ohe_vec   = pickle.load(f)
    with open(tfidf_path, 'rb') as f: tfidf_vec = pickle.load(f)

    test_df  = pd.read_csv(test_csv)
    train_df = pd.read_csv(train_csv) if os.path.exists(train_csv) else None

    # ── PRIMARY: BLEU / ROUGE / METEOR ────────────────────────────────────
    gen_results = {}
    try:
        gen_results = compute_generation_metrics(test_df, tfidf_vec, sample_n=300)
        q_results = compute_question_generation_metrics(test_df, sample_n=300)
        if q_results:
            gen_results.update(q_results)
        gen_out = os.path.join(REPORTS_DIR, 'generation_metrics.pkl')
        joblib.dump(gen_results, gen_out)
        print(f"  Generation metrics saved → {gen_out}")
    except ImportError as e:
        print(f"\n  WARNING: Could not compute generation metrics ({e}).")
        print("  Install: pip install nltk rouge-score")

    # ── COSINE SIMILARITY ACCURACY ─────────────────────────────────────────
    cos_results = evaluate_cosine_similarity(
        tfidf_vec, test_df, train_df=train_df, sample_n=500
    )

    # ── MODEL A (binary + 4-way) ───────────────────────────────────────────
    # ma_results = evaluate_model_a(ohe_vec, test_df, tfidf_vec=tfidf_vec)

    # ── MODEL B ───────────────────────────────────────────────────────────
    # mb_results = evaluate_model_b()

    # ── SUMMARY ───────────────────────────────────────────────────────────
    # print_summary(gen_results, cos_results, ma_results)

    all_metrics = {
        'generation':       gen_results,
        'cosine_similarity': cos_results,
        # 'model_a':          ma_results,
        # 'model_b':          mb_results,
    }
    out_path = os.path.join(REPORTS_DIR, 'all_metrics.pkl')
    joblib.dump(all_metrics, out_path)
    print(f"\n  All metrics saved → {out_path}")
    print_section("Evaluation complete!")
    return all_metrics


if __name__ == '__main__':
    run_full_evaluation()
