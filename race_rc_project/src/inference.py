"""
inference.py
Unified inference API — used by the Streamlit UI and the Colab notebook.

Public functions:
  run_inference(article, race_rows=None) → {'questions': [...], 'latency_ms': N}
  verify_answer(article, question, chosen_text, correct_text) → dict
  get_model_metrics() → dict
"""

import os
import sys
import pickle
import time
import random

import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import clean_text, tokenize, split_into_sentences, PROCESSED_DIR, BASE_DIR
from model_a_train import generate_questions_from_passage, verify_answer as verify_answer_ma

try:
    from model_b_train import generate_distractors, generate_hints
except Exception:
    def generate_distractors(article, answer, model, n=3):
        return []

    def generate_hints(article, question, model, n=3):
        sents = split_into_sentences(article)
        if not sents:
            return ["Read carefully", "Focus keywords", "Check passage"]
        return [f"Hint {i+1}: {s[:100]} …" for i, s in enumerate(sents[:n])]

MODEL_A_DIR = os.path.join(BASE_DIR, 'models', 'model_a', 'traditional')
MODEL_B_DIR = os.path.join(BASE_DIR, 'models', 'model_b', 'traditional')

MIN_QUIZ_QUESTIONS = 5
MAX_QUIZ_QUESTIONS = 10


_cache = {}


def load_models():
    """Load all trained models from disk on first call; return cache thereafter."""
    if _cache:
        return _cache

    def safe_load(path, label):
        if os.path.exists(path):
            return joblib.load(path)
        print(f"  WARNING: {label} not found at {path}.")
        return None

    _cache['lr']  = safe_load(os.path.join(MODEL_A_DIR, 'logistic_regression.pkl'), 'LR')
    _cache['svm'] = safe_load(os.path.join(MODEL_A_DIR, 'svm.pkl'),                 'SVM')
    _cache['meta'] = safe_load(os.path.join(MODEL_A_DIR, 'stacking_meta.pkl'),      'StackMeta')
    _cache['dist_ranker'] = safe_load(os.path.join(MODEL_B_DIR, 'distractor.pkl'),  'Distractor')
    _cache['hint_scorer'] = safe_load(os.path.join(MODEL_B_DIR, 'hint.pkl'),       'Hint')

    def _try_pickle(path, label):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"  WARNING: could not load {label} ({path}): {e}")
            return None

    for pkl_key, fname in [('ohe', 'ohe_vectorizer.pkl'), ('tfidf', 'tfidf_vectorizer.pkl')]:
        path = os.path.join(PROCESSED_DIR, fname)
        if os.path.exists(path):
            _cache[pkl_key] = _try_pickle(path, fname)
        else:
            alt = os.path.join(PROCESSED_DIR, 'tfidf.pkl') if pkl_key == 'tfidf' else None
            if alt and os.path.exists(alt):
                _cache[pkl_key] = _try_pickle(alt, 'tfidf.pkl')
            else:
                print(f"  WARNING: {fname} not found. Run preprocessing / training first.")
                _cache[pkl_key] = None

    return _cache


def remove_answer_from_question(question_text, answer_text):
    """Utility: replace answer tokens in question with '___'."""
    import re
    for token in tokenize(answer_text):
        if len(token) > 3:
            pattern = r'\b' + re.escape(token) + r'\b'
            question_text = re.sub(pattern, '___', question_text, flags=re.IGNORECASE)
    return question_text


def shuffle_options(correct_answer, distractors):
    """Randomly assign correct answer + 3 distractors to slots A/B/C/D."""
    pool = [correct_answer] + distractors[:3]
    random.shuffle(pool)
    idx  = pool.index(correct_answer)
    lbls = ['A', 'B', 'C', 'D']
    return {lbls[i]: pool[i] for i in range(4)}, lbls[idx]


def _get_hints(article, question, models):
    if models.get('hint_scorer'):
        return generate_hints(article, question, models['hint_scorer'], n=3)
    sents = split_into_sentences(article)
    return [f"Hint {i+1}: {s[:100]} …" for i, s in enumerate(sents[:3])]


def _build_fallback_distractors(article, correct_answer, n=3):
    ans_set = set(tokenize(correct_answer))
    pool = list(dict.fromkeys([
        t for t in tokenize(article) if t not in ans_set and len(t) > 3
    ]))
    while len(pool) < n:
        pool.append(f"option {len(pool) + 1}")
    return pool[:n]


def _model_a_pack(models):
    return {
        'lr':   models.get('lr'),
        'svm':  models.get('svm'),
        'meta': models.get('meta'),
    }


def _build_fallback_question(article, idx, models):
    """Create a safe fallback MCQ so we can always reach the minimum quiz size."""
    sentences = split_into_sentences(article)
    source = sentences[min(idx, max(len(sentences) - 1, 0))] if sentences else article
    source = source.strip() if source else article.strip()

    question_text = f"Which statement best matches the passage detail #{idx + 1}?"
    correct_answer = source[:120] if source else "A key point from the passage"
    distractors = _build_fallback_distractors(article, correct_answer, n=3)
    options, correct_label = shuffle_options(correct_answer, distractors)
    hints = _get_hints(article, question_text, models)

    return {
        'question': question_text,
        'correct_answer': correct_answer,
        'correct_label': correct_label,
        'options': options,
        'hints': hints,
        'source_sentence': source[:200],
    }


def build_question_from_race_row(article, row, models):
    correct_letter = str(row.get('answer', 'A')).strip().upper()
    correct_answer = str(row.get(correct_letter, ''))
    hints = _get_hints(article, str(row.get('question', '')), models)

    return {
        'question':        str(row.get('question', '')),
        'correct_answer':  correct_answer,
        'correct_label':   correct_letter,
        'options': {
            'A': str(row.get('A', '')),
            'B': str(row.get('B', '')),
            'C': str(row.get('C', '')),
            'D': str(row.get('D', '')),
        },
        'hints':           hints,
        'source_sentence': article[:200],
    }


def build_question_from_generated(article, item, models):
    """Map model_a_train.generate_questions_from_passage() dict to UI shape."""
    q_text = item['question']
    opts = item.get('options') or {}
    correct_letter = str(item.get('correct_letter', 'A')).strip().upper()
    correct_answer = item.get('answer', opts.get(correct_letter, ''))

    if models.get('dist_ranker') and correct_answer:
        distractors = generate_distractors(
            article, correct_answer, models['dist_ranker'], n=3
        )
        opts2, correct_letter = shuffle_options(correct_answer, distractors)
        opts = opts2

    hints = _get_hints(article, q_text, models)
    return {
        'question':        q_text,
        'correct_answer':  correct_answer,
        'correct_label':   correct_letter,
        'options':         opts,
        'hints':           hints,
        'source_sentence': str(item.get('source_sentence', '')),
    }


def run_inference(article, race_rows=None):
    t0 = time.time()
    models = load_models()

    question_list = []

    if race_rows:
        for row in race_rows[:MAX_QUIZ_QUESTIONS]:
            question_list.append(build_question_from_race_row(article, row, models))

    n_needed = MIN_QUIZ_QUESTIONS - len(question_list)
    if n_needed > 0:
        try:
            raw = generate_questions_from_passage(article, count=n_needed + 2)
            for item in raw[:n_needed]:
                question_list.append(build_question_from_generated(article, item, models))
        except Exception as e:
            pass

    # Guarantee minimum count even when model generation returns too few items.
    while len(question_list) < MIN_QUIZ_QUESTIONS:
        question_list.append(_build_fallback_question(article, len(question_list), models))

    # Never exceed the configured maximum quiz size.
    question_list = question_list[:MAX_QUIZ_QUESTIONS]

    latency_ms = int((time.time() - t0) * 1000)
    print(f"  run_inference: {len(question_list)} questions in {latency_ms} ms")

    return {'questions': question_list, 'latency_ms': latency_ms}


def verify_answer(article, question, chosen_option_text, correct_answer_text):
    models = load_models()
    is_correct = clean_text(chosen_option_text) == clean_text(correct_answer_text)

    pack = _model_a_pack(models)
    if pack['lr'] and pack['svm'] and models.get('ohe'):
        try:
            out = verify_answer_ma(
                article, question, chosen_option_text, pack, models['ohe']
            )
            conf = float(out['probability'])
            return {
                'is_correct': is_correct,
                'confidence': conf if is_correct else (1.0 - conf),
                'method': 'stacking ensemble + direct match',
            }
        except Exception:
            pass

    if models.get('tfidf'):
        try:
            from preprocessing import tfidf_cosine
            cos_chosen = tfidf_cosine(article, chosen_option_text, models['tfidf'])
            cos_correct = tfidf_cosine(article, correct_answer_text, models['tfidf'])
            conf = cos_chosen / (cos_chosen + cos_correct + 1e-9)
            return {
                'is_correct': is_correct,
                'confidence': conf if is_correct else (1.0 - conf),
                'method': 'tfidf cosine similarity',
            }
        except Exception:
            pass

    return {
        'is_correct': is_correct,
        'confidence': 1.0 if is_correct else 0.0,
        'method': 'direct match',
    }


def get_model_metrics():
    reports_dir = os.path.join(PROCESSED_DIR, 'reports')
    metrics = {}

    ma_csv = os.path.join(reports_dir, 'model_a_binary_metrics.csv')
    if os.path.exists(ma_csv):
        df = pd.read_csv(ma_csv)
        ma = {}
        for _, row in df.iterrows():
            model = row.get('model', row.get('strategy', 'unknown'))
            if model in ('LR', 'SVM'):
                ma[model.lower()] = {
                    'accuracy':  float(row.get('accuracy', 0)),
                    'precision': float(row.get('precision', 0)),
                    'recall':    float(row.get('recall', 0)),
                    'f1':        float(row.get('f1', 0)),
                    '4way_acc':  float(row.get('4way_acc', 0)),
                }
        metrics['model_a'] = ma

    ens_csv = os.path.join(reports_dir, 'model_a_ensemble_metrics.csv')
    if os.path.exists(ens_csv):
        ens_df = pd.read_csv(ens_csv)
        if 'model_a' not in metrics:
            metrics['model_a'] = {}
        ens = {}
        for _, row in ens_df.iterrows():
            strat = str(row.get('strategy', ''))
            ens[strat] = {
                k: float(v) for k, v in row.items()
                if k != 'strategy' and pd.notna(v)
            }
        metrics['model_a']['ensemble'] = ens

    cos_csv = os.path.join(reports_dir, 'model_a_cosine_retrieval_metrics.csv')
    if os.path.exists(cos_csv):
        if 'model_a' not in metrics:
            metrics['model_a'] = {}
        metrics['model_a']['cosine_similarity'] = {
            k: float(v) for k, v in pd.read_csv(cos_csv).iloc[0].items()
            if k != 'strategy'
        }

    gen_csv = os.path.join(reports_dir, 'model_a_text_generation_metrics.csv')
    if os.path.exists(gen_csv):
        metrics['text_generation'] = {
            k: float(v) for k, v in pd.read_csv(gen_csv).iloc[0].items()
        }

    mb_csv = os.path.join(reports_dir, 'model_b_metrics.csv')
    if os.path.exists(mb_csv):
        mb = {}
        for _, row in pd.read_csv(mb_csv).iterrows():
            model = str(row.get('model', ''))
            split = str(row.get('split', ''))
            key = f"{model}_{split}"
            mb[key] = {
                'accuracy':  float(row.get('accuracy', 0)),
                'f1':        float(row.get('f1', 0)),
                'precision': float(row.get('precision', 0)),
                'recall':    float(row.get('recall', 0)),
            }
        metrics['model_b'] = mb

    return metrics


if __name__ == '__main__':
    sample = (
        "The Amazon rainforest is often referred to as the lungs of the Earth. "
        "It produces 20 percent of the world's oxygen and is home to more than "
        "10 million species of plants, animals, and insects. The rainforest covers "
        "5.5 million square kilometres across nine countries. Deforestation is one "
        "of the biggest threats to this vital ecosystem. Scientists are urging "
        "governments and companies to take immediate action to protect the forest."
    )
    result = run_inference(sample)
    for i, q in enumerate(result['questions'], 1):
        print(f"\n--- Question {i}/5 ---")
        print(f"Q : {q['question']}")
        for k, v in q['options'].items():
            marker = " ← correct" if k == q['correct_label'] else ""
            print(f"  {k}: {v}{marker}")
    print(f"\nLatency: {result['latency_ms']} ms")
