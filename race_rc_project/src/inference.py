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
import re
import time
import random

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity as _sk_cos
from scipy.sparse import hstack as sparse_hstack

try:
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
except NameError:
    PROJECT_ROOT = os.getcwd()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import clean_text, tokenize, split_into_sentences, tfidf_cosine, PROCESSED_DIR, BASE_DIR
from src.model_a_train import generate_questions_from_passage, make_sample_string, verify_answer as verify_answer_ma
from src.nn_models import load_checkpoint, AnswerVerifier, DistractorScorer, HintScorer, TransformerAnswerVerifier

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

MODEL_A_DIR = os.path.join(BASE_DIR, 'models', 'model_a', 'neural')
MODEL_A_TRANSFORMER_DIR = os.path.join(BASE_DIR, 'models', 'model_a', 'transformer')
MODEL_B_DIR = os.path.join(BASE_DIR, 'models', 'model_b', 'neural')
MODEL_B_TRANSFORMER_DIR = os.path.join(BASE_DIR, 'models', 'model_b', 'transformer')

MIN_QUIZ_QUESTIONS = 5
MAX_QUIZ_QUESTIONS = 10


_cache = {}


def _try_load_transformers(cache):
    """Load transformer models if available."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return

    # Transformer answer verifier
    meta_path = os.path.join(MODEL_A_TRANSFORMER_DIR, 'transformer_meta.json')
    ckpt_path = os.path.join(MODEL_A_TRANSFORMER_DIR, 'answer_verifier_transformer.pt')
    if os.path.exists(ckpt_path) and os.path.exists(meta_path):
        try:
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            tokenizer = AutoTokenizer.from_pretrained(meta.get('model_name', 'bert-base-uncased'))
            model = TransformerAnswerVerifier(model_name=meta['model_name'], num_labels=2)
            model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
            model.eval()
            cache['transformer_model'] = model
            cache['transformer_tokenizer'] = tokenizer
            print(f"  loaded transformer answer verifier ({meta['model_name']})")
        except Exception as e:
            print(f"  WARNING: transformer model load failed: {e}")

    # Question generator
    qg_path = os.path.join(MODEL_B_TRANSFORMER_DIR, 'question_generator.pt')
    qg_meta_path = os.path.join(MODEL_B_TRANSFORMER_DIR, 'qg_meta.json')
    if os.path.exists(qg_path) and os.path.exists(qg_meta_path):
        try:
            import json
            with open(qg_meta_path) as f:
                meta = json.load(f)
            from src.nn_models import QuestionGenerator
            cache['question_generator'] = QuestionGenerator(
                model_name=meta['model_name'], use_lora=False
            )
            cache['question_generator'].load_state_dict(
                torch.load(qg_path, map_location='cpu', weights_only=True)
            )
            cache['question_generator'].eval()
            print(f"  loaded question generator ({meta['model_name']})")
        except Exception as e:
            print(f"  WARNING: question generator load failed: {e}")

    # Distractor generator
    dg_path = os.path.join(MODEL_B_TRANSFORMER_DIR, 'distractor_generator.pt')
    dg_meta_path = os.path.join(MODEL_B_TRANSFORMER_DIR, 'dg_meta.json')
    if os.path.exists(dg_path) and os.path.exists(dg_meta_path):
        try:
            import json
            with open(dg_meta_path) as f:
                meta = json.load(f)
            from src.nn_models import DistractorGenerator
            cache['distractor_generator'] = DistractorGenerator(
                model_name=meta['model_name'], use_lora=False
            )
            cache['distractor_generator'].load_state_dict(
                torch.load(dg_path, map_location='cpu', weights_only=True)
            )
            cache['distractor_generator'].eval()
            print(f"  loaded distractor generator ({meta['model_name']})")
        except Exception as e:
            print(f"  WARNING: distractor generator load failed: {e}")


def load_models():
    """Load all trained models from disk on first call; return cache thereafter."""
    if _cache:
        return _cache

    # Load NN answer verifier
    nn_path = os.path.join(MODEL_A_DIR, 'answer_verifier.pt')
    if os.path.exists(nn_path):
        try:
            ckpt = torch.load(nn_path, map_location='cpu', weights_only=True)
            meta = ckpt.get('meta', {})
            input_dim = meta.get('input_dim', 10000)
            model = AnswerVerifier(input_dim=input_dim)
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()
            _cache['model'] = model
        except Exception as e:
            print(f"  WARNING: could not load NN model: {e}")
            _cache['model'] = None
    else:
        print(f"  WARNING: NN checkpoint not found at {nn_path}.")
        _cache['model'] = None

    # Load NN distractor scorer
    dist_path = os.path.join(MODEL_B_DIR, 'distractor.pt')
    if os.path.exists(dist_path):
        try:
            _cache['dist_ranker'] = load_checkpoint(DistractorScorer, dist_path)
        except Exception as e:
            print(f"  WARNING: could not load distractor model: {e}")
            _cache['dist_ranker'] = None
    else:
        print(f"  WARNING: distractor checkpoint not found at {dist_path}.")
        _cache['dist_ranker'] = None

    # Load NN hint scorer
    hint_path = os.path.join(MODEL_B_DIR, 'hint.pt')
    if os.path.exists(hint_path):
        try:
            _cache['hint_scorer'] = load_checkpoint(HintScorer, hint_path)
        except Exception as e:
            print(f"  WARNING: could not load hint model: {e}")
            _cache['hint_scorer'] = None
    else:
        print(f"  WARNING: hint checkpoint not found at {hint_path}.")
        _cache['hint_scorer'] = None

    # Try loading transformer models
    _try_load_transformers(_cache)

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


# ---------------------------------------------------------------------------
# Handcrafted features + ensemble prediction
# ---------------------------------------------------------------------------

ENSEMBLE_WEIGHTS = {
    "nn": 0.35,
    "transformer": 0.50,
    "tfidf": 0.15,
}


def _compute_hc_row(article, question, opt_text, tfidf_vec, idf_dict=None, vocab=None):
    """Compute 12 handcrafted features for one option (mirrors preprocessing.py)."""
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
    art_bigrams = set(" ".join(article.split()[i:i+2]) for i in range(len(article.split()) - 1))
    opt_bigrams = set(" ".join(opt_text.split()[i:i+2]) for i in range(len(opt_text.split()) - 1))
    shared_bigrams = len(art_bigrams & opt_bigrams) / (len(opt_bigrams) + 1e-9) if opt_bigrams else 0.0
    starts_cap = 1.0 if opt_text and opt_text[0].isupper() else 0.0

    return [sim_ao, sim_qo, overlap_ao, overlap_qo, opt_len_ratio,
            overlap_oa, overlap_oq, opt_idf_mean, has_digit,
            shared_bigrams, opt_tfidf_max, starts_cap]


def _nn_predict_proba(article, question, option_texts, nn_model, ohe_vec, tfidf_vec):
    """P(correct) for each option from the NN with handcrafted features."""
    idf_dict = dict(zip(tfidf_vec.get_feature_names_out(), tfidf_vec.idf_)) if tfidf_vec is not None else None
    vocab = set(tfidf_vec.get_feature_names_out()) if tfidf_vec is not None else None
    samples = ohe_vec.transform([make_sample_string(article, question, t) for t in option_texts])
    if tfidf_vec is not None:
        hc = np.array([_compute_hc_row(article, question, t, tfidf_vec, idf_dict, vocab) for t in option_texts], dtype=np.float32)
        samples = sparse_hstack([samples, hc.astype(np.float32)], format="csr")
    return nn_model.predict_proba(samples)[:, 1]


def _transformer_predict_proba(article, question, option_texts, trf_model, tokenizer):
    """P(correct) for each option from the transformer."""
    from src.model_a_train import verify_answer_transformer
    probs = []
    for t in option_texts:
        out = verify_answer_transformer(article, question, t, trf_model, tokenizer)
        probs.append(out["probability"])
    return np.array(probs)


def _tfidf_scores(article, question, option_texts, tfidf_vec):
    """Normalized TF-IDF cosine similarity scores for each option."""
    scores = np.array([tfidf_cosine(article, t, tfidf_vec) for t in option_texts])
    total = scores.sum() + 1e-9
    return scores / total


def _build_fallback_distractors(article, correct_answer, n=3):
    ans_set = set(tokenize(correct_answer))
    pool = list(dict.fromkeys([
        t for t in tokenize(article) if t not in ans_set and len(t) > 3
    ]))
    while len(pool) < n:
        pool.append(f"option {len(pool) + 1}")
    return pool[:n]


def _model_a_pack(models):
    """Return a dict that verify_answer_ma (the NN-based one) expects."""
    return {'model': models.get('model')}


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

    def _norm_prob(p_chosen, p_correct):
        return p_chosen / (p_chosen + p_correct + 1e-9)

    # --- Ensemble: combine NN + transformer + TF-IDF ---
    option_texts = [chosen_option_text, correct_answer_text]
    weights = {}
    probs = {}

    # NN
    if models.get('model') and models.get('ohe') and models.get('tfidf'):
        try:
            p = _nn_predict_proba(
                clean_text(article), clean_text(question), option_texts,
                models['model'], models['ohe'], models['tfidf'],
            )
            probs['nn'] = p
            weights['nn'] = ENSEMBLE_WEIGHTS['nn']
        except Exception:
            pass

    # Transformer
    if models.get('transformer_model') and models.get('transformer_tokenizer'):
        try:
            p = _transformer_predict_proba(
                article, question, option_texts,
                models['transformer_model'], models['transformer_tokenizer'],
            )
            probs['transformer'] = p
            weights['transformer'] = ENSEMBLE_WEIGHTS['transformer']
        except Exception:
            pass

    # TF-IDF
    if models.get('tfidf'):
        try:
            probs['tfidf'] = _tfidf_scores(article, question, option_texts, models['tfidf'])
            weights['tfidf'] = ENSEMBLE_WEIGHTS['tfidf']
        except Exception:
            pass

    if weights:
        total_w = sum(weights.values())
        p_ens = sum(weights[k] * probs[k] for k in weights) / total_w
        conf = p_ens[0]  # P(correct) for chosen_text
        method = "ensemble (" + "+".join(weights.keys()) + ")"
        return {
            'is_correct': is_correct,
            'confidence': conf if is_correct else (1.0 - conf),
            'method': method,
        }

    # --- Fallback: direct text match ---
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
            model = row.get('model', 'nn')
            ma[model.lower()] = {
                'accuracy':  float(row.get('accuracy', 0)),
                'precision': float(row.get('precision', 0)),
                'recall':    float(row.get('recall', 0)),
                'f1':        float(row.get('f1', 0)),
                '4way_acc':  float(row.get('4way_acc', 0)),
            }
        metrics['model_a'] = ma

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
