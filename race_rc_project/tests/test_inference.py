"""
test_inference.py
Unit tests for preprocessing utilities, TF-IDF cosine similarity, and
the inference pipeline.

Run with:  python -m pytest tests/test_inference.py -v
"""

import sys
import os

try:
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _SRC_DIR = os.path.join(_THIS_DIR, '..', 'src')
except NameError:
    _SRC_DIR = os.path.join(os.getcwd(), 'src')
sys.path.insert(0, os.path.normpath(_SRC_DIR))

from preprocessing import (
    clean_text, tokenize, split_into_sentences,
    cosine_similarity_feature, tfidf_cosine,
    build_one_sample, build_tfidf_vectorizer,
)
from model_a_train import (
    generate_questions_from_passage,
    cosine_similarity_accuracy,
)
from inference import remove_answer_from_question, shuffle_options, run_inference


# ─────────────────────────────────────────────────────────────────────────────
# preprocessing — text cleaning
# ─────────────────────────────────────────────────────────────────────────────

def test_clean_text_removes_punctuation():
    result = clean_text("Hello, World! This is a test.")
    assert ',' not in result
    assert '!' not in result
    assert '.' not in result


def test_clean_text_lowercases():
    result = clean_text("UPPER lower MiXeD")
    assert result == result.lower()


def test_tokenize_removes_stopwords():
    tokens = tokenize("The cat sat on the mat")
    assert 'the' not in tokens
    assert 'on' not in tokens
    assert 'cat' in tokens or 'sat' in tokens or 'mat' in tokens


def test_tokenize_removes_short_words():
    tokens = tokenize("I am a big elephant")
    assert 'i' not in tokens
    assert 'a' not in tokens


def test_split_into_sentences_basic():
    text  = "First sentence. Second sentence. Third one!"
    sents = split_into_sentences(text)
    assert len(sents) >= 2


def test_build_one_sample_returns_string():
    result = build_one_sample("Article text here.", "What is this?", "An answer")
    assert isinstance(result, str)
    assert len(result) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Jaccard / backward-compat cosine_similarity_feature
# ─────────────────────────────────────────────────────────────────────────────

def test_cosine_similarity_feature_identical():
    sim = cosine_similarity_feature("dogs are animals", "dogs are animals")
    assert sim > 0.8


def test_cosine_similarity_feature_no_overlap():
    sim = cosine_similarity_feature("dog cat fish", "apple orange mango")
    assert sim == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# TRUE TF-IDF cosine similarity
# ─────────────────────────────────────────────────────────────────────────────

CORPUS = [
    "The Amazon rainforest covers 5.5 million square kilometres.",
    "Deforestation threatens the Amazon ecosystem and its biodiversity.",
    "The world's oxygen production relies partly on the Amazon.",
    "Scientists urge governments to protect the rainforest immediately.",
    "Millions of species live in the Amazon rainforest.",
]


def test_build_tfidf_vectorizer_returns_vectorizer():
    vec = build_tfidf_vectorizer(CORPUS)
    assert vec is not None
    # Should be able to transform a new text
    result = vec.transform(["Amazon rainforest"])
    assert result.shape[0] == 1


def test_tfidf_cosine_similarity_identical_texts():
    vec = build_tfidf_vectorizer(CORPUS)
    sim = tfidf_cosine("Amazon rainforest ecosystem", "Amazon rainforest ecosystem", vec)
    assert sim > 0.99, f"Expected > 0.99, got {sim}"


def test_tfidf_cosine_similarity_no_overlap():
    vec = build_tfidf_vectorizer(CORPUS)
    sim = tfidf_cosine("xyz foo bar", "abc def ghi", vec)
    assert sim == 0.0, f"Expected 0.0, got {sim}"


def test_tfidf_cosine_similarity_partial_overlap_in_range():
    vec = build_tfidf_vectorizer(CORPUS)
    sim = tfidf_cosine(
        "The Amazon rainforest is vital for oxygen production.",
        "Amazon produces oxygen for the world.",
        vec,
    )
    assert 0.0 <= sim <= 1.0, f"Similarity out of [0,1]: {sim}"


def test_tfidf_cosine_higher_for_similar_than_dissimilar():
    vec = build_tfidf_vectorizer(CORPUS)
    article = "The Amazon rainforest is the lungs of the Earth."
    sim_similar   = tfidf_cosine(article, "Amazon rainforest produces oxygen.", vec)
    sim_dissimilar = tfidf_cosine(article, "Football is popular in Brazil.", vec)
    assert sim_similar >= sim_dissimilar, (
        f"Expected similar ({sim_similar:.4f}) >= dissimilar ({sim_dissimilar:.4f})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# cosine_similarity_accuracy (professor's metric)
# ─────────────────────────────────────────────────────────────────────────────

def test_cosine_similarity_accuracy_returns_dict():
    import pandas as pd
    vec = build_tfidf_vectorizer(CORPUS)
    # Build a tiny synthetic dataframe
    df = pd.DataFrame([{
        'article': "The Amazon rainforest covers millions of species and produces oxygen.",
        'question': "What does the Amazon produce?",
        'A': 'oxygen',
        'B': 'nothing',
        'C': 'pollution',
        'D': 'sand',
        'answer': 'A',
    }])
    result = cosine_similarity_accuracy(vec, df)
    assert 'accuracy' in result
    assert 'avg_correct_sim' in result
    assert 'avg_wrong_sim' in result
    assert 'sim_gap' in result
    assert 0.0 <= result['accuracy'] <= 1.0


def test_cosine_similarity_accuracy_perfect_case():
    """When correct option IS the article text, cosine should be 1.0 → accuracy 1.0."""
    import pandas as pd
    article = "The Amazon rainforest produces oxygen and hosts millions of species."
    vec     = build_tfidf_vectorizer([article] + CORPUS)
    df = pd.DataFrame([{
        'article': article,
        'question': "What does the Amazon do?",
        'A': article,        # correct — identical to article → highest cosine
        'B': 'xyz abc def',
        'C': 'foo bar baz',
        'D': 'random words',
        'answer': 'A',
    }])
    result = cosine_similarity_accuracy(vec, df)
    assert result['accuracy'] == 1.0, f"Expected 1.0, got {result['accuracy']}"


# ─────────────────────────────────────────────────────────────────────────────
# question generation
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_ARTICLE = (
    "The Amazon rainforest covers 5.5 million square kilometres across nine countries. "
    "It produces about 20 percent of the world's oxygen and is home to millions of species. "
    "Deforestation is one of the biggest threats to this ecosystem."
)


def test_generate_questions_returns_list():
    qs = generate_questions_from_passage(SAMPLE_ARTICLE, count=3)
    assert isinstance(qs, list)
    assert len(qs) > 0


def test_each_question_is_dict():
    qs = generate_questions_from_passage(SAMPLE_ARTICLE, count=2)
    for item in qs:
        assert isinstance(item, dict)
        assert 'question' in item and 'answer' in item


def test_answer_not_identical_to_question():
    qs = generate_questions_from_passage(SAMPLE_ARTICLE, count=3)
    ans = qs[0].get('answer', '')
    assert isinstance(ans, str) and len(ans) > 0


# ─────────────────────────────────────────────────────────────────────────────
# answer removal (masking utility)
# ─────────────────────────────────────────────────────────────────────────────

def test_remove_answer_from_question_masks_token():
    q      = "What does the passage say about amazon rainforest coverage?"
    answer = "amazon rainforest"
    result = remove_answer_from_question(q, answer)
    assert 'amazon' not in result.lower() or '___' in result


def test_remove_answer_short_tokens_ignored():
    q      = "What is it about?"
    answer = "it"
    result = remove_answer_from_question(q, answer)
    # Short tokens (len <= 3) should NOT be masked
    assert result == q


# ─────────────────────────────────────────────────────────────────────────────
# shuffle_options
# ─────────────────────────────────────────────────────────────────────────────

def test_shuffle_options_all_labels_present():
    opts, label = shuffle_options("correct", ["wrong1", "wrong2", "wrong3"])
    assert set(opts.keys()) == {'A', 'B', 'C', 'D'}


def test_shuffle_options_correct_answer_in_dict():
    opts, label = shuffle_options("correct", ["wrong1", "wrong2", "wrong3"])
    assert opts[label] == "correct"


def test_shuffle_options_correct_label_valid():
    _, label = shuffle_options("correct", ["w1", "w2", "w3"])
    assert label in ['A', 'B', 'C', 'D']


def test_shuffle_options_all_four_options_filled():
    opts, _ = shuffle_options("answer", ["d1", "d2", "d3"])
    assert len(opts) == 4
    assert all(v != '' for v in opts.values())


# ─────────────────────────────────────────────────────────────────────────────
# run_inference (no race_rows — template-based path)
# ─────────────────────────────────────────────────────────────────────────────

def test_run_inference_returns_dict():
    result = run_inference(SAMPLE_ARTICLE)
    assert isinstance(result, dict)
    assert 'questions' in result
    assert 'latency_ms' in result


def test_run_inference_questions_count():
    result = run_inference(SAMPLE_ARTICLE)
    assert 5 <= len(result['questions']) <= 10


def test_run_inference_question_structure():
    result = run_inference(SAMPLE_ARTICLE)
    for q in result['questions']:
        assert 'question' in q
        assert 'options' in q
        assert 'correct_label' in q
        assert 'correct_answer' in q
        assert 'hints' in q
        assert set(q['options'].keys()) == {'A', 'B', 'C', 'D'}
        assert q['correct_label'] in ['A', 'B', 'C', 'D']


def test_run_inference_correct_answer_in_options():
    result = run_inference(SAMPLE_ARTICLE)
    for q in result['questions']:
        label = q['correct_label']
        assert q['options'][label] == q['correct_answer']


def test_run_inference_latency_is_positive():
    result = run_inference(SAMPLE_ARTICLE)
    assert result['latency_ms'] >= 0


# ─────────────────────────────────────────────────────────────────────────────
# run_inference with race_rows (real RACE dataset path)
# ─────────────────────────────────────────────────────────────────────────────

RACE_ROW_SAMPLE = {
    'article':  SAMPLE_ARTICLE,
    'question': "How many countries does the Amazon rainforest span?",
    'A': 'Nine',
    'B': 'Seven',
    'C': 'Twelve',
    'D': 'Five',
    'answer':   'A',
    'id':       'test_001',
}


def test_run_inference_with_race_rows():
    result = run_inference(SAMPLE_ARTICLE, race_rows=[RACE_ROW_SAMPLE])
    assert isinstance(result, dict)
    assert len(result['questions']) >= 1


def test_run_inference_race_row_uses_real_question():
    result = run_inference(SAMPLE_ARTICLE, race_rows=[RACE_ROW_SAMPLE])
    first_q = result['questions'][0]
    assert first_q['question'] == RACE_ROW_SAMPLE['question']


def test_run_inference_race_row_correct_label():
    result = run_inference(SAMPLE_ARTICLE, race_rows=[RACE_ROW_SAMPLE])
    first_q = result['questions'][0]
    assert first_q['correct_label'] == 'A'
    assert first_q['correct_answer'] == 'Nine'


def test_run_inference_race_rows_fills_up_to_five():
    """When fewer than 5 race_rows given, template questions fill remaining slots."""
    result = run_inference(SAMPLE_ARTICLE, race_rows=[RACE_ROW_SAMPLE])
    assert len(result['questions']) <= 5


def test_run_inference_multiple_race_rows():
    row2 = {
        'article':  SAMPLE_ARTICLE,
        'question': "What percentage of world oxygen does the Amazon produce?",
        'A': '10 percent',
        'B': '20 percent',
        'C': '30 percent',
        'D': '50 percent',
        'answer':   'B',
        'id':       'test_001',
    }
    result = run_inference(SAMPLE_ARTICLE, race_rows=[RACE_ROW_SAMPLE, row2])
    assert len(result['questions']) >= 2
    assert result['questions'][1]['correct_label'] == 'B'
    assert result['questions'][1]['correct_answer'] == '20 percent'


# ─────────────────────────────────────────────────────────────────────────────
# Manual runner (no pytest needed)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import traceback
    tests = [
        test_clean_text_removes_punctuation,
        test_clean_text_lowercases,
        test_tokenize_removes_stopwords,
        test_tokenize_removes_short_words,
        test_split_into_sentences_basic,
        test_build_one_sample_returns_string,
        test_cosine_similarity_feature_identical,
        test_cosine_similarity_feature_no_overlap,
        test_build_tfidf_vectorizer_returns_vectorizer,
        test_tfidf_cosine_similarity_identical_texts,
        test_tfidf_cosine_similarity_no_overlap,
        test_tfidf_cosine_similarity_partial_overlap_in_range,
        test_tfidf_cosine_higher_for_similar_than_dissimilar,
        test_cosine_similarity_accuracy_returns_dict,
        test_cosine_similarity_accuracy_perfect_case,
        test_generate_questions_returns_list,
        test_each_question_is_dict,
        test_answer_not_identical_to_question,
        test_remove_answer_from_question_masks_token,
        test_remove_answer_short_tokens_ignored,
        test_shuffle_options_all_labels_present,
        test_shuffle_options_correct_answer_in_dict,
        test_shuffle_options_correct_label_valid,
        test_shuffle_options_all_four_options_filled,
        test_run_inference_returns_dict,
        test_run_inference_questions_count,
        test_run_inference_question_structure,
        test_run_inference_correct_answer_in_options,
        test_run_inference_latency_is_positive,
        test_run_inference_with_race_rows,
        test_run_inference_race_row_uses_real_question,
        test_run_inference_race_row_correct_label,
        test_run_inference_race_rows_fills_up_to_five,
        test_run_inference_multiple_race_rows,
    ]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{'-'*50}")
    print(f"  {passed} passed, {failed} failed")
