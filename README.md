# Intelligent Reading Comprehension and Quiz Generation System

A machine learning system that automatically generates multiple-choice quiz questions from reading passages, verifies answers, creates plausible distractors, and provides graduated hints.

**Course:** Artificial Intelligence — BS (CS) Spring 2026  
**Institution:** NUCES FAST, Islamabad Campus

---

## Project Overview

An AI-powered system that:

1. Accepts a reading passage as input
2. Generates a multiple-choice question (template-based + ML-ranked)
3. Produces 4 options (1 correct + 3 ML-generated distractors)
4. Verifies user answers (Logistic Regression + SVM ensemble)
5. Provides graduated hints (extractive + ML-scored)
6. Displays analytics via a Streamlit dashboard

---

## Features

### Model A: Question & Answer Generator/Verifier

- **Question Generation**: Generate meaningful MCQ questions from passages using Wh-word templates
- **Answer Verification**: Use supervised models (Logistic Regression, SVM, Naive Bayes, Random Forest, XGBoost) to verify answer correctness
- **Ensemble Approach**: Combine multiple classifiers using soft/hard voting and stacking for robust predictions
- **Unsupervised Learning**: K-Means clustering, Label Propagation, and Gaussian Mixture Models for question grouping and semi-supervised learning

### Model B: Distractor & Hint Generator

- **Distractor Generation**: Create 3 plausible but definitively wrong answer options
- **Hint Extraction**: Provide graduated hints (vague → specific) without revealing the answer
- **Ranking Models**: Use Logistic Regression and Random Forest to score distractors and hints

### Streamlit UI (4 Screens)

1. **Article Input**: Paste or upload passages, load RACE samples
2. **Quiz View**: Interactive MCQ interface with instant feedback
3. **Hint Panel**: Progressive hint revelation system
4. **Analytics Dashboard**: Model performance metrics, session results, CSV export

---

## Project Structure

```
race_rc_project/
├── data/
│   ├── raw/                  # Original RACE dataset CSVs
│   └── processed/            # Feature-engineered datasets
├── models/
│   ├── model_a/
│   │   └── traditional/      # Pickled sklearn models
│   └── model_b/
│       └── traditional/      # Pickled sklearn/Word2Vec models
├── src/
│   ├── preprocessing.py      # Data loading & feature engineering
│   ├── model_a_train.py      # Model A training script
│   ├── model_b_train.py      # Model B training script
│   ├── inference.py          # Unified inference API
│   └── evaluate.py           # Evaluation metrics
├── ui/
│   └── app.py                # Streamlit application
├── notebooks/
│   ├── EDA.ipynb             # Exploratory Data Analysis
│   └── experiments.ipynb     # Experiment tracking
├── tests/
│   └── test_inference.py     # Unit tests
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.9+
- Virtual environment (venv)
- NVIDIA GPU (RTX 3060 12GB recommended)

### Setup

1. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download RACE dataset**
   - Download from [Kaggle RACE Dataset](https://www.kaggle.com/datasets/ankitdhiman7/race-dataset)
   - Place CSV files in `data/raw/`

---

## Quick Start

### 1. Run Preprocessing
```bash
cd src
python preprocessing.py
```

Optional: maximize cosine-similarity features:
```bash
TFIDF_MAX_FEATURES=50000 TFIDF_MIN_DF=1 TFIDF_NGRAM_MAX=2 python preprocessing.py
```

### 2. Train Model A
```bash
python model_a_train.py
```

### 3. Train Model B
```bash
python model_b_train.py
```

### 4. Evaluate
```bash
python evaluate.py
```

### 5. Launch Streamlit UI
```bash
cd ..
streamlit run ui/app.py
```

> **Note:** The UI works in demo mode even without trained models, using rule-based inference.

---

## Models Trained

### Model A: Answer Verification

| Model | Task | Approach |
|-------|------|----------|
| Logistic Regression | Fast baseline for Q&A verification | Linear classification |
| SVM | Non-linear classification | RBF kernel for answer ranking |
| Soft Voting Ensemble | Combines LR + SVM | 50/50 weighted predictions |
| K-Means Clustering | Unsupervised grouping | Question similarity clustering |

### Model B: Distractor & Hint Generation

| Approach | Description |
|----------|-------------|
| One-Hot Encoding + Cosine Similarity | Compare semantic overlap |
| Word2Vec Nearest Neighbors | Find semantically similar phrases |
| Frequency-Based Substitution | Rank by passage co-occurrence |
| Logistic Regression Ranker | Trained model to score distractors |
| Extractive Hints | Cosine similarity between passage sentences and question |
| ML-Scored Hints | Logistic Regression on sentence features |

---

## Feature Engineering

### Primary Features
- **One-Hot Encoding**: Binary vocabulary matrix (passage + question + options)
- **TF-IDF Vectorization**: Term frequency-inverse document frequency
- **Cosine Similarity Matrix**: Semantic similarity between sentences

### Handcrafted Lexical Features
- Word overlap count (question ↔ option)
- Sentence length
- Position in passage
- Character-level match score
- Passage frequency of candidate terms

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **BLEU** | N-gram overlap between generated and reference text |
| **ROUGE-1 / ROUGE-2 / ROUGE-L** | Overlap quality at unigram, bigram, and sequence level |
| **METEOR** | Token, stem, and synonym-aware text similarity |

---

## Dataset

**RACE** (ReAding Comprehension from Examinations) — Lai et al., EMNLP 2017
- ~28,000 passages
- ~100,000 questions
- Source: Chinese school English exams

---

## Python API (Inference)

```python
from src.inference import UnifiedInference

# Initialize with trained models
model_a_paths = {
    'lr': 'models/model_a/traditional/lr_model.pkl',
    'svm': 'models/model_a/traditional/svm_model.pkl',
    'ensemble': 'models/model_a/traditional/ensemble_model.pkl',
}
model_b_paths = {
    'distractor_ranker': 'models/model_b/traditional/distractor_ranker.pkl',
}

inference = UnifiedInference(model_a_paths, model_b_paths)

# Generate complete MCQ
passage = "Your reading passage here..."
mcq = inference.generate_and_verify_mcq(passage)
print(mcq)
# Output: {'question': '...', 'correct_answer': '...', 'distractors': [...], 'hints': [...]}

# Verify user answer
result = inference.verify_user_answer(passage, mcq['question'], 
                                      mcq['correct_answer'], user_answer)
```

---

## Technical Constraints

- **GPU**: NVIDIA RTX 3060 12GB or Google Colab T4
- **Inference Time**: <10 seconds per MCQ generation
- **Memory**: Never convert sparse matrices to dense on full RACE data
- **Data Leakage**: Always `fit_transform()` on training set, `transform()` on val/test

---

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| scikit-learn | 1.3.0 | ML models, metrics, pipelines |
| XGBoost | 2.0.0 | Gradient boosting classifier |
| pandas | 2.0.3 | Data manipulation |
| numpy | 1.24.3 | Numerical computing |
| gensim | 4.3.1 | Word2Vec embeddings |
| sentence-transformers | 2.2.2 | Semantic similarity |
| streamlit | 1.28.0 | Web UI |
| joblib | 1.3.2 | Model serialization |
| nltk | 3.8.1 | Text processing |

---

## Testing

```bash
pytest tests/test_inference.py -v
```

---

## Ethical Considerations

### Bias
- **Issue**: RACE passages from Chinese school exams; cultural & linguistic biases
- **Mitigation**: Include non-English datasets, analyze performance across demographics

### Academic Integrity
- Generated questions **must not** be used in real exams without human review
- UI disclaimer: "AI-generated content; human review recommended"

### Model Transparency
- UI indicates which answers are AI-generated
- Confidence scores displayed

---

## Grading Breakdown (100 marks)

| Component | Marks |
|-----------|-------|
| EDA & Preprocessing | 10 |
| Model A — Traditional ML | 15 |
| Model A — Unsupervised/Semi-Supervised | 20 |
| Model A — Ensemble | 5 |
| Model B — Distractor Gen | 15 |
| Model B — Hint Gen | 10 |
| Streamlit UI | 15 |
| Final Report | 5 |
| Code Quality | 5 |
| **Total** | **100** |

---

## Deliverables Checklist

- [ ] GitHub repository with clean commit history
- [ ] `requirements.txt` with pinned versions
- [ ] `README.md` with setup & training instructions
- [ ] EDA notebook on Kaggle
- [ ] Trained model checkpoints (Model A, Model B)
- [ ] Final report PDF (10+ pages)
- [ ] Streamlit UI running end-to-end
- [ ] 10-minute demo video or live session

---

## Future Enhancements

- Abstractive question generation (seq2seq models)
- Multi-modal questions (images + text)
- Difficulty level control
- Domain-specific models (medical, legal exams)
- Real-time model retraining on user feedback

---

## License

MIT License

---

**Last Updated:** May 2024  
**Version:** 1.0.0  
**Status:** Active Development
