# Intelligent Reading Comprehension and Quiz Generation System

A machine learning system that automatically generates multiple-choice quiz questions from reading passages, verifies answers, creates plausible distractors, and provides graduated hints.

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
│   ├── EDA.ipynb             # Exploratory Data Analysis (Kaggle)
│   └── experiments.ipynb     # Experiment tracking (Kaggle)
├── tests/
│   └── test_inference.py     # Unit tests
├── requirements.txt
├── README.md
└── report/
    └── final_report.pdf
```

## Installation

### Prerequisites
- Python 3.9+
- Virtual environment (venv)
- NVIDIA GPU (RTX 3060 12GB recommended)

### Setup

1. **Clone repository** (or create new directory)
   ```bash
   cd race_rc_project
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download RACE dataset**
   - Download from [RACE Kaggle Dataset](https://www.kaggle.com/datasets/fudan-gpt/race)
   - Place CSV files in `data/raw/`

## Quick Start

### 0. Preprocess Data (Optional)
```bash
# Prepare Q&A and distractor datasets from raw RACE CSV
python scripts/preprocess_data.py --input data/raw/train.csv --output data/processed/

# Or test with a smaller subset
python scripts/preprocess_data.py --input data/raw/train.csv --output data/processed/ --sample 1000
```

This creates:
- `data/processed/train_qa.csv` (Q&A pairs for Model A)
- `data/processed/train_distractors.csv` (distractors for Model B)

### 2. Run Streamlit App (UI only)
```bash
cd ui
streamlit run app.py
```
This opens the interactive UI at `http://localhost:8501`

### 3. Train Models (Kaggle Notebook)
Upload `notebooks/model_a_train.ipynb` to Kaggle Notebooks:
- Load and preprocess RACE data
- Train Logistic Regression, SVM, Random Forest, XGBoost
- Implement K-Means, Label Propagation, GMM (unsupervised)
- Ensemble voting and stacking
- Export models to `models/model_a/traditional/`
- Save `model_a_checkpoint.pkl` periodically in `models/model_a/traditional/` so training can resume

Do the same for `notebooks/model_b_train.ipynb` for distractor and hint models, and keep `model_b_checkpoint.pkl` for resume.

### 4. Local Training (Optional)
```bash
python src/model_a_train.py --data data/raw/train.csv --output models/model_a/traditional/
python src/model_b_train.py --data data/raw/train.csv --output models/model_b/traditional/
```

## Usage

### Basic Workflow
1. Navigate to "Article Input" screen
2. Paste a reading passage or load a sample
3. Click "Generate Quiz Question"
4. Go to "Quiz View" to answer the question
5. Check hint progression in "Hint Panel"
6. View analytics in "Analytics Dashboard"

### Python API (Inference)
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

## Training on Kaggle

### Why Kaggle?
- Free GPU (T4 or P100)
- Preloaded RACE dataset
- Easy environment setup
- Reproducible notebooks

### Steps

1. **Upload RACE Dataset**
   - Go to [Kaggle Datasets](https://www.kaggle.com/datasets/fudan-gpt/race)
   - Add to your workspace

2. **Create Notebook**
   - New Python Notebook
   - Add input: `/kaggle/input/race/` (dataset path)
   - Add output: `/kaggle/working/` (model export path)

3. **Copy Training Code**
   - See `notebooks/model_a_train.ipynb` template
   - Execute cells sequentially
   - Export trained models

4. **Download Models**
   - Models saved to `/kaggle/working/`
   - Download as `.zip`
   - Extract to `models/` directory locally

## Models Trained

Model A trains the following models:

### Supervised
- **Logistic Regression**: Fast baseline for Q&A verification
- **Support Vector Machine (SVM)**: Non-linear classification for answer ranking

### Ensemble
- **Soft Voting (LR + SVM)**: Combines LR and SVM predictions with 50/50 weighting

### Unsupervised
- **K-Means Clustering**: Unsupervised grouping of Q&A pairs by similarity

All models use text generation evaluation (BLEU, ROUGE, METEOR).

### Distractor Generation (Model B)
- **One-Hot Encoding + Cosine Similarity**: Compare semantic overlap
- **Word2Vec Nearest Neighbors**: Find semantically similar phrases
- **Frequency-Based Substitution**: Rank by passage co-occurrence
- **Logistic Regression Ranker**: Trained model to score distractors

### Hint Extraction (Model B)
- **Extractive**: Cosine similarity between passage sentences and question
- **ML-Scored**: Logistic Regression on sentence features (keyword overlap, position, length)

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

## Evaluation Metrics

### Model A and Model B
| Metric | Description |
|--------|-------------|
| **BLEU** | N-gram overlap between generated and reference text |
| **ROUGE-1 / ROUGE-2 / ROUGE-L** | Overlap quality at unigram, bigram, and sequence level |
| **METEOR** | Token, stem, and synonym-aware text similarity |

For this project, final evaluation/reporting uses BLEU, ROUGE, and METEOR only.

## Checkpointing During Kaggle Training

To handle session disconnects and compute limits, both training scripts persist progress checkpoints:

- `models/model_a/traditional/model_a_checkpoint.pkl`
- `models/model_b/traditional/model_b_checkpoint.pkl`

If training stops, rerun the same script and it resumes from the last completed stage.

## GPU + Numba Acceleration

Training code is CUDA-aware for Kaggle GPU sessions:

- Model B distractor semantic scoring uses Numba CUDA kernels (with automatic CPU fallback).
- Model A XGBoost uses GPU when CUDA is available (`device='cuda'`).
- CUDA availability is checked at runtime using Numba.

## Technical Constraints

- **GPU**: NVIDIA RTX 3060 12GB or Google Colab T4
- **Inference Time**: <10 seconds per MCQ generation
- **Memory**: Never convert sparse matrices to dense on full RACE data
- **Data Leakage**: Always `fit_transform()` on training set, `transform()` on val/test

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

## Testing

Run unit tests:
```bash
pytest tests/test_inference.py -v
```

## Ethical Considerations

### Bias
- **Issue**: RACE passages from Chinese school exams; cultural & linguistic biases
- **Mitigation**: Include non-English datasets, analyze performance across demographics
- **Disclosure**: Report generalization limits in final report

### Accessibility
- UI supports keyboard navigation
- Color contrast ratios meet WCAG AA standards
- Alt text for images; semantic HTML

### Academic Integrity
- Generated questions **must not** be used in real exams without human review
- UI disclaimer: "AI-generated content; human review recommended"

### Model Transparency
- UI indicates which answers are AI-generated
- Confidence scores displayed
- Error analysis included in report

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

## Deliverables Checklist

- [ ] GitHub repository with clean commit history
- [ ] `requirements.txt` with pinned versions
- [ ] `README.md` with setup & training instructions
- [ ] EDA notebook on Kaggle
- [ ] Trained model checkpoints (Model A, Model B)
- [ ] Final report PDF (10+ pages)
- [ ] Streamlit UI running end-to-end
- [ ] 10-minute demo video or live session
- [ ] Human evaluation forms (sample)

## Report Structure

1. **Abstract** (200 words max)
2. **Introduction & Motivation**
3. **Related Work** (5+ papers cited)
4. **Dataset Analysis** (RACE: size, distribution, biases)
5. **Model A: Design, Training, Results**
6. **Model B: Design, Training, Results**
7. **User Interface Description**
8. **Evaluation & Discussion**
9. **Limitations & Future Work**
10. **Conclusion**
11. **References**

## Future Enhancements

- Abstractive question generation (seq2seq models)
- Multi-modal questions (images + text)
- Difficulty level control
- Domain-specific models (medical, legal exams)
- Real-time model retraining on user feedback
- Mobile app version

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m 'Add your feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details

## Support

- **Issues**: File a GitHub issue with reproduction steps
- **Kaggle Notebook**: Follow training guide in `notebooks/model_a_train.ipynb`
- **Documentation**: See inline code comments and docstrings

## Contact

For questions or feedback:
- **Email**: [your-email@example.com]
- **GitHub**: [your-github-username]

---

**Last Updated:** May 2024  
**Version:** 1.0.0  
**Status:** Active Development
