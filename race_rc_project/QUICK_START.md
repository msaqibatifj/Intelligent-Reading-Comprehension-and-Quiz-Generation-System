# Quick Start Guide

Get your Intelligent Reading Comprehension system running in 5 minutes.

## Prerequisites

✓ Python 3.9+  
✓ pip / venv  
✓ Kaggle account (for training models)  

## Installation (2 minutes)

```bash
cd race_rc_project
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run Streamlit App (Local Demo)

```bash
cd ui
streamlit run app.py
```

Opens at `http://localhost:8501`

**Features:**
- Article Input screen: Paste text or load sample
- Quiz View: Interactive MCQ answering
- Hint Panel: Graduated hint revelation
- Analytics Dashboard: Model metrics & results

## Train Models on Kaggle (5-10 minutes)

### Step 1: Prepare Dataset
- Go to [Kaggle RACE Dataset](https://www.kaggle.com/datasets/fudan-gpt/race)
- Add to your workspace

### Step 2: Create Notebook
- New → Python Notebook
- Add input: `/kaggle/input/race/` (auto-mounts)
- Enable GPU (Settings → Accelerator)

### Step 3: Copy & Run Training Code

**For Model A:**
```python
# Copy entire src/model_a_train.py into notebook cells
# Adjust CONFIG['data_path'] = '/kaggle/input/race/race-train.csv'
python src/model_a_train.py
```

**For Model B:**
```python
# Same for src/model_b_train.py
python src/model_b_train.py
```

### Step 4: Export Models
```python
# Models auto-save to /kaggle/working/
# Download the .zip file
```

### Step 5: Extract Locally
```bash
# Extract downloaded zip to:
# models/model_a/traditional/   (all Model A .pkl files)
# models/model_b/traditional/   (all Model B .pkl files)
```

## Project Structure Overview

```
race_rc_project/
│
├── data/
│   ├── raw/          ← Place RACE CSV files here
│   └── processed/    ← Auto-generated feature matrices
│
├── models/
│   ├── model_a/traditional/  ← Trained Model A (.pkl files)
│   └── model_b/traditional/  ← Trained Model B (.pkl files)
│
├── src/
│   ├── preprocessing.py   (Feature engineering)
│   ├── inference.py       (Unified API)
│   ├── evaluate.py        (Metrics)
│   ├── model_a_train.py   (Training script)
│   └── model_b_train.py   (Training script)
│
├── ui/
│   └── app.py            (Streamlit app)
│
├── notebooks/
│   ├── EDA.ipynb         (Exploratory analysis)
│   └── experiments.ipynb (Experiment tracking)
│
├── tests/
│   └── test_inference.py (Unit tests)
│
├── README.md
├── KAGGLE_TRAINING_GUIDE.md
└── requirements.txt
```

## Key Commands Reference

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
cd ui && streamlit run app.py

# Run tests (requires pytest)
pytest tests/test_inference.py -v

# Train locally (requires RACE CSV in data/raw/)
python src/model_a_train.py
python src/model_b_train.py
```

## What's Inside

### Model A: Q&A Verification
- Logistic Regression, SVM, Naive Bayes
- Random Forest, XGBoost
- K-Means Clustering (unsupervised)
- Label Propagation (semi-supervised)
- Gaussian Mixture Models
- Ensemble: Soft Voting + Stacking

### Model B: Distractor & Hint Generation
- Distractor Ranker (Logistic Regression + Random Forest)
- Hint Extractor (binary classification)
- Hint Scorer (regression)
- Word2Vec embeddings for semantic similarity

### Feature Engineering
- One-Hot Encoding
- TF-IDF Vectorization
- Cosine Similarity
- Handcrafted Lexical Features (word overlap, position, length, frequency)

### UI Features
- 4 interactive screens
- Real-time inference
- Hint progression system
- Analytics dashboard with CSV export
- Responsive design

## Common Issues & Fixes

### "ModuleNotFoundError: No module named 'streamlit'"
```bash
pip install -r requirements.txt
```

### Kaggle GPU not working
- In notebook settings, enable "Accelerator" → "GPU"
- Check remaining GPU quota in account

### Memory error during training
- Reduce `max_features` in CONFIG
- Use smaller training subset for rapid prototyping

### Models not loading in Streamlit
- Ensure `.pkl` files are in `models/model_a/traditional/` and `models/model_b/traditional/`
- Check file paths in `ui/app.py`

## Next Steps

1. **Download RACE dataset** (or use mock data initially)
2. **Train on Kaggle** using KAGGLE_TRAINING_GUIDE.md
3. **Download and extract models** locally
4. **Run Streamlit app** and test end-to-end
5. **Generate final report** using EDA notebook
6. **Push to GitHub** with clean commit history

## Files to Focus On (In Order)

1. **README.md** — Comprehensive documentation
2. **KAGGLE_TRAINING_GUIDE.md** — Step-by-step Kaggle workflow
3. **src/preprocessing.py** — Understand feature engineering
4. **src/model_a_train.py** — Supervised + unsupervised models
5. **src/model_b_train.py** — Distractor + hint models
6. **ui/app.py** — Streamlit UI implementation
7. **src/inference.py** — Inference API for production

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

## Support & Documentation

- **Main README**: Complete setup, usage, and API docs
- **Kaggle Guide**: Step-by-step training instructions
- **Inline Comments**: Every `.py` file is well-commented
- **Test File**: `tests/test_inference.py` shows API usage examples

## Example Usage (Python API)

```python
from src.inference import UnifiedInference

# Initialize
inference = UnifiedInference(
    model_a_paths={'ensemble': 'models/model_a/traditional/ensemble_model.pkl'},
    model_b_paths={'distractor_ranker': 'models/model_b/traditional/distractor_ranker.pkl'}
)

# Generate MCQ
passage = "The Earth is the third planet from the Sun..."
mcq = inference.generate_and_verify_mcq(passage)
print(mcq['question'])      # Generated question
print(mcq['options'])       # 4 options (1 correct + 3 distractors)
print(mcq['hints'])         # 3 graduated hints

# Verify answer
result = inference.verify_user_answer(
    passage, 
    mcq['question'], 
    mcq['correct_answer'],
    "The Sun"  # User's answer
)
print(f"Correct: {result['is_correct']}, Confidence: {result['confidence']:.2%}")
```

---

**Ready?** Start with:
```bash
cd race_rc_project/ui
streamlit run app.py
```

Then follow KAGGLE_TRAINING_GUIDE.md to train your models!
