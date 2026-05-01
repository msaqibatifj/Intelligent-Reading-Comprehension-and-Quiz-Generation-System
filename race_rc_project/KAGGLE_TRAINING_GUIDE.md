# Kaggle Training Guide: Model A & Model B

This guide explains how to train models on Kaggle and export them for local use.

## Why Train on Kaggle?

- ✓ Free GPU (T4 ~15GB VRAM or P100 ~16GB)
- ✓ RACE dataset pre-loaded
- ✓ No local compute cost
- ✓ Reproducible environment
- ✓ Easy collaboration

## Prerequisites

1. **Kaggle Account** — Free at [kaggle.com](https://kaggle.com)
2. **RACE Dataset** — Add to your Kaggle workspace: [RACE Kaggle](https://www.kaggle.com/datasets/fudan-gpt/race)
3. **Notebook Template** — Copy code from `src/model_a_train.py` and `src/model_b_train.py`

## Step-by-Step: Train Model A on Kaggle

### 1. Create a New Notebook

- Go to **Kaggle.com** → **Code** → **Create** → **New Notebook**
- Choose **Python** language
- Enable **GPU** (via "Accelerator" in sidebar)

### 2. Add Dataset Input

In notebook sidebar:
- Click **+ Data**
- Search for "RACE" → Add the RACE dataset
- This mounts to `/kaggle/input/race/` in your notebook

### 3. Copy Training Code

In the first cell, add the necessary imports and configuration:

```python
# Cell 1: Imports
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("✓ Imports successful")
```

### 4. Load Preprocessing Module

```python
# Cell 2: Preprocessing Functions (from src/preprocessing.py)
# Copy the FeatureEngineer class and utility functions here
# OR: If you've uploaded preprocessing.py to kaggle, import it
```

### 5. Load Dataset

```python
# Cell 3: Load RACE Dataset
data_path = '/kaggle/input/race/race-train.csv'  # Adjust filename
df = pd.read_csv(data_path)
print(f"Loaded {len(df)} records")
print(df.head())
print(f"Columns: {df.columns.tolist()}")
```

### 6. Run Training

```python
# Cell 4-N: Copy training functions from model_a_train.py
# Execute train_logistic_regression(), train_svm(), etc.
```

### 7. Export Models

```python
# Final Cell: Export Models
import os
os.makedirs('/kaggle/working/models/model_a/traditional/', exist_ok=True)

# Save all trained models
joblib.dump(lr_model, '/kaggle/working/models/model_a/traditional/lr_model.pkl')
joblib.dump(svm_model, '/kaggle/working/models/model_a/traditional/svm_model.pkl')
joblib.dump(rf_model, '/kaggle/working/models/model_a/traditional/rf_model.pkl')
joblib.dump(xgb_model, '/kaggle/working/models/model_a/traditional/xgb_model.pkl')
joblib.dump(ensemble_model, '/kaggle/working/models/model_a/traditional/ensemble_model.pkl')
joblib.dump(feature_engineer, '/kaggle/working/models/model_a/traditional/feature_engineer.pkl')

print("✓ All models saved to /kaggle/working/")
```

### 8. Download Models

- Click **Output** (bottom right)
- Download the `.zip` file containing all models
- Extract to `models/model_a/traditional/` in your local project

## Step-by-Step: Train Model B on Kaggle

Repeat the same process for Model B:

1. Create new notebook
2. Add RACE dataset
3. Copy code from `src/model_b_train.py`
4. Run training cells
5. Export distractor ranker, hint extractor, hint scorer
6. Download and extract

## RACE Dataset Format

The CSV typically has columns:
```
article | question | options | answer | level | source
```

Where:
- `article`: The reading passage (text)
- `question`: The MCQ question (text)
- `options`: Options separated by `|` (e.g., "Option A|Option B|Option C|Option D")
- `answer`: Correct answer (A, B, C, or D)
- `level`: Difficulty level (middle, high)
- `source`: Data source (RACE, etc.)

## Troubleshooting

### "Module not found" errors
- Copy the preprocessing.py functions directly into the notebook
- Don't try to import from local directories; Kaggle uses `/kaggle/` paths

### GPU not enabled
- In notebook settings, enable "Accelerator" → "GPU"
- Check that VRAM is sufficient (12GB+ recommended)

### Memory errors during training
- Reduce `max_features` in CONFIG
- Use `batch_size` for large datasets
- Convert sparse matrices efficiently

### Slow training
- Use GPU (not CPU)
- Reduce training set size for rapid prototyping
- Monitor `/proc/meminfo` and GPU usage

## Advanced: Using Kaggle Secrets

To push trained models to a database:

1. Store API key in **Kaggle Secrets** (notebook settings)
2. Access via:
```python
import kaggle
api_key = '...'  # Load from secrets
```

## Export Checklist

Before downloading, ensure you've saved:
- [ ] `lr_model.pkl`
- [ ] `svm_model.pkl`
- [ ] `nb_model.pkl`
- [ ] `rf_model.pkl`
- [ ] `xgb_model.pkl`
- [ ] `ensemble_voting_model.pkl`
- [ ] `ensemble_stacking_model.pkl`
- [ ] `kmeans_model.pkl`
- [ ] `label_propagation_model.pkl`
- [ ] `gmm_model.pkl`
- [ ] `feature_engineer.pkl`

For Model B:
- [ ] `distractor_ranker.pkl`
- [ ] `distractor_ranker_lr.pkl`
- [ ] `distractor_ranker_rf.pkl`
- [ ] `hint_extractor.pkl`
- [ ] `hint_scorer.pkl`
- [ ] `word2vec_model.pkl`
- [ ] `feature_engineer.pkl`

## Next Steps

1. Download all exported models
2. Place in `models/model_a/traditional/` and `models/model_b/traditional/`
3. Run Streamlit app locally:
   ```bash
   streamlit run ui/app.py
   ```
4. Test inference end-to-end

## Sample Kaggle Notebook URL

After creating and running a notebook, share it:
- Notebook → **Share** → Make public
- Copy URL for report/documentation

---

**Tips for Success:**
- Start with small training sets to debug quickly
- Use Kaggle's built-in data exploration to understand RACE format
- Save model versions (e.g., `rf_model_v1.pkl`, `rf_model_v2.pkl`)
- Document any preprocessing changes in the notebook

