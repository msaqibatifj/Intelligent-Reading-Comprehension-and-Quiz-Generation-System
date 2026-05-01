# Kaggle Notebook: Model A Training (Copy-Paste Ready)
# 
# Instructions:
# 1. Create new Kaggle notebook (Python)
# 2. Add RACE dataset to workspace
# 3. Enable GPU (Settings → Accelerator)
# 4. Copy each cell below into separate Kaggle cells
# 5. Run them in order
# 6. Download /kaggle/working/ when done

# ============================================================================
# CELL 1: Imports
# ============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
from tqdm import tqdm

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.cluster import KMeans
from sklearn.semi_supervised import LabelPropagation
from sklearn.mixture import GaussianMixture
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import csr_matrix, hstack

# NLP imports
import nltk
from nltk.tokenize import sent_tokenize

warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

print("✓ All imports successful")


# ============================================================================
# CELL 2: Preprocessing Functions (from src/preprocessing.py)
# ============================================================================

class FeatureEngineer:
    """Feature engineering for Q&A and distractor generation."""
    
    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.tfidf_vectorizer = None
        self.onehot_vectorizer = None
        self.scaler = StandardScaler()
        
    def compute_word_overlap(self, text1, text2):
        """Compute word overlap between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if len(words1) == 0 or len(words2) == 0:
            return 0.0
        overlap = len(words1 & words2)
        return overlap / max(len(words1), len(words2))
    
    def compute_char_match_score(self, text1, text2):
        """Compute character-level similarity."""
        i = 0
        while i < len(text1) and i < len(text2) and text1[i] == text2[i]:
            i += 1
        return i / max(len(text1), len(text2), 1)
    
    def extract_lexical_features(self, question, options, passage):
        """Extract handcrafted lexical features for each option."""
        features_list = []
        
        for option in options:
            word_overlap = self.compute_word_overlap(question, option)
            char_match = self.compute_char_match_score(question, option)
            option_length = len(option.split())
            passage_freq = self.compute_passage_frequency(option, passage)
            
            features = [word_overlap, char_match, option_length, passage_freq]
            features_list.append(features)
        
        return np.array(features_list)
    
    def compute_passage_frequency(self, word, passage):
        """Compute how often a word appears in passage."""
        words = passage.lower().split()
        if len(words) == 0:
            return 0.0
        return words.count(word.lower()) / len(words)
    
    def fit_onehot(self, texts):
        """Fit One-Hot (CountVectorizer binary) on texts."""
        self.onehot_vectorizer = CountVectorizer(
            max_features=self.max_features,
            lowercase=True,
            binary=True,
            stop_words='english'
        )
        self.onehot_vectorizer.fit(texts)
        return self
    
    def transform_onehot(self, texts):
        """Transform texts using fitted One-Hot vectorizer."""
        if self.onehot_vectorizer is None:
            raise ValueError("One-Hot vectorizer not fitted.")
        return self.onehot_vectorizer.transform(texts)
    
    def fit_tfidf(self, texts):
        """Fit TF-IDF vectorizer."""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            lowercase=True,
            stop_words='english'
        )
        self.tfidf_vectorizer.fit(texts)
        return self
    
    def transform_tfidf(self, texts):
        """Transform texts using fitted TF-IDF."""
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted.")
        return self.tfidf_vectorizer.transform(texts)
    
    def save(self, path):
        """Save fitted vectorizers."""
        joblib.dump({
            'tfidf': self.tfidf_vectorizer,
            'onehot': self.onehot_vectorizer,
            'scaler': self.scaler,
            'max_features': self.max_features
        }, path)
        print(f"✓ Saved feature engineer to {path}")
    
    @staticmethod
    def load(path):
        """Load saved vectorizers."""
        data = joblib.load(path)
        fe = FeatureEngineer(max_features=data['max_features'])
        fe.tfidf_vectorizer = data['tfidf']
        fe.onehot_vectorizer = data['onehot']
        fe.scaler = data['scaler']
        return fe


def prepare_qa_dataset(df):
    """Prepare question-answer dataset for Model A."""
    records = []
    errors = {'skip_count': 0, 'reasons': {}}
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preparing QA data"):
        try:
            passage = str(row.get('article', ''))
            question = str(row.get('question', ''))
            
            # Handle options parsing (pipe-separated from A, B, C, D columns)
            options_raw = row.get('options')
            
            if pd.isna(options_raw):
                reason = 'options_nan'
                errors['reasons'][reason] = errors['reasons'].get(reason, 0) + 1
                errors['skip_count'] += 1
                continue
            
            if isinstance(options_raw, str):
                if '|' in options_raw:
                    options = options_raw.split('|')
                else:
                    options = [options_raw]
            elif isinstance(options_raw, list):
                options = options_raw
            else:
                reason = f'options_type_{type(options_raw)}'
                errors['reasons'][reason] = errors['reasons'].get(reason, 0) + 1
                errors['skip_count'] += 1
                continue
            
            # Clean options
            options = [str(opt).strip() for opt in options if opt]
            
            if len(options) < 2:
                reason = f'few_options_{len(options)}'
                errors['reasons'][reason] = errors['reasons'].get(reason, 0) + 1
                errors['skip_count'] += 1
                continue
            
            # Get correct answer
            answer = str(row.get('answer', '')).strip()
            
            if not answer:
                reason = 'empty_answer'
                errors['reasons'][reason] = errors['reasons'].get(reason, 0) + 1
                errors['skip_count'] += 1
                continue
            
            # Try to map answer to index
            if answer in ['A', 'B', 'C', 'D']:
                answer_idx = ord(answer) - ord('A')
            elif answer in ['0', '1', '2', '3']:
                answer_idx = int(answer)
            else:
                reason = f'unmapped_answer_{answer}'
                errors['reasons'][reason] = errors['reasons'].get(reason, 0) + 1
                errors['skip_count'] += 1
                continue
            
            if answer_idx >= len(options) or answer_idx < 0:
                reason = f'answer_idx_oob_{answer_idx}_{len(options)}'
                errors['reasons'][reason] = errors['reasons'].get(reason, 0) + 1
                errors['skip_count'] += 1
                continue
            
            correct_answer = options[answer_idx]
            
            # Create training records
            for opt_idx, option in enumerate(options):
                label = 1 if opt_idx == answer_idx else 0
                records.append({
                    'article': passage,
                    'question': question,
                    'option': option,
                    'label': label,
                    'answer': correct_answer,
                    'question_id': idx
                })
        except Exception as e:
            reason = f'exception_{str(e)[:50]}'
            errors['reasons'][reason] = errors['reasons'].get(reason, 0) + 1
            errors['skip_count'] += 1
            continue
    
    print(f"\n[DEBUG] Record creation details:")
    print(f"  Created: {len(records)} records")
    print(f"  Skipped: {errors['skip_count']} records")
    if errors['reasons']:
        print(f"  Skip reasons:")
        for reason, count in sorted(errors['reasons'].items(), key=lambda x: x[1], reverse=True):
            print(f"    - {reason}: {count}")
    
    return pd.DataFrame(records)


def build_feature_matrix_model_a(qa_df, feature_engineer, fit=False):
    """Build combined feature matrix (sparse + dense) for Model A."""
    combined_texts = qa_df['question'].astype(str) + ' ' + qa_df['option'].astype(str)
    
    if fit:
        feature_engineer.fit_onehot(combined_texts.tolist())
    
    # One-Hot features (sparse)
    onehot_features = feature_engineer.transform_onehot(combined_texts.tolist())
    
    # Lexical features (dense)
    lexical_list = []
    for _, row in tqdm(qa_df.iterrows(), total=len(qa_df), desc="Computing lexical features"):
        lexical = feature_engineer.extract_lexical_features(
            str(row['question']),
            [str(row['option'])],
            str(row['article'])
        )
        lexical_list.append(lexical[0])
    
    lexical_features = np.array(lexical_list)
    
    # Combine sparse + dense
    lexical_sparse = csr_matrix(lexical_features)
    X = hstack([onehot_features, lexical_sparse])
    y = qa_df['label'].values
    
    return X, y


print("✓ Preprocessing functions loaded")


# ============================================================================
# CELL 3: Check Dataset
# ============================================================================

# List files in RACE dataset
import os
race_path = '/kaggle/input/datasets/ankitdhiman7/race-dataset/'
files = os.listdir(race_path)
print("Files in RACE dataset:")
for f in files:
    print(f"  - {f}")

# CSV files available: train.csv, test.csv, dev.csv
csv_files = [f for f in files if f.endswith('.csv')]
print(f"\nCSV files found: {csv_files}")

# Use train.csv for training
data_file = 'train.csv'
data_path = f'{race_path}{data_file}'

print(f"Loading from: {data_path}")


# ============================================================================
# CELL 4: Load Dataset & Detect Column Names
# ============================================================================

# Load the dataset
df = pd.read_csv(data_path)
print(f"✓ Loaded {len(df)} records")
print(f"\nDataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst row:")
print(df.iloc[0])

# Check for missing values
print(f"\nMissing values:\n{df.isnull().sum()}")

# Check if options are in separate columns (A, B, C, D format)
if all(col in df.columns for col in ['A', 'B', 'C', 'D']):
    print("✓ Detected options in separate columns (A, B, C, D)")
    # Combine them into a single 'options' column
    df['options'] = df[['A', 'B', 'C', 'D']].apply(lambda x: '|'.join(x.astype(str)), axis=1)
    print("✓ Created 'options' column from A, B, C, D")
    print(f"  Sample options column (first 3):")
    print(df['options'].head(3).tolist())
    print(f"  NaN count in options: {df['options'].isna().sum()}")
elif 'options' in df.columns:
    print("✓ Options already in single column")
else:
    print("⚠ Could not find options columns")

# Verify we have the required columns
required = ['article', 'question', 'options', 'answer']
missing = [col for col in required if col not in df.columns]
if missing:
    print(f"❌ Missing columns: {missing}")
    print(f"Available columns: {df.columns.tolist()}")
else:
    print(f"✓ All required columns found")

# Sample data to understand format
print(f"\nSample article (first 100 chars):")
print(df['article'].iloc[0][:100])
print(f"\nSample question:")
print(df['question'].iloc[0])
print(f"\nSample options:")
print(df['options'].iloc[0])
print(f"\nSample answer:")
print(df['answer'].iloc[0])


# ============================================================================
# CELL 5: Prepare Dataset
# ============================================================================

# Debug: Check dataframe state before processing
print("[DEBUG] DataFrame state before processing:")
print(f"  Columns: {df.columns.tolist()}")
print(f"  Shape: {df.shape}")
if 'options' in df.columns:
    print(f"  options column exists")
    print(f"  options dtype: {df['options'].dtype}")
    print(f"  options NaN count: {df['options'].isna().sum()}")
    print(f"  Sample options values:")
    for i in range(min(3, len(df))):
        print(f"    [{i}] {repr(df['options'].iloc[i])}")
else:
    print(f"  WARNING: options column NOT found!")

# Verify required columns exist
required_cols = ['article', 'question', 'options', 'answer']
if not all(col in df.columns for col in required_cols):
    missing = [col for col in required_cols if col not in df.columns]
    print(f"❌ ERROR: Missing required columns: {missing}")
    print(f"Available columns: {df.columns.tolist()}")
    print("\nTry renaming columns or checking the CSV format")
else:
    print("✓ All required columns present")

# Use subset for faster training (optional: use full dataset by changing 5000)
df_subset = df.head(5000)
print(f"\nUsing {len(df_subset)} records for training")

# Prepare QA dataset
print("\n[Step 1] Preparing Q&A dataset...")
qa_df = prepare_qa_dataset(df_subset)

if len(qa_df) == 0:
    print("❌ ERROR: No Q&A pairs created. Check the data format.")
    print(f"Sample row: {df_subset.iloc[0]}")
else:
    print(f"✓ Prepared {len(qa_df)} Q&A pairs")
    print(f"  Class distribution: {qa_df['label'].value_counts().to_dict()}")
    
    # Remove any NaN rows
    qa_df = qa_df.dropna()
    print(f"✓ After cleanup: {len(qa_df)} records")


# ============================================================================
# CELL 6: Feature Engineering
# ============================================================================

print("\n[Step 2] Feature Engineering...")

feature_engineer = FeatureEngineer(max_features=5000)

# Build feature matrix
X, y = build_feature_matrix_model_a(qa_df, feature_engineer, fit=True)

print(f"✓ Feature matrix shape: {X.shape}")
print(f"  Sparse matrix memory: {X.data.nbytes / 1e6:.2f} MB")
print(f"  Sparsity: {1 - (X.nnz / (X.shape[0] * X.shape[1])):.4f}")
print(f"  Class distribution: {np.bincount(y)}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Train: {X_train.shape}, Test: {X_test.shape}")


# ============================================================================
# CELL 7: Train Logistic Regression
# ============================================================================

print("\n[Model 1] Training Logistic Regression...")
import time
start_time = time.time()

with tqdm(total=2, desc="Logistic Regression", leave=True) as pbar:
    lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    lr_model.fit(X_train, y_train)
    pbar.update(1)
    
    y_pred_lr = lr_model.predict(X_test)
    y_pred_proba_lr = lr_model.predict_proba(X_test)
    pbar.update(1)

elapsed = time.time() - start_time
print(f"  ✓ Training completed in {elapsed:.2f}s")

acc_lr = accuracy_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr, average='macro')
precision_lr = precision_score(y_test, y_pred_lr, average='macro')
recall_lr = recall_score(y_test, y_pred_lr, average='macro')

print(f"  Metrics computed ✓")
print(f"  Accuracy:  {acc_lr:.4f}")
print(f"  Precision: {precision_lr:.4f}")
print(f"  Recall:    {recall_lr:.4f}")
print(f"  F1 Score:  {f1_lr:.4f}")


# ============================================================================
# CELL 8: Train SVM
# ============================================================================

print("\n[Model 2] Training SVM...")
start_time = time.time()

with tqdm(total=2, desc="SVM", leave=True) as pbar:
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    pbar.update(1)
    
    y_pred_svm = svm_model.predict(X_test)
    y_pred_proba_svm = svm_model.predict_proba(X_test)
    pbar.update(1)

elapsed = time.time() - start_time
print(f"  ✓ Training completed in {elapsed:.2f}s")

acc_svm = accuracy_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm, average='macro')
precision_svm = precision_score(y_test, y_pred_svm, average='macro')
recall_svm = recall_score(y_test, y_pred_svm, average='macro')

print(f"  Metrics computed ✓")
print(f"  Accuracy:  {acc_svm:.4f}")
print(f"  Precision: {precision_svm:.4f}")
print(f"  Recall:    {recall_svm:.4f}")
print(f"  F1 Score:  {f1_svm:.4f}")


# ============================================================================
# CELL 9: Train Naive Bayes
# ============================================================================

print("\n[Model 3] Training Naive Bayes...")
start_time = time.time()

with tqdm(total=3, desc="Naive Bayes", leave=True) as pbar:
    # Convert sparse to dense for Naive Bayes
    X_train_dense = X_train.toarray()
    X_test_dense = X_test.toarray()
    pbar.update(1)
    
    nb_model = GaussianNB()
    nb_model.fit(X_train_dense, y_train)
    pbar.update(1)
    
    y_pred_nb = nb_model.predict(X_test_dense)
    y_pred_proba_nb = nb_model.predict_proba(X_test_dense)
    pbar.update(1)

elapsed = time.time() - start_time
print(f"  ✓ Training completed in {elapsed:.2f}s")

acc_nb = accuracy_score(y_test, y_pred_nb)
f1_nb = f1_score(y_test, y_pred_nb, average='macro')
precision_nb = precision_score(y_test, y_pred_nb, average='macro')
recall_nb = recall_score(y_test, y_pred_nb, average='macro')

print(f"  Metrics computed ✓")
print(f"  Accuracy:  {acc_nb:.4f}")
print(f"  Precision: {precision_nb:.4f}")
print(f"  Recall:    {recall_nb:.4f}")
print(f"  F1 Score:  {f1_nb:.4f}")


# ============================================================================
# TRAINING PROGRESS TRACKER
# ============================================================================

print("\n" + "=" * 70)
print("TRAINING PROGRESS")
print("=" * 70)
print("✓ [1/10] Logistic Regression      (completed)")
print("✓ [2/10] SVM                       (completed)")
print("✓ [3/10] Naive Bayes               (completed)")
print("▶ [4/10] Random Forest             (training...)")
print("○ [5/10] XGBoost                   (pending)")
print("○ [6/10] Soft Voting Ensemble      (pending)")
print("○ [7/10] Stacking Classifier       (pending)")
print("○ [8/10] K-Means                   (pending)")
print("○ [9/10] Label Propagation         (pending)")
print("○ [10/10] Gaussian Mixture Model   (pending)")
print("=" * 70 + "\n")


# ============================================================================
# CELL 10: Train Random Forest
# ============================================================================

print("\n[Model 4] Training Random Forest...")
start_time = time.time()

with tqdm(total=2, desc="Random Forest", leave=True) as pbar:
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_dense, y_train)
    pbar.update(1)
    
    y_pred_rf = rf_model.predict(X_test_dense)
    y_pred_proba_rf = rf_model.predict_proba(X_test_dense)
    pbar.update(1)

elapsed = time.time() - start_time
print(f"  ✓ Training completed in {elapsed:.2f}s")

acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='macro')
precision_rf = precision_score(y_test, y_pred_rf, average='macro')
recall_rf = recall_score(y_test, y_pred_rf, average='macro')

print(f"  Metrics computed ✓")
print(f"  Accuracy:  {acc_rf:.4f}")
print(f"  Precision: {precision_rf:.4f}")
print(f"  Recall:    {recall_rf:.4f}")
print(f"  F1 Score:  {f1_rf:.4f}")


# ============================================================================
# CELL 11: Train XGBoost
# ============================================================================

print("\n[Model 5] Training XGBoost...")
start_time = time.time()

with tqdm(total=2, desc="XGBoost", leave=True) as pbar:
    xgb_model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train_dense, y_train, verbose=0)
    pbar.update(1)
    
    y_pred_xgb = xgb_model.predict(X_test_dense)
    y_pred_proba_xgb = xgb_model.predict_proba(X_test_dense)
    pbar.update(1)

elapsed = time.time() - start_time
print(f"  ✓ Training completed in {elapsed:.2f}s")

acc_xgb = accuracy_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb, average='macro')
precision_xgb = precision_score(y_test, y_pred_xgb, average='macro')
recall_xgb = recall_score(y_test, y_pred_xgb, average='macro')

print(f"  Metrics computed ✓")
print(f"  Accuracy:  {acc_xgb:.4f}")
print(f"  Precision: {precision_xgb:.4f}")
print(f"  Recall:    {recall_xgb:.4f}")
print(f"  F1 Score:  {f1_xgb:.4f}")


# ============================================================================
# CELL 12: Train Ensemble (Soft Voting)
# ============================================================================

print("\n[Model 6] Training Ensemble (Soft Voting)...")
start_time = time.time()

with tqdm(total=2, desc="Soft Voting Ensemble", leave=True) as pbar:
    # Use compatible models for ensemble
    ensemble_voting = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('xgb', xgb_model),
            ('nb', nb_model)
        ],
        voting='soft'
    )
    ensemble_voting.fit(X_train_dense, y_train)
    pbar.update(1)
    
    y_pred_ensemble_voting = ensemble_voting.predict(X_test_dense)
    y_pred_proba_ensemble_voting = ensemble_voting.predict_proba(X_test_dense)
    pbar.update(1)

elapsed = time.time() - start_time
print(f"  ✓ Training completed in {elapsed:.2f}s")

acc_ensemble_voting = accuracy_score(y_test, y_pred_ensemble_voting)
f1_ensemble_voting = f1_score(y_test, y_pred_ensemble_voting, average='macro')
precision_ensemble_voting = precision_score(y_test, y_pred_ensemble_voting, average='macro')
recall_ensemble_voting = recall_score(y_test, y_pred_ensemble_voting, average='macro')

print(f"  Metrics computed ✓")
print(f"  Accuracy:  {acc_ensemble_voting:.4f}")
print(f"  Precision: {precision_ensemble_voting:.4f}")
print(f"  Recall:    {recall_ensemble_voting:.4f}")
print(f"  F1 Score:  {f1_ensemble_voting:.4f}")


# ============================================================================
# CELL 13: Train Stacking Classifier
# ============================================================================

print("\n[Model 7] Training Stacking Classifier...")
start_time = time.time()

with tqdm(total=2, desc="Stacking Classifier", leave=True) as pbar:
    estimators = [
        ('rf', rf_model),
        ('xgb', xgb_model),
        ('nb', nb_model)
    ]
    
    ensemble_stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5
    )
    ensemble_stacking.fit(X_train_dense, y_train)
    pbar.update(1)
    
    y_pred_ensemble_stacking = ensemble_stacking.predict(X_test_dense)
    y_pred_proba_ensemble_stacking = ensemble_stacking.predict_proba(X_test_dense)
    pbar.update(1)

elapsed = time.time() - start_time
print(f"  ✓ Training completed in {elapsed:.2f}s")

acc_ensemble_stacking = accuracy_score(y_test, y_pred_ensemble_stacking)
f1_ensemble_stacking = f1_score(y_test, y_pred_ensemble_stacking, average='macro')
precision_ensemble_stacking = precision_score(y_test, y_pred_ensemble_stacking, average='macro')
recall_ensemble_stacking = recall_score(y_test, y_pred_ensemble_stacking, average='macro')

print(f"  Metrics computed ✓")
print(f"  Accuracy:  {acc_ensemble_stacking:.4f}")
print(f"  Precision: {precision_ensemble_stacking:.4f}")
print(f"  Recall:    {recall_ensemble_stacking:.4f}")
print(f"  F1 Score:  {f1_ensemble_stacking:.4f}")


# ============================================================================
# CELL 14: Unsupervised Learning - K-Means
# ============================================================================

print("\n[Unsupervised 1] Training K-Means...")
start_time = time.time()

with tqdm(total=1, desc="K-Means", leave=True) as pbar:
    kmeans_model = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans_clusters = kmeans_model.fit_predict(X_train_dense)
    pbar.update(1)

elapsed = time.time() - start_time
print(f"  ✓ Training completed in {elapsed:.2f}s")
print(f"  Clustered into 5 clusters")
print(f"  Distribution: {np.bincount(kmeans_clusters)}")


# ============================================================================
# CELL 15: Unsupervised Learning - Label Propagation
# ============================================================================

print("\n[Semi-Supervised 1] Training Label Propagation...")
start_time = time.time()

with tqdm(total=3, desc="Label Propagation", leave=True) as pbar:
    # Create semi-labeled dataset (70% labeled, 30% unlabeled)
    n_samples = len(y_train)
    n_unlabeled = int(n_samples * 0.3)
    pbar.update(1)
    
    y_train_semi = y_train.copy()
    unlabeled_idx = np.random.choice(n_samples, n_unlabeled, replace=False)
    y_train_semi[unlabeled_idx] = -1
    pbar.update(1)
    
    lp_model = LabelPropagation(kernel='rbf', gamma=0.1)
    lp_model.fit(X_train_dense, y_train_semi)
    pbar.update(1)

# Evaluate on test set
y_test_pred_lp = lp_model.predict(X_test_dense)
acc_lp = accuracy_score(y_test, y_test_pred_lp)

elapsed = time.time() - start_time
print(f"  ✓ Training completed in {elapsed:.2f}s")
print(f"  Label Propagation Test Accuracy: {acc_lp:.4f}")


# ============================================================================
# CELL 16: Unsupervised Learning - Gaussian Mixture Model
# ============================================================================

print("\n[Unsupervised 2] Training Gaussian Mixture Model...")
start_time = time.time()

with tqdm(total=1, desc="Gaussian Mixture Model", leave=True) as pbar:
    # Use diagonal covariance for speed on high-dimensional data
    gmm_model = GaussianMixture(
        n_components=5, 
        covariance_type='diag',
        max_iter=50,
        n_init=3,
        random_state=42
    )
    gmm_clusters = gmm_model.fit_predict(X_train_dense)
    pbar.update(1)

elapsed = time.time() - start_time
print(f"  ✓ Training completed in {elapsed:.2f}s")
print(f"  Clustered into 5 components")
print(f"  BIC Score: {gmm_model.bic(X_train_dense):.2f}")
print(f"  Distribution: {np.bincount(gmm_clusters)}")


# ============================================================================
# CELL 17: Evaluation Summary
# ============================================================================

print("\n" + "=" * 70)
print("EVALUATION SUMMARY - Model A")
print("=" * 70)

results = {
    'Logistic Regression': {'Acc': acc_lr, 'F1': f1_lr, 'Prec': precision_lr, 'Rec': recall_lr},
    'SVM': {'Acc': acc_svm, 'F1': f1_svm, 'Prec': precision_svm, 'Rec': recall_svm},
    'Naive Bayes': {'Acc': acc_nb, 'F1': f1_nb, 'Prec': precision_nb, 'Rec': recall_nb},
    'Random Forest': {'Acc': acc_rf, 'F1': f1_rf, 'Prec': precision_rf, 'Rec': recall_rf},
    'XGBoost': {'Acc': acc_xgb, 'F1': f1_xgb, 'Prec': precision_xgb, 'Rec': recall_xgb},
    'Soft Voting': {'Acc': acc_ensemble_voting, 'F1': f1_ensemble_voting, 'Prec': precision_ensemble_voting, 'Rec': recall_ensemble_voting},
    'Stacking': {'Acc': acc_ensemble_stacking, 'F1': f1_ensemble_stacking, 'Prec': precision_ensemble_stacking, 'Rec': recall_ensemble_stacking},
}

results_df = pd.DataFrame(results).T
print(results_df.to_string())

best_model = results_df['Acc'].idxmax()
print(f"\n✓ Best performing model: {best_model} (Accuracy: {results_df.loc[best_model, 'Acc']:.4f})")


# ============================================================================
# CELL 18: Save All Models
# ============================================================================

print("\n[Final] Saving all models...")

import os
os.makedirs('/kaggle/working/models/model_a/traditional/', exist_ok=True)

# Save supervised models
joblib.dump(lr_model, '/kaggle/working/models/model_a/traditional/lr_model.pkl')
joblib.dump(svm_model, '/kaggle/working/models/model_a/traditional/svm_model.pkl')
joblib.dump(nb_model, '/kaggle/working/models/model_a/traditional/nb_model.pkl')
joblib.dump(rf_model, '/kaggle/working/models/model_a/traditional/rf_model.pkl')
joblib.dump(xgb_model, '/kaggle/working/models/model_a/traditional/xgb_model.pkl')

# Save ensemble models
joblib.dump(ensemble_voting, '/kaggle/working/models/model_a/traditional/ensemble_voting_model.pkl')
joblib.dump(ensemble_stacking, '/kaggle/working/models/model_a/traditional/ensemble_stacking_model.pkl')

# Save unsupervised/semi-supervised models
joblib.dump(kmeans_model, '/kaggle/working/models/model_a/traditional/kmeans_model.pkl')
joblib.dump(lp_model, '/kaggle/working/models/model_a/traditional/label_propagation_model.pkl')
joblib.dump(gmm_model, '/kaggle/working/models/model_a/traditional/gmm_model.pkl')

# Save feature engineer
feature_engineer.save('/kaggle/working/models/model_a/traditional/feature_engineer.pkl')

# Save results summary
results_df.to_csv('/kaggle/working/models/model_a/traditional/results_summary.csv')

print("✓ All models saved to /kaggle/working/models/model_a/traditional/")
print("\nSaved files:")
print("  - lr_model.pkl")
print("  - svm_model.pkl")
print("  - nb_model.pkl")
print("  - rf_model.pkl")
print("  - xgb_model.pkl")
print("  - ensemble_voting_model.pkl")
print("  - ensemble_stacking_model.pkl")
print("  - kmeans_model.pkl")
print("  - label_propagation_model.pkl")
print("  - gmm_model.pkl")
print("  - feature_engineer.pkl")
print("  - results_summary.csv")
print("\n✓ DOWNLOAD /kaggle/working/ when complete!")

