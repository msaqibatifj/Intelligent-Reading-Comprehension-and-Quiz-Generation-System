"""
Model B Training Script: Distractor & Hint Generator
Trains models for plausible distractor generation and graduated hint extraction.

Run this locally or as a Kaggle notebook cell.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error, r2_score
)
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import FeatureEngineer, prepare_distractor_dataset, build_feature_matrix_model_b
from src.evaluate import ModelBEvaluator

# ============================================================================
# Configuration
# ============================================================================
CONFIG = {
    'data_path': 'data/raw/train.csv',  # Update with your RACE CSV path
    'test_size': 0.2,
    'random_state': 42,
    'max_features': 5000,
    'models_output_dir': 'models/model_b/traditional/',
    'word2vec_window': 5,
    'word2vec_min_count': 2,
    'word2vec_workers': 4,
}


# ============================================================================
# Model Training Functions
# ============================================================================

def train_word2vec_embeddings(texts, output_path):
    """Train Word2Vec embeddings on corpus."""
    print("\n[Model B] Training Word2Vec embeddings...")
    
    # Tokenize texts
    sentences = []
    for text in texts:
        tokens = word_tokenize(text.lower())
        sentences.append(tokens)
    
    # Train Word2Vec
    w2v_model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=CONFIG['word2vec_window'],
        min_count=CONFIG['word2vec_min_count'],
        workers=CONFIG['word2vec_workers'],
        sg=1  # Skip-gram model
    )
    
    print(f"  Trained on {len(sentences)} sentences")
    print(f"  Vocabulary size: {len(w2v_model.wv)}")
    
    joblib.dump(w2v_model, output_path)
    return w2v_model


def extract_distractor_features(passage, question, correct_answer, distractors, feature_engineer):
    """Extract features for distractor ranking."""
    features_list = []
    
    for distractor in distractors:
        lexical = feature_engineer.extract_lexical_features(
            question, [distractor], passage
        )
        features_list.append(lexical[0])
    
    # Also add features for correct answer
    lexical_correct = feature_engineer.extract_lexical_features(
        question, [correct_answer], passage
    )
    features_list.append(lexical_correct[0])
    
    return np.array(features_list), len(distractors)


def train_distractor_ranker(X, y):
    """Train distractor ranking model (binary: correct=1, distractor=0)."""
    print("\n[Model B] Training Distractor Ranker (Logistic Regression)...")
    
    model = LogisticRegression(max_iter=1000, random_state=CONFIG['random_state'])
    model.fit(X, y)
    
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    
    print(f"  Training Accuracy: {acc:.4f}")
    
    return model


def train_distractor_ranker_rf(X, y):
    """Train distractor ranking model using Random Forest."""
    print("\n[Model B] Training Distractor Ranker (Random Forest)...")
    
    model = RandomForestClassifier(n_estimators=100, random_state=CONFIG['random_state'])
    model.fit(X, y)
    
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    
    print(f"  Training Accuracy: {acc:.4f}")
    
    return model


def extract_hint_features(passage, question, correct_answer):
    """Extract features for hint scoring/extraction."""
    sentences = sent_tokenize(passage)
    
    hint_features = []
    hint_sentences = []
    
    for sent_idx, sent in enumerate(sentences):
        # Feature 1: Keyword overlap with question
        q_words = set(question.lower().split())
        s_words = set(sent.lower().split())
        overlap = len(q_words & s_words) / max(len(q_words), 1)
        
        # Feature 2: Position in passage (normalized)
        position = sent_idx / max(len(sentences), 1)
        
        # Feature 3: Sentence length
        length = len(sent.split()) / 20.0  # Normalize by typical max
        
        # Feature 4: Contains answer (should be low for good hints)
        contains_answer = 1.0 if correct_answer.lower() in sent.lower() else 0.0
        
        features = [overlap, position, length, contains_answer]
        hint_features.append(features)
        hint_sentences.append(sent)
    
    return np.array(hint_features), hint_sentences


def train_hint_extractor_lr(X, y):
    """Train hint extraction model using Logistic Regression."""
    print("\n[Model B] Training Hint Extractor (Logistic Regression)...")
    
    model = LogisticRegression(max_iter=1000, random_state=CONFIG['random_state'])
    model.fit(X, y)
    
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    
    print(f"  Training Accuracy: {acc:.4f}")
    
    return model


def train_hint_scorer_regression(X, y):
    """Train hint scoring regression model."""
    print("\n[Model B] Training Hint Scorer (Random Forest Regression)...")
    
    model = RandomForestRegressor(n_estimators=100, random_state=CONFIG['random_state'])
    model.fit(X, y)
    
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    print(f"  Training R²: {r2:.4f} | RMSE: {rmse:.4f}")
    
    return model


# ============================================================================
# Main Training Pipeline
# ============================================================================

def main():
    print("=" * 80)
    print("Model B Training Pipeline: Distractor & Hint Generator")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(CONFIG['models_output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"\n[Step 1] Loading dataset from {CONFIG['data_path']}...")
    try:
        df = pd.read_csv(CONFIG['data_path'])
        print(f"  Loaded {len(df)} records")
    except FileNotFoundError:
        print(f"  ❌ Dataset not found at {CONFIG['data_path']}")
        print("  Please download RACE dataset and place it in data/raw/")
        return
    
    # Prepare distractor dataset
    print("\n[Step 2] Preparing distractor dataset...")
    distractor_df = prepare_distractor_dataset(df)
    print(f"  Prepared {len(distractor_df)} MCQs with distractors")
    
    # Feature engineering
    print("\n[Step 3] Feature engineering...")
    feature_engineer = FeatureEngineer(max_features=CONFIG['max_features'])
    X_distractor, y_distractor = build_feature_matrix_model_b(distractor_df, feature_engineer, fit=False)
    print(f"  Distractor feature matrix shape: {X_distractor.shape}")
    
    # Save feature engineer
    feature_engineer.save(str(output_dir / 'feature_engineer.pkl'))
    
    # ========================================================================
    # Word2Vec Embeddings
    # ========================================================================
    print("\n" + "=" * 80)
    print("Training Word2Vec Embeddings")
    print("=" * 80)
    
    all_texts = (df['passage'].astype(str) + ' ' + df['question'].astype(str)).tolist()
    w2v_model = train_word2vec_embeddings(all_texts, str(output_dir / 'word2vec_model.pkl'))
    
    # ========================================================================
    # Distractor Ranking Models
    # ========================================================================
    print("\n" + "=" * 80)
    print("Training Distractor Ranking Models")
    print("=" * 80)
    
    # Train/test split
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
        X_distractor, y_distractor, test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state']
    )
    
    # Logistic Regression for distractor ranking
    lr_distractor = train_distractor_ranker(X_train_d, y_train_d)
    y_pred_lr_d = lr_distractor.predict(X_test_d)
    acc_lr_d = accuracy_score(y_test_d, y_pred_lr_d)
    print(f"  LR Distractor Ranker - Test Accuracy: {acc_lr_d:.4f}")
    joblib.dump(lr_distractor, output_dir / 'distractor_ranker_lr.pkl')
    
    # Random Forest for distractor ranking
    rf_distractor = train_distractor_ranker_rf(X_train_d, y_train_d)
    y_pred_rf_d = rf_distractor.predict(X_test_d)
    acc_rf_d = accuracy_score(y_test_d, y_pred_rf_d)
    print(f"  RF Distractor Ranker - Test Accuracy: {acc_rf_d:.4f}")
    joblib.dump(rf_distractor, output_dir / 'distractor_ranker_rf.pkl')
    
    # Use best distractor ranker
    best_distractor_ranker = rf_distractor if acc_rf_d > acc_lr_d else lr_distractor
    joblib.dump(best_distractor_ranker, output_dir / 'distractor_ranker.pkl')
    
    # ========================================================================
    # Hint Extraction & Scoring Models
    # ========================================================================
    print("\n" + "=" * 80)
    print("Training Hint Extraction & Scoring Models")
    print("=" * 80)
    
    hint_features_list = []
    hint_labels_list = []
    hint_scores_list = []
    
    print("\n[Step 4] Extracting hint features...")
    for idx, row in tqdm(distractor_df.iterrows(), total=len(distractor_df)):
        passage = row['passage']
        question = row['question']
        correct_answer = row['correct_answer']
        
        hint_feats, hint_sents = extract_hint_features(passage, question, correct_answer)
        
        for feat_idx, feat in enumerate(hint_feats):
            hint_features_list.append(feat)
            
            # Label: relevant (1) if high keyword overlap and doesn't contain answer
            keyword_overlap = feat[0]
            contains_answer = feat[3]
            label = 1 if keyword_overlap > 0.1 and contains_answer < 0.5 else 0
            hint_labels_list.append(label)
            
            # Score: cosine similarity to question (mock score)
            hint_scores_list.append(keyword_overlap)
    
    X_hints = np.array(hint_features_list)
    y_hints_binary = np.array(hint_labels_list)
    y_hints_scores = np.array(hint_scores_list)
    
    print(f"  Extracted {len(X_hints)} hint features")
    
    # Train/test split for hints
    X_train_h, X_test_h, y_train_h, y_test_h, s_train_h, s_test_h = train_test_split(
        X_hints, y_hints_binary, y_hints_scores, test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state']
    )
    
    # Hint Extractor (binary classification)
    hint_extractor = train_hint_extractor_lr(X_train_h, y_train_h)
    y_pred_h = hint_extractor.predict(X_test_h)
    acc_h = accuracy_score(y_test_h, y_pred_h)
    f1_h = f1_score(y_test_h, y_pred_h, average='binary')
    print(f"  Hint Extractor - Test Accuracy: {acc_h:.4f} | F1: {f1_h:.4f}")
    joblib.dump(hint_extractor, output_dir / 'hint_extractor.pkl')
    
    # Hint Scorer (regression)
    hint_scorer = train_hint_scorer_regression(X_train_h, s_train_h)
    s_pred_h = hint_scorer.predict(X_test_h)
    r2_h = r2_score(s_test_h, s_pred_h)
    print(f"  Hint Scorer - Test R²: {r2_h:.4f}")
    joblib.dump(hint_scorer, output_dir / 'hint_scorer.pkl')
    
    # ========================================================================
    # Evaluation Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    
    print(f"Distractor Ranker (RF)      | Accuracy: {acc_rf_d:.4f}")
    print(f"Hint Extractor (LR)         | Accuracy: {acc_h:.4f} | F1: {f1_h:.4f}")
    print(f"Hint Scorer (RF Regression) | R²: {r2_h:.4f}")
    
    print(f"\n✓ All models saved to {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
