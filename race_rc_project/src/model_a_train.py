"""
Model A Training Script: Question & Answer Generator/Verifier
Trains supervised, unsupervised, and ensemble models for answer verification.

Run this locally or as a Kaggle notebook cell.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from tqdm import tqdm
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import FeatureEngineer, prepare_qa_dataset, build_feature_matrix_model_a
from src.evaluate import ModelAEvaluator

# ============================================================================
# Configuration
# ============================================================================
CONFIG = {
    'data_path': 'data/raw/train.csv',  # Update with your RACE CSV path
    'test_size': 0.2,
    'random_state': 42,
    'max_features': 5000,
    'models_output_dir': 'models/model_a/traditional/',
}


# ============================================================================
# Model Training Functions
# ============================================================================

def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train Logistic Regression model."""
    print("\n[Model A] Training Logistic Regression...")
    
    model = LogisticRegression(max_iter=1000, random_state=CONFIG['random_state'])
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  F1 Score: {f1_score(y_test, y_pred, average='macro'):.4f}")
    
    return model, y_pred, y_pred_proba


def train_svm(X_train, y_train, X_test, y_test):
    """Train Support Vector Machine model."""
    print("\n[Model A] Training SVM...")
    
    model = SVC(kernel='rbf', probability=True, random_state=CONFIG['random_state'])
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  F1 Score: {f1_score(y_test, y_pred, average='macro'):.4f}")
    
    return model, y_pred, y_pred_proba


def train_naive_bayes(X_train, y_train, X_test, y_test):
    """Train Naive Bayes model for question type classification."""
    print("\n[Model A] Training Naive Bayes...")
    
    # Convert sparse to dense for Naive Bayes
    X_train_dense = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
    X_test_dense = X_test.toarray() if hasattr(X_test, 'toarray') else X_test
    
    model = GaussianNB()
    model.fit(X_train_dense, y_train)
    
    y_pred = model.predict(X_test_dense)
    y_pred_proba = model.predict_proba(X_test_dense)
    
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  F1 Score: {f1_score(y_test, y_pred, average='macro'):.4f}")
    
    return model, y_pred, y_pred_proba


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest model for difficulty estimation."""
    print("\n[Model A] Training Random Forest...")
    
    X_train_dense = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
    X_test_dense = X_test.toarray() if hasattr(X_test, 'toarray') else X_test
    
    model = RandomForestClassifier(n_estimators=100, random_state=CONFIG['random_state'])
    model.fit(X_train_dense, y_train)
    
    y_pred = model.predict(X_test_dense)
    y_pred_proba = model.predict_proba(X_test_dense)
    
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  F1 Score: {f1_score(y_test, y_pred, average='macro'):.4f}")
    
    return model, y_pred, y_pred_proba


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost model."""
    print("\n[Model A] Training XGBoost...")
    
    X_train_dense = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
    X_test_dense = X_test.toarray() if hasattr(X_test, 'toarray') else X_test
    
    model = XGBClassifier(n_estimators=100, random_state=CONFIG['random_state'], use_label_encoder=False)
    model.fit(X_train_dense, y_train)
    
    y_pred = model.predict(X_test_dense)
    y_pred_proba = model.predict_proba(X_test_dense)
    
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  F1 Score: {f1_score(y_test, y_pred, average='macro'):.4f}")
    
    return model, y_pred, y_pred_proba


def train_ensemble_voting(models_dict, X_train, y_train, X_test, y_test):
    """Train ensemble model using soft voting."""
    print("\n[Model A] Training Ensemble (Soft Voting)...")
    
    estimators = [(name, model) for name, model in models_dict.items()]
    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    ensemble.fit(X_train, y_train)
    
    y_pred = ensemble.predict(X_test)
    y_pred_proba = ensemble.predict_proba(X_test)
    
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  F1 Score: {f1_score(y_test, y_pred, average='macro'):.4f}")
    
    return ensemble, y_pred, y_pred_proba


def train_stacking_classifier(base_models, X_train, y_train, X_test, y_test):
    """Train stacking classifier."""
    print("\n[Model A] Training Stacking Classifier...")
    
    estimators = [(name, model) for name, model in base_models.items()]
    meta_learner = LogisticRegression()
    
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5
    )
    stacking.fit(X_train, y_train)
    
    y_pred = stacking.predict(X_test)
    y_pred_proba = stacking.predict_proba(X_test)
    
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  F1 Score: {f1_score(y_test, y_pred, average='macro'):.4f}")
    
    return stacking, y_pred, y_pred_proba


# ============================================================================
# Unsupervised/Semi-Supervised Learning
# ============================================================================

def train_kmeans_clustering(X, n_clusters=5):
    """K-Means clustering for grouping Q&A pairs."""
    print("\n[Model A] Training K-Means Clustering...")
    
    X_dense = X.toarray() if hasattr(X, 'toarray') else X
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=CONFIG['random_state'])
    clusters = kmeans.fit_predict(X_dense)
    
    print(f"  Clustered {len(clusters)} samples into {n_clusters} clusters")
    print(f"  Cluster distribution: {np.bincount(clusters)}")
    
    return kmeans, clusters


def train_label_propagation(X, y, unlabeled_ratio=0.3):
    """Label Propagation for semi-supervised learning."""
    print("\n[Model A] Training Label Propagation...")
    
    X_dense = X.toarray() if hasattr(X, 'toarray') else X
    
    # Create semi-labeled dataset
    n_samples = len(y)
    n_unlabeled = int(n_samples * unlabeled_ratio)
    
    y_semi = y.copy()
    unlabeled_idx = np.random.choice(n_samples, n_unlabeled, replace=False)
    y_semi[unlabeled_idx] = -1  # Mark as unlabeled
    
    lp = LabelPropagation(kernel='rbf', gamma=0.1)
    y_pred = lp.fit_predict(X_dense, y_semi)
    
    accuracy = accuracy_score(y, y_pred)
    print(f"  Label Propagation Accuracy: {accuracy:.4f}")
    
    return lp, y_pred


def train_gaussian_mixture_model(X, n_components=5):
    """Gaussian Mixture Model for soft clustering."""
    print("\n[Model A] Training Gaussian Mixture Model...")
    
    X_dense = X.toarray() if hasattr(X, 'toarray') else X
    
    gmm = GaussianMixture(n_components=n_components, random_state=CONFIG['random_state'])
    clusters = gmm.fit_predict(X_dense)
    
    print(f"  Clustered {len(clusters)} samples into {n_components} components")
    print(f"  BIC Score: {gmm.bic(X_dense):.2f}")
    
    return gmm, clusters


# ============================================================================
# Main Training Pipeline
# ============================================================================

def main():
    print("=" * 80)
    print("Model A Training Pipeline: Q&A Generator/Verifier")
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
    
    # Prepare dataset
    print("\n[Step 2] Preparing Q&A dataset...")
    qa_df = prepare_qa_dataset(df)
    print(f"  Prepared {len(qa_df)} Q&A pairs")
    
    # Feature engineering
    print("\n[Step 3] Feature engineering...")
    feature_engineer = FeatureEngineer(max_features=CONFIG['max_features'])
    X, y = build_feature_matrix_model_a(qa_df, feature_engineer, fit=True)
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Class distribution: {np.bincount(y)}")
    
    # Save feature engineer
    feature_engineer.save(str(output_dir / 'feature_engineer.pkl'))
    print(f"  Saved feature engineer to {output_dir / 'feature_engineer.pkl'}")
    
    # Train/test split
    print("\n[Step 4] Train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG['test_size'], random_state=CONFIG['random_state']
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Initialize evaluator
    evaluator = ModelAEvaluator()
    
    # ========================================================================
    # Supervised Models
    # ========================================================================
    print("\n" + "=" * 80)
    print("Training Supervised Models")
    print("=" * 80)
    
    models = {}
    predictions = {}
    
    # Logistic Regression
    models['lr'], predictions['lr'], _ = train_logistic_regression(X_train, y_train, X_test, y_test)
    joblib.dump(models['lr'], output_dir / 'lr_model.pkl')
    
    # SVM
    models['svm'], predictions['svm'], _ = train_svm(X_train, y_train, X_test, y_test)
    joblib.dump(models['svm'], output_dir / 'svm_model.pkl')
    
    # Naive Bayes
    models['nb'], predictions['nb'], _ = train_naive_bayes(X_train, y_train, X_test, y_test)
    joblib.dump(models['nb'], output_dir / 'nb_model.pkl')
    
    # Random Forest
    models['rf'], predictions['rf'], _ = train_random_forest(X_train, y_train, X_test, y_test)
    joblib.dump(models['rf'], output_dir / 'rf_model.pkl')
    
    # XGBoost
    models['xgb'], predictions['xgb'], _ = train_xgboost(X_train, y_train, X_test, y_test)
    joblib.dump(models['xgb'], output_dir / 'xgb_model.pkl')
    
    # ========================================================================
    # Unsupervised/Semi-Supervised Models
    # ========================================================================
    print("\n" + "=" * 80)
    print("Training Unsupervised/Semi-Supervised Models")
    print("=" * 80)
    
    # K-Means
    kmeans, kmeans_clusters = train_kmeans_clustering(X_train, n_clusters=5)
    joblib.dump(kmeans, output_dir / 'kmeans_model.pkl')
    
    # Label Propagation
    lp, lp_pred = train_label_propagation(X_train, y_train)
    joblib.dump(lp, output_dir / 'label_propagation_model.pkl')
    print(f"  Label Propagation - Test Accuracy: {accuracy_score(y_test, lp.predict(X_test.toarray() if hasattr(X_test, 'toarray') else X_test)):.4f}")
    
    # Gaussian Mixture Model
    gmm, gmm_clusters = train_gaussian_mixture_model(X_train, n_components=5)
    joblib.dump(gmm, output_dir / 'gmm_model.pkl')
    
    # ========================================================================
    # Ensemble Models
    # ========================================================================
    print("\n" + "=" * 80)
    print("Training Ensemble Models")
    print("=" * 80)
    
    # Select base models for ensemble (dense models only for compatibility)
    base_models_for_ensemble = {
        'rf': models['rf'],
        'xgb': models['xgb'],
        'nb': models['nb'],
    }
    
    # Soft Voting
    ensemble_voting, predictions['ensemble_voting'], _ = train_ensemble_voting(
        base_models_for_ensemble, X_train.toarray() if hasattr(X_train, 'toarray') else X_train, 
        y_train, X_test.toarray() if hasattr(X_test, 'toarray') else X_test, y_test
    )
    joblib.dump(ensemble_voting, output_dir / 'ensemble_voting_model.pkl')
    
    # Stacking
    ensemble_stacking, predictions['ensemble_stacking'], _ = train_stacking_classifier(
        base_models_for_ensemble, X_train.toarray() if hasattr(X_train, 'toarray') else X_train,
        y_train, X_test.toarray() if hasattr(X_test, 'toarray') else X_test, y_test
    )
    joblib.dump(ensemble_stacking, output_dir / 'ensemble_stacking_model.pkl')
    
    # ========================================================================
    # Evaluation Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    
    for model_name in ['lr', 'svm', 'nb', 'rf', 'xgb', 'ensemble_voting', 'ensemble_stacking']:
        if model_name in predictions:
            y_pred = predictions[model_name]
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            print(f"{model_name:20s} | Accuracy: {acc:.4f} | F1: {f1:.4f}")
    
    print(f"\n✓ All models saved to {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
