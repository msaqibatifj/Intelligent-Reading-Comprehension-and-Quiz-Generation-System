"""
Kaggle Training Script for Model A - Multi-Class Version (Select Correct Answer)
Run this in a Kaggle Notebook (GPU-enabled)

This version predicts which option (0,1,2,3) is correct, not just verify one answer.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.semi_supervised import LabelPropagation
from sklearn.mixture import GaussianMixture
import xgboost as xgb
from scipy.sparse import csr_matrix, hstack
import joblib
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class FeatureEngineer:
    """Feature engineering for multi-class Q&A (selecting correct option)."""
    
    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.onehot_vectorizer = None
        self.scaler = StandardScaler()
    
    def compute_word_overlap(self, text1, text2):
        """Compute word overlap between texts."""
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
    
    def compute_passage_frequency(self, word, passage):
        """Compute how frequently option appears in passage."""
        words = passage.lower().split()
        if len(words) == 0:
            return 0.0
        return words.count(word.lower()) / len(words)
    
    def extract_lexical_features_for_options(self, question, options, passage):
        """
        Extract lexical features for all options.
        Returns: [4, 4] array - 4 features for each of 4 options
        """
        features_list = []
        for option in options:
            word_overlap = self.compute_word_overlap(question, option)
            char_match = self.compute_char_match_score(question, option)
            option_length = len(option.split())
            passage_freq = self.compute_passage_frequency(option, passage)
            
            features = [word_overlap, char_match, option_length, passage_freq]
            features_list.append(features)
        
        return np.array(features_list)  # [4, 4]
    
    def fit_onehot(self, texts):
        """Fit One-Hot vectorizer on texts."""
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
    
    def save(self, path):
        """Save fitted vectorizers."""
        joblib.dump({
            'onehot': self.onehot_vectorizer,
            'scaler': self.scaler,
            'max_features': self.max_features
        }, path)
    
    @staticmethod
    def load(path):
        """Load saved vectorizers."""
        data = joblib.load(path)
        fe = FeatureEngineer(max_features=data['max_features'])
        fe.onehot_vectorizer = data['onehot']
        fe.scaler = data['scaler']
        return fe


def build_multiclass_feature_matrix(questions, all_options_list, passages, feature_engineer):
    """
    Build combined feature matrix for multi-class classification.
    
    For each question:
    - Extract features for each of the 4 options
    - Concatenate all features into one vector
    - This allows model to compare across options
    
    Args:
        questions: list of question texts
        all_options_list: list of [option_0, option_1, option_2, option_3] for each question
        passages: list of passage texts
        feature_engineer: fitted FeatureEngineer instance
    
    Returns:
        X: [N, feature_dim] feature matrix where feature_dim = 5000*4 + 4*4
    """
    all_features = []
    
    for question, options, passage in zip(questions, all_options_list, passages):
        # One-Hot features for each option
        onehot_parts = []
        for option in options:
            combined = question + ' ' + option
            onehot_feat = feature_engineer.transform_onehot([combined])
            onehot_parts.append(onehot_feat)
        
        # Stack one-hot features from all 4 options
        onehot_all = hstack(onehot_parts)
        
        # Lexical features for all options
        lexical_all = feature_engineer.extract_lexical_features_for_options(
            question, options, passage
        )
        lexical_all_sparse = csr_matrix(lexical_all.flatten()).reshape(1, -1)
        
        # Combine
        combined_features = hstack([onehot_all, lexical_all_sparse])
        all_features.append(combined_features)
    
    # Stack all samples
    X = hstack(all_features)
    return X.toarray()


def convert_answer_to_label(answer_str):
    """Convert answer letter to index: A->0, B->1, C->2, D->3"""
    answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    return answer_map.get(answer_str.strip().upper(), 0)


def train_model_a_multiclass(train_df, val_df, output_dir='./models/model_a/traditional/'):
    """
    Train Model A with multi-class classification.
    
    Expected columns in train_df and val_df:
    - 'article': passage text
    - 'question': question text
    - 'options': list of [option_A, option_B, option_C, option_D]
    - 'answer': 'A', 'B', 'C', or 'D'
    """
    
    print("=" * 70)
    print("TRAINING MODEL A (MULTI-CLASS) ON KAGGLE")
    print("=" * 70)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Parse options column (might be string representation of list)
    def parse_options(opt):
        if isinstance(opt, str):
            import ast
            try:
                return ast.literal_eval(opt)
            except:
                return [opt, '', '', '']
        return opt
    
    train_df['options'] = train_df['options'].apply(parse_options)
    val_df['options'] = val_df['options'].apply(parse_options)
    
    # Convert answers to labels
    train_df['label'] = train_df['answer'].apply(convert_answer_to_label)
    val_df['label'] = val_df['answer'].apply(convert_answer_to_label)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Class distribution (train): {np.bincount(train_df['label'], minlength=4)}")
    
    # 1. Fit feature engineer
    print("\n1. Fitting feature engineer...")
    all_question_texts = train_df['question'].tolist()
    all_option_texts = []
    for opts in train_df['options']:
        all_option_texts.extend(opts)
    
    combined_texts = all_question_texts + all_option_texts
    feature_engineer = FeatureEngineer(max_features=5000)
    feature_engineer.fit_onehot(combined_texts)
    print("   ✓ Feature engineer fitted")
    
    # 2. Build training features
    print("2. Building training feature matrix...")
    X_train = build_multiclass_feature_matrix(
        train_df['question'].tolist(),
        train_df['options'].tolist(),
        train_df['article'].tolist(),
        feature_engineer
    )
    y_train = train_df['label'].values
    
    print(f"   X_train shape: {X_train.shape}")
    print(f"   Class balance: {np.bincount(y_train, minlength=4) / len(y_train) * 100:.1f}%")
    
    # 3. Build validation features
    print("3. Building validation feature matrix...")
    X_val = build_multiclass_feature_matrix(
        val_df['question'].tolist(),
        val_df['options'].tolist(),
        val_df['article'].tolist(),
        feature_engineer
    )
    y_val = val_df['label'].values
    
    # 4. Train multi-class classifiers
    print("\n4. Training classifiers...")
    models = {}
    
    # Logistic Regression
    print("   - Logistic Regression...")
    models['lr_model'] = LogisticRegression(
        max_iter=1000,
        multi_class='multinomial',
        class_weight='balanced',
        random_state=42,
        C=0.5
    )
    models['lr_model'].fit(X_train, y_train)
    lr_score = models['lr_model'].score(X_val, y_val)
    print(f"     Validation accuracy: {lr_score:.4f}")
    
    # Naive Bayes
    print("   - Gaussian Naive Bayes...")
    models['nb_model'] = GaussianNB()
    models['nb_model'].fit(X_train, y_train)
    nb_score = models['nb_model'].score(X_val, y_val)
    print(f"     Validation accuracy: {nb_score:.4f}")
    
    # Random Forest
    print("   - Random Forest...")
    models['rf_model'] = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    models['rf_model'].fit(X_train, y_train)
    rf_score = models['rf_model'].score(X_val, y_val)
    print(f"     Validation accuracy: {rf_score:.4f}")
    
    # SVM
    print("   - Support Vector Machine...")
    models['svm_model'] = SVC(
        kernel='rbf',
        class_weight='balanced',
        probability=True,
        random_state=42,
        gamma='scale',
        decision_function_shape='ovr'
    )
    models['svm_model'].fit(X_train, y_train)
    svm_score = models['svm_model'].score(X_val, y_val)
    print(f"     Validation accuracy: {svm_score:.4f}")
    
    # XGBoost
    print("   - XGBoost...")
    models['xgb_model'] = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        num_class=4
    )
    models['xgb_model'].fit(X_train, y_train)
    xgb_score = models['xgb_model'].score(X_val, y_val)
    print(f"     Validation accuracy: {xgb_score:.4f}")
    
    # 5. Ensemble
    print("   - Ensemble (Voting)...")
    models['ensemble_voting_model'] = VotingClassifier(
        estimators=[
            ('lr', models['lr_model']),
            ('rf', models['rf_model']),
            ('svm', models['svm_model']),
        ],
        voting='soft'
    )
    models['ensemble_voting_model'].fit(X_train, y_train)
    voting_score = models['ensemble_voting_model'].score(X_val, y_val)
    print(f"     Validation accuracy: {voting_score:.4f}")
    
    # 6. Unsupervised (for completeness)
    print("   - K-Means...")
    models['kmeans_model'] = KMeans(n_clusters=4, random_state=42, n_init=10)
    models['kmeans_model'].fit(X_train)
    
    print("   - Label Propagation...")
    models['label_propagation_model'] = LabelPropagation(n_neighbors=7)
    models['label_propagation_model'].fit(X_train, y_train)
    lp_score = models['label_propagation_model'].score(X_val, y_val)
    print(f"     Validation accuracy: {lp_score:.4f}")
    
    print("   - Gaussian Mixture Model...")
    models['gmm_model'] = GaussianMixture(n_components=4, random_state=42)
    models['gmm_model'].fit(X_train)
    
    # 7. Save all models
    print("\n5. Saving models...")
    for model_name, model in models.items():
        path = Path(output_dir) / f"{model_name}.pkl"
        joblib.dump(model, path)
        print(f"   Saved: {path}")
    
    # Save feature engineer
    feature_engineer_path = Path(output_dir) / "feature_engineer.pkl"
    feature_engineer.save(feature_engineer_path)
    print(f"   Saved: {feature_engineer_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("MULTI-CLASS MODEL TRAINED SUCCESSFULLY")
    print("=" * 70)
    print("Model predicts: which option (0,1,2,3) is correct")
    print("Expected accuracy: 60-80%+ (4-way classification)")
    print("\nTop performers:")
    print(f"  1. XGBoost:   {xgb_score:.4f}")
    print(f"  2. Random Forest: {rf_score:.4f}")
    print(f"  3. Voting Ensemble: {voting_score:.4f}")


if __name__ == '__main__':
    # In Kaggle, read from /kaggle/input/race-dataset/
    print("Loading RACE dataset from Kaggle...\n")
    
    # For Arrow format, use pyarrow
    try:
        import pyarrow.ipc as ipc
        
        print("Reading Arrow format...")
        
        # Read training data
        with open('/kaggle/input/race-dataset/train/data-00000-of-00001.arrow', 'rb') as f:
            reader = ipc.open_stream(f)
            train_table = reader.read_all()
            train_df = train_table.to_pandas()
        
        # Read validation data
        with open('/kaggle/input/race-dataset/validation/data-00000-of-00001.arrow', 'rb') as f:
            reader = ipc.open_stream(f)
            val_table = reader.read_all()
            val_df = val_table.to_pandas()
        
        print(f"✓ Loaded {len(train_df)} training samples")
        print(f"✓ Loaded {len(val_df)} validation samples\n")
        
    except Exception as e:
        print(f"Arrow format error: {e}")
        print("Trying CSV format...")
        train_df = pd.read_csv('/kaggle/input/race-dataset/train.csv')
        val_df = pd.read_csv('/kaggle/input/race-dataset/val.csv')
    
    # Train models
    train_model_a_multiclass(train_df, val_df, output_dir='/kaggle/working/models/model_a/traditional/')
    
    print("\n✓ Models ready for download!")
