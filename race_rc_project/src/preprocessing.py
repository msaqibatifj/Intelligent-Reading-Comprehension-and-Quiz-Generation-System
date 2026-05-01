"""
Preprocessing and feature engineering for RACE dataset.
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix, hstack
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import joblib

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


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
        """Compute character-level similarity (normalized edit distance proxy)."""
        # Simple: common prefix length as fraction of max length
        i = 0
        while i < len(text1) and i < len(text2) and text1[i] == text2[i]:
            i += 1
        return i / max(len(text1), len(text2), 1)
    
    def extract_sentences(self, passage):
        """Extract sentences from passage."""
        return sent_tokenize(passage)
    
    def compute_passage_frequency(self, word, passage):
        """Compute how often a word appears in passage."""
        words = passage.lower().split()
        if len(words) == 0:
            return 0.0
        return words.count(word.lower()) / len(words)
    
    def extract_lexical_features(self, question, options, passage):
        """
        Extract handcrafted lexical features for each option.
        Returns: np.array of shape (num_options, num_features)
        """
        features_list = []
        
        for option in options:
            word_overlap = self.compute_word_overlap(question, option)
            char_match = self.compute_char_match_score(question, option)
            option_length = len(option.split())
            passage_freq = self.compute_passage_frequency(option, passage)
            
            features = [
                word_overlap,
                char_match,
                option_length,
                passage_freq
            ]
            features_list.append(features)
        
        return np.array(features_list)
    
    def fit_tfidf(self, texts):
        """Fit TF-IDF vectorizer on texts."""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            lowercase=True,
            stop_words='english'
        )
        self.tfidf_vectorizer.fit(texts)
        return self
    
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
    
    def transform_tfidf(self, texts):
        """Transform texts using fitted TF-IDF."""
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit_tfidf() first.")
        return self.tfidf_vectorizer.transform(texts)
    
    def transform_onehot(self, texts):
        """Transform texts using fitted One-Hot vectorizer."""
        if self.onehot_vectorizer is None:
            raise ValueError("One-Hot vectorizer not fitted. Call fit_onehot() first.")
        return self.onehot_vectorizer.transform(texts)
    
    def save(self, path):
        """Save fitted vectorizers and scaler."""
        joblib.dump({
            'tfidf': self.tfidf_vectorizer,
            'onehot': self.onehot_vectorizer,
            'scaler': self.scaler,
            'max_features': self.max_features
        }, path)
    
    @staticmethod
    def load(path):
        """Load saved vectorizers and scaler."""
        data = joblib.load(path)
        fe = FeatureEngineer(max_features=data['max_features'])
        fe.tfidf_vectorizer = data['tfidf']
        fe.onehot_vectorizer = data['onehot']
        fe.scaler = data['scaler']
        return fe


def load_race_dataset(csv_path):
    """
    Load RACE dataset from CSV.
    Expected columns: passage, question, options (comma-separated), answer (A/B/C/D)
    """
    df = pd.read_csv(csv_path)
    return df


def prepare_qa_dataset(df):
    """
    Prepare question-answer dataset for Model A.
    Returns: DataFrame with additional features
    """
    records = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preparing QA data"):
        passage = row['passage']
        question = row['question']
        options = row['options'].split('|')  # Assuming pipe-separated options
        answer_idx = ord(row['answer']) - ord('A')  # Convert A/B/C/D to 0/1/2/3
        correct_answer = options[answer_idx]
        
        for opt_idx, option in enumerate(options):
            label = 1 if opt_idx == answer_idx else 0
            records.append({
                'passage': passage,
                'question': question,
                'option': option,
                'label': label,
                'answer': correct_answer,
                'question_id': idx
            })
    
    return pd.DataFrame(records)


def prepare_distractor_dataset(df):
    """
    Prepare distractor dataset for Model B.
    Returns: DataFrame with correct answer vs. distractors
    """
    records = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preparing distractor data"):
        passage = row['passage']
        question = row['question']
        options = row['options'].split('|')
        answer_idx = ord(row['answer']) - ord('A')
        correct_answer = options[answer_idx]
        
        distractors = [opt for i, opt in enumerate(options) if i != answer_idx]
        
        records.append({
            'passage': passage,
            'question': question,
            'correct_answer': correct_answer,
            'distractors': distractors,
            'question_id': idx
        })
    
    return pd.DataFrame(records)


def build_feature_matrix_model_a(qa_df, feature_engineer, fit=False):
    """
    Build combined feature matrix (sparse + dense) for Model A.
    """
    # Combine question and option for semantic features
    combined_texts = qa_df['question'] + ' ' + qa_df['option']
    
    if fit:
        feature_engineer.fit_onehot(combined_texts.tolist())
    
    # One-Hot features (sparse)
    onehot_features = feature_engineer.transform_onehot(combined_texts.tolist())
    
    # Lexical features (dense)
    lexical_list = []
    for _, row in qa_df.iterrows():
        lexical = feature_engineer.extract_lexical_features(
            row['question'],
            [row['option']],
            row['passage']
        )
        lexical_list.append(lexical[0])
    
    lexical_features = np.array(lexical_list)
    
    # Combine sparse + dense
    lexical_sparse = csr_matrix(lexical_features)
    X = hstack([onehot_features, lexical_sparse])
    y = qa_df['label'].values
    
    return X, y


def build_feature_matrix_model_b(distractor_df, feature_engineer, fit=False):
    """
    Build feature matrix for Model B (distractor ranking).
    """
    features_list = []
    labels_list = []
    
    for _, row in tqdm(distractor_df.iterrows(), total=len(distractor_df), desc="Building Model B features"):
        question = row['question']
        correct_answer = row['correct_answer']
        passage = row['passage']
        distractors = row['distractors']
        
        # Correct answer: label = 1
        lexical_correct = feature_engineer.extract_lexical_features(
            question, [correct_answer], passage
        )
        features_list.append(lexical_correct[0])
        labels_list.append(1)
        
        # Distractors: label = 0
        for distractor in distractors:
            lexical_distractor = feature_engineer.extract_lexical_features(
                question, [distractor], passage
            )
            features_list.append(lexical_distractor[0])
            labels_list.append(0)
    
    X = np.array(features_list)
    y = np.array(labels_list)
    
    return X, y


if __name__ == "__main__":
    print("Preprocessing module loaded successfully.")
