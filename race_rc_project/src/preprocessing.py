"""
Preprocessing and feature engineering for RACE dataset.
"""
import os
import re

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix, hstack, save_npz
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import joblib


# Local: run from the repo root or the notebook folder.
# Kaggle: set the notebook working directory to the uploaded project folder.
BASE_DIR = os.getcwd()
if not os.path.isdir(os.path.join(BASE_DIR, "data")):
    parent_dir = os.path.dirname(BASE_DIR)
    if os.path.isdir(os.path.join(parent_dir, "data")):
        BASE_DIR = parent_dir
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RAW_DIR = os.path.join(DATA_DIR, "raw")

# Relative paths
RAW_DATA_PATH = os.path.join(RAW_DIR, "train.csv")
PROCESSED_OUT_DIR = PROCESSED_DIR

# Set to None to use the full dataset.
# Set to an integer to limit preprocessing to the first N rows.
DATASET_ROW_LIMIT = None


def _limit_dataframe(df, max_rows=None):
    """Return a row-limited copy of *df* when a limit is provided."""
    limit = DATASET_ROW_LIMIT if max_rows is None else max_rows
    if limit is None:
        return df
    if limit <= 0:
        return df.iloc[0:0].copy()
    return df.head(limit).copy()

def clean_text(text):
    """Lowercase text and collapse non-alphanumeric characters into spaces."""
    text = "" if text is None else str(text)
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    """Tokenize using the shared text normalisation used by the training code."""
    cleaned = clean_text(text)
    return cleaned.split() if cleaned else []


def split_into_sentences(passage):
    """Split text into sentences with a regex fallback if punkt is unavailable."""
    passage = "" if passage is None else str(passage)
    if not passage.strip():
        return []
    try:
        sentences = sent_tokenize(passage)
    except LookupError:
        sentences = re.split(r"(?<=[.!?])\s+", passage)
    return [sentence.strip() for sentence in sentences if sentence and sentence.strip()]


def cosine_similarity_feature(text1, text2):
    """Return a simple cosine-style overlap score for two texts."""
    text_a = clean_text(text1)
    text_b = clean_text(text2)
    if not text_a or not text_b:
        return 0.0
    vectorizer = CountVectorizer(binary=True)
    matrix = vectorizer.fit_transform([text_a, text_b])
    return float(cosine_similarity(matrix[0], matrix[1])[0][0])


def tfidf_cosine_similarity(text1, text2, vectorizer):
    """Compute cosine similarity between two texts using a fitted TF-IDF/Count vectorizer."""
    if vectorizer is None:
        return cosine_similarity_feature(text1, text2)
    vecs = vectorizer.transform([str(text1), str(text2)])
    if vecs.shape[1] == 0:
        return 0.0
    return float(cosine_similarity(vecs[0], vecs[1])[0][0])


def batch_tfidf_cosine(texts_a, texts_b, vectorizer):
    """Batch cosine scores for paired text lists."""
    return [tfidf_cosine_similarity(a, b, vectorizer) for a, b in zip(texts_a, texts_b)]


def build_one_sample(article, question, option):
    """Build the text sample used by the answer-verification models."""
    parts = [clean_text(article), clean_text(question), clean_text(option)]
    return " ".join(part for part in parts if part)


def load_race(csv_path):
    """Load a RACE-format CSV and normalise common column variants."""
    df = pd.read_csv(csv_path)
    df = _limit_dataframe(df)
    if "article" not in df.columns and "passage" in df.columns:
        df = df.rename(columns={"passage": "article"})
    if "answer" in df.columns and df["answer"].dtype != object:
        df["answer"] = df["answer"].map({0: "A", 1: "B", 2: "C", 3: "D"}).fillna(df["answer"])
    if "options" in df.columns and not {"A", "B", "C", "D"}.issubset(df.columns):
        expanded = df["options"].astype(str).str.split("|")
        for idx, label in enumerate(["A", "B", "C", "D"]):
            df[label] = expanded.str.get(idx)
    return df

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
        words1 = set(tokenize(text1))
        words2 = set(tokenize(text2))
        if len(words1) == 0 or len(words2) == 0:
            return 0.0
        overlap = len(words1 & words2)
        return overlap / max(len(words1), len(words2))
    
    def compute_char_match_score(self, text1, text2):
        """Compute character-level similarity (normalized edit distance proxy)."""
        # Simple: common prefix length as fraction of max length
        text1 = clean_text(text1)
        text2 = clean_text(text2)
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
        question_tokens = tokenize(question)
        passage_tokens = tokenize(passage)
        passage_token_set = set(passage_tokens)
        question_token_set = set(question_tokens)
        passage_len = max(len(passage_tokens), 1)
        
        for option in options:
            option_tokens = tokenize(option)
            option_token_set = set(option_tokens)

            word_overlap = self.compute_word_overlap(question, option)
            passage_overlap = len(option_token_set & passage_token_set) / max(len(option_token_set), 1)
            question_passage_overlap = len(question_token_set & passage_token_set) / max(len(question_token_set), 1)

            char_match = self.compute_char_match_score(question, option)
            passage_char_match = self.compute_char_match_score(passage, option)

            option_length = len(option.split())
            passage_freq = self.compute_passage_frequency(option, passage)
            length_ratio = min(option_length / max(len(question.split()), 1), 5.0)
            
            features = [
                word_overlap,
                passage_overlap,
                question_passage_overlap,
                char_match,
                passage_char_match,
                option_length,
                passage_freq
                ,length_ratio
            ]
            features_list.append(features)
        
        return np.array(features_list)
    
    def fit_tfidf(self, texts):
        """Fit TF-IDF vectorizer on texts."""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.tfidf_vectorizer.fit(texts)
        return self
    
    def fit_onehot(self, texts):
        """Fit One-Hot (CountVectorizer binary) on texts."""
        self.onehot_vectorizer = CountVectorizer(
            max_features=self.max_features,
            lowercase=True,
            binary=True,
            stop_words='english',
            ngram_range=(1, 2),
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
    df = _limit_dataframe(df)

    if 'passage' not in df.columns and 'article' in df.columns:
        df = df.rename(columns={'article': 'passage'})

    if 'options' not in df.columns and {'A', 'B', 'C', 'D'}.issubset(df.columns):
        df['options'] = df[['A', 'B', 'C', 'D']].astype(str).agg('|'.join, axis=1)

    return df


def prepare_qa_dataset(df):
    """
    Prepare question-answer dataset for Model A.
    Returns: DataFrame with additional features
    """
    df = _limit_dataframe(df)
    records = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preparing QA data"):
        passage = str(row.get('passage', row.get('article', '')))
        question = str(row.get('question', ''))

        if 'options' in row and pd.notna(row.get('options', None)):
            options = str(row['options']).split('|')
        else:
            options = [str(row.get(label, '')) for label in ['A', 'B', 'C', 'D']]

        answer_label = str(row.get('answer', '')).strip().upper()
        if answer_label not in ['A', 'B', 'C', 'D']:
            answer_label = 'A'
        answer_idx = ord(answer_label) - ord('A')  # Convert A/B/C/D to 0/1/2/3
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
    df = _limit_dataframe(df)
    records = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preparing distractor data"):
        passage = str(row.get('passage', row.get('article', '')))
        question = str(row.get('question', ''))

        if 'options' in row and pd.notna(row.get('options', None)):
            options = str(row['options']).split('|')
        else:
            options = [str(row.get(label, '')) for label in ['A', 'B', 'C', 'D']]

        answer_label = str(row.get('answer', '')).strip().upper()
        if answer_label not in ['A', 'B', 'C', 'D']:
            answer_label = 'A'
        answer_idx = ord(answer_label) - ord('A')
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


def run_preprocessing_pipeline(raw_csv_path=RAW_DATA_PATH):
    """Build and save processed datasets and vectorizers from the raw CSV."""
    df = load_race_dataset(raw_csv_path)

    if df.empty:
        raise ValueError("Raw dataset is empty.")

    if 'answer' in df.columns:
        stratify_target = df['answer'].astype(str).str.upper()
        class_counts = stratify_target.value_counts(dropna=False)
        can_stratify = len(df) >= 20 and class_counts.min() >= 2
    else:
        stratify_target = None
        can_stratify = False

    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=stratify_target if can_stratify else None,
    )

    if can_stratify:
        temp_stratify = temp_df['answer'].astype(str).str.upper()
    else:
        temp_stratify = None

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_stratify if temp_stratify is not None else None,
    )

    split_frames = {
        'train': train_df.reset_index(drop=True),
        'val': val_df.reset_index(drop=True),
        'test': test_df.reset_index(drop=True),
    }

    for name, frame in split_frames.items():
        frame.to_csv(os.path.join(PROCESSED_DIR, f'{name}_clean.csv'), index=False)

    feature_engineer = FeatureEngineer(max_features=10000)

    train_qa = prepare_qa_dataset(split_frames['train'])
    val_qa = prepare_qa_dataset(split_frames['val'])
    test_qa = prepare_qa_dataset(split_frames['test'])

    X_train, y_train = build_feature_matrix_model_a(train_qa, feature_engineer, fit=True)
    X_val, y_val = build_feature_matrix_model_a(val_qa, feature_engineer, fit=False)
    X_test, y_test = build_feature_matrix_model_a(test_qa, feature_engineer, fit=False)

    save_npz(os.path.join(PROCESSED_DIR, 'X_train_ohe.npz'), X_train)
    save_npz(os.path.join(PROCESSED_DIR, 'X_val_ohe.npz'), X_val)
    save_npz(os.path.join(PROCESSED_DIR, 'X_test_ohe.npz'), X_test)

    np.save(os.path.join(PROCESSED_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_DIR, 'y_val.npy'), y_val)
    np.save(os.path.join(PROCESSED_DIR, 'y_test.npy'), y_test)

    hc_train = np.vstack(train_qa.apply(
        lambda row: feature_engineer.extract_lexical_features(
            row['question'], [row['option']], row['passage']
        )[0],
        axis=1,
    ).to_list())
    hc_val = np.vstack(val_qa.apply(
        lambda row: feature_engineer.extract_lexical_features(
            row['question'], [row['option']], row['passage']
        )[0],
        axis=1,
    ).to_list())
    hc_test = np.vstack(test_qa.apply(
        lambda row: feature_engineer.extract_lexical_features(
            row['question'], [row['option']], row['passage']
        )[0],
        axis=1,
    ).to_list())

    np.save(os.path.join(PROCESSED_DIR, 'hc_train.npy'), hc_train)
    np.save(os.path.join(PROCESSED_DIR, 'hc_val.npy'), hc_val)
    np.save(os.path.join(PROCESSED_DIR, 'hc_test.npy'), hc_test)

    feature_engineer.save(os.path.join(PROCESSED_DIR, 'feature_engineer.pkl'))
    joblib.dump(feature_engineer.onehot_vectorizer, os.path.join(PROCESSED_DIR, 'ohe_vectorizer.pkl'))
    joblib.dump(feature_engineer.tfidf_vectorizer, os.path.join(PROCESSED_DIR, 'tfidf_vectorizer.pkl'))

    return {
        'train_rows': len(split_frames['train']),
        'val_rows': len(split_frames['val']),
        'test_rows': len(split_frames['test']),
        'qa_train_rows': len(train_qa),
        'qa_val_rows': len(val_qa),
        'qa_test_rows': len(test_qa),
    }


def build_feature_matrix_model_a(qa_df, feature_engineer, fit=False):
    """
    Build combined feature matrix (sparse + dense) for Model A.
    """
    # Combine passage, question, and option for semantic features.
    # This gives the model the same context it will see at inference time.
    combined_texts = qa_df['passage'] + ' [SEP] ' + qa_df['question'] + ' [SEP] ' + qa_df['option']
    
    if fit:
        feature_engineer.fit_onehot(combined_texts.tolist())
        feature_engineer.fit_tfidf(combined_texts.tolist())
    
    # One-Hot features (sparse)
    onehot_features = feature_engineer.transform_onehot(combined_texts.tolist())

    # TF-IDF features (sparse)
    tfidf_features = feature_engineer.transform_tfidf(combined_texts.tolist())
    
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
    X = hstack([onehot_features, tfidf_features, lexical_sparse])
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
    summary = run_preprocessing_pipeline()
    print("Preprocessing complete.")
    print(summary)
