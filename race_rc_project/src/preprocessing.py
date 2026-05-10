import math
import os
import pickle
import re
import string
from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# PATHS (hard-coded project data layout)
# ─────────────────────────────────────────────

DATA_ROOT = r"C:\Users\mosaq\Desktop\AI Proj\race_rc_project\data"
BASE_DIR = r"C:\Users\mosaq\Desktop\AI Proj\race_rc_project"
RAW_DIR = os.path.join(DATA_ROOT, "raw")
PROCESSED_DIR = os.path.join(DATA_ROOT, "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

DATA_SIZE = None  # set number for debugging

# ─────────────────────────────────────────────
# SAFE CLEANING
# ─────────────────────────────────────────────

def clean_text(text):
    if text is None:
        return ""

    if isinstance(text, float) and np.isnan(text):
        return ""

    text = str(text).lower()
    text = text.replace("nan", "")
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    return text


def safe_str(x):
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    x = str(x).strip()
    if x.lower() == "nan":
        return ""
    return x


# ─────────────────────────────────────────────
# SENTENCE SPLITTER (FIXED + REQUIRED)
# ─────────────────────────────────────────────

def split_into_sentences(text):
    if not isinstance(text, str):
        text = str(text)

    text = text.replace("\n", " ")
    sentences = re.split(r"(?<=[.!?])\s+", text)

    return [s.strip() for s in sentences if len(s.strip()) > 5]


# ─────────────────────────────────────────────
# TOKENIZER
# ─────────────────────────────────────────────

STOPWORDS = set([
    "a","an","the","is","it","in","on","at","to","for","of",
    "and","or","but","this","that","are","was","were","be"
])

def tokenize(text):
    text = clean_text(text)
    return [w for w in text.split() if w not in STOPWORDS and len(w) > 1]


# ─────────────────────────────────────────────
# LOAD DATA (RACE FORMAT SAFE)
# ─────────────────────────────────────────────

def load_data():
    path = os.path.join(RAW_DIR, "train.csv")

    df = pd.read_csv(path, nrows=DATA_SIZE)

    # SAFE CLEAN ALL COLUMNS
    for col in ["article", "question", "A", "B", "C", "D", "answer"]:
        if col in df.columns:
            df[col] = df[col].apply(safe_str)

    # REMOVE BAD ROWS
    df = df[
        (df["article"].str.len() > 20) &
        (df["question"].str.len() > 3)
    ]

    df = df[df["answer"].isin(["A", "B", "C", "D"])]

    df = df.reset_index(drop=True)

    train, temp = train_test_split(df, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    return {"train": train, "val": val, "test": test}


# ─────────────────────────────────────────────
# TF-IDF (MODEL A CORE FEATURE)
# ─────────────────────────────────────────────

def build_tfidf_vectorizer(texts):
    vec = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1,2),
        stop_words="english"
    )
    vec.fit(texts)
    return vec


def tfidf_cosine(a, b, vec):
    a = clean_text(a)
    b = clean_text(b)

    if not a or not b:
        return 0.0

    v = vec.transform([a, b])
    return float(cosine_similarity(v[0], v[1])[0][0])


def tfidf_cosine_similarity(text_a, text_b, vectorizer):
    """Alias expected by model_a_train (same as tfidf_cosine)."""
    return tfidf_cosine(text_a, text_b, vectorizer)


def cosine_similarity_feature(text_a, text_b):
    a, b = Counter(tokenize(text_a)), Counter(tokenize(text_b))
    if not a or not b:
        return 0.0
    dot = sum(a[k] * b[k] for k in a if k in b)
    denom = math.sqrt(sum(v**2 for v in a.values())) * math.sqrt(
        sum(v**2 for v in b.values())
    ) + 1e-9
    return dot / denom


def build_one_sample(article, question, option):
    return clean_text(f"{article} {question} {option}")


def encode_texts(texts, vectorizer):
    return vectorizer.transform(texts)


# ─────────────────────────────────────────────
# MODEL A DATASET BUILDER (FIXED)
# ─────────────────────────────────────────────

def build_model_a_dataset(df):
    texts = []
    labels = []

    for _, row in df.iterrows():
        gold = str(row["answer"]).strip().upper()

        for opt in ["A", "B", "C", "D"]:

            opt_text = clean_text(row[opt])

            if opt_text == "":
                continue

            combined = (
                clean_text(row["article"]) + " " +
                clean_text(row["question"]) + " " +
                opt_text
            )

            texts.append(combined)
            labels.append(1 if opt == gold else 0)

    return texts, labels


# ─────────────────────────────────────────────
# OHE MATRICES + VECTORIZERS (Model A / B on disk)
# ─────────────────────────────────────────────

def build_training_artifacts(processed_dir, max_ohe_features=10000):
    """Sparse OHE features, labels, ohe_vectorizer.pkl; tfidf.pkl to tfidf_vectorizer.pkl."""
    for split in ("train", "val", "test"):
        path = os.path.join(processed_dir, f"{split}_clean.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}; run run_preprocessing() first."
            )

    train_df = pd.read_csv(os.path.join(processed_dir, "train_clean.csv"))
    val_df = pd.read_csv(os.path.join(processed_dir, "val_clean.csv"))
    test_df = pd.read_csv(os.path.join(processed_dir, "test_clean.csv"))

    tr_texts, tr_y = build_model_a_dataset(train_df)
    va_texts, va_y = build_model_a_dataset(val_df)
    te_texts, te_y = build_model_a_dataset(test_df)

    vec = CountVectorizer(max_features=max_ohe_features, ngram_range=(1, 2))
    vec.fit(tr_texts)

    X_tr = vec.transform(tr_texts)
    X_va = vec.transform(va_texts)
    X_te = vec.transform(te_texts)

    y_tr = np.asarray(tr_y, dtype=np.int64)
    y_va = np.asarray(va_y, dtype=np.int64)
    y_te = np.asarray(te_y, dtype=np.int64)

    save_npz(os.path.join(processed_dir, "X_train_ohe.npz"), X_tr)
    save_npz(os.path.join(processed_dir, "X_val_ohe.npz"), X_va)
    save_npz(os.path.join(processed_dir, "X_test_ohe.npz"), X_te)
    np.save(os.path.join(processed_dir, "y_train.npy"), y_tr)
    np.save(os.path.join(processed_dir, "y_val.npy"), y_va)
    np.save(os.path.join(processed_dir, "y_test.npy"), y_te)

    for name, n in ("train", X_tr.shape[0]), ("val", X_va.shape[0]), ("test", X_te.shape[0]):
        np.save(
            os.path.join(processed_dir, f"hc_{name}.npy"),
            np.zeros((n, 0), dtype=np.float32),
        )

    with open(os.path.join(processed_dir, "ohe_vectorizer.pkl"), "wb") as f:
        pickle.dump(vec, f)

    tfidf_pkl = os.path.join(processed_dir, "tfidf.pkl")
    tfidf_out = os.path.join(processed_dir, "tfidf_vectorizer.pkl")
    if os.path.exists(tfidf_pkl):
        with open(tfidf_pkl, "rb") as f:
            tfidf = pickle.load(f)
        with open(tfidf_out, "wb") as f:
            pickle.dump(tfidf, f)
    elif not os.path.exists(tfidf_out):
        raise FileNotFoundError(
            f"Expected {tfidf_pkl} or existing {tfidf_out} for model_b_train."
        )


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_preprocessing(max_ohe_features=10000):

    print("Loading data...")
    splits = load_data()

    train = splits["train"]
    val = splits["val"]
    test = splits["test"]

    print("Building TF-IDF...")
    tfidf = build_tfidf_vectorizer(train["article"].astype(str))

    print("Saving cleaned datasets...")

    train.to_csv(os.path.join(PROCESSED_DIR, "train_clean.csv"), index=False)
    val.to_csv(os.path.join(PROCESSED_DIR, "val_clean.csv"), index=False)
    test.to_csv(os.path.join(PROCESSED_DIR, "test_clean.csv"), index=False)

    with open(os.path.join(PROCESSED_DIR, "tfidf.pkl"), "wb") as f:
        pickle.dump(tfidf, f)

    print("Building OHE matrices and tfidf_vectorizer.pkl …")
    build_training_artifacts(PROCESSED_DIR, max_ohe_features=max_ohe_features)

    print("\nDONE — PREPROCESSING SUCCESSFUL (MODEL A READY)")


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_preprocessing()