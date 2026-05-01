# Kaggle Notebook: Model B Training - Distractor & Hint Generation
# 
# Instructions:
# 1. Create new Kaggle notebook (Python)
# 2. Add RACE dataset to workspace
# 3. Enable GPU (Settings → Accelerator)
# 4. Copy each cell below into separate Kaggle cells
# 5. Run them in order (after Model A training)
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
import re
from collections import Counter

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, hstack

# NLP imports
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec

warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

print("✓ All imports successful")


# ============================================================================
# CELL 2: Feature Engineering & Distractor Functions
# ============================================================================

class DistractorHintGenerator:
    """Generate distractors and hints for quiz questions."""
    
    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.tfidf_vectorizer = None
        self.word2vec_model = None
        self.stopwords = set(stopwords.words('english'))
        
    def tokenize_and_clean(self, text):
        """Tokenize and clean text."""
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t.isalnum() and t not in self.stopwords]
        return tokens
    
    def train_word2vec(self, texts, vector_size=100, window=5, min_count=2):
        """Train Word2Vec model on corpus."""
        sentences = [self.tokenize_and_clean(text) for text in texts]
        self.word2vec_model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4,
            sg=1  # Skip-gram
        )
        return self
    
    def get_word_vector(self, word):
        """Get Word2Vec vector for a word."""
        if self.word2vec_model is None:
            return np.zeros(100)
        try:
            return self.word2vec_model.wv[word.lower()]
        except KeyError:
            return np.zeros(100)
    
    def compute_semantic_similarity(self, text1, text2):
        """Compute semantic similarity using Word2Vec."""
        if self.word2vec_model is None:
            return 0.0
        
        tokens1 = self.tokenize_and_clean(text1)
        tokens2 = self.tokenize_and_clean(text2)
        
        if len(tokens1) == 0 or len(tokens2) == 0:
            return 0.0
        
        # Average word vectors
        vec1 = np.mean([self.get_word_vector(t) for t in tokens1], axis=0)
        vec2 = np.mean([self.get_word_vector(t) for t in tokens2], axis=0)
        
        # Cosine similarity
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def compute_lexical_similarity(self, text1, text2):
        """Compute lexical overlap."""
        words1 = set(self.tokenize_and_clean(text1))
        words2 = set(self.tokenize_and_clean(text2))
        
        if len(words1) == 0 or len(words2) == 0:
            return 0.0
        
        overlap = len(words1 & words2)
        return overlap / max(len(words1), len(words2))
    
    def rank_distractors(self, correct_answer, distractor_candidates, question, article, alpha=0.5):
        """Rank distractors by similarity to correct answer and dissimilarity to question."""
        scores = []
        
        for distractor in distractor_candidates:
            # Similarity to correct answer (should be high)
            sim_to_answer = self.compute_semantic_similarity(distractor, correct_answer)
            lex_sim_to_answer = self.compute_lexical_similarity(distractor, correct_answer)
            answer_score = (sim_to_answer + lex_sim_to_answer) / 2
            
            # Dissimilarity to question (should be low)
            sim_to_question = self.compute_semantic_similarity(distractor, question)
            question_penalty = 1 - sim_to_question  # Penalize if too similar to question
            
            # Dissimilarity to article (should be somewhat different)
            sim_to_article = self.compute_semantic_similarity(distractor, article[:200])
            article_penalty = 1 - sim_to_article
            
            # Combined score
            score = alpha * answer_score + (1 - alpha) * (question_penalty + article_penalty) / 2
            scores.append((distractor, score))
        
        # Sort by score (highest first)
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def extract_hint_candidates(self, article, correct_answer, max_hints=5):
        """Extract sentence fragments as hints."""
        sentences = sent_tokenize(article)
        hint_candidates = []
        
        for sentence in sentences:
            # Check if sentence contains keywords from correct answer
            answer_words = set(self.tokenize_and_clean(correct_answer))
            sentence_words = set(self.tokenize_and_clean(sentence))
            
            overlap = len(answer_words & sentence_words)
            if overlap > 0:
                # Extract key phrases (up to 20 chars)
                hint = sentence[:100].strip()
                if len(hint) > 10:
                    hint_candidates.append((hint, overlap))
        
        # Sort by overlap and return top
        hint_candidates.sort(key=lambda x: x[1], reverse=True)
        return [h[0] for h in hint_candidates[:max_hints]]
    
    def score_hint(self, hint, correct_answer, question):
        """Score how good a hint is."""
        # Hint should be related to answer but not directly be the answer
        sim_to_answer = self.compute_semantic_similarity(hint, correct_answer)
        sim_to_question = self.compute_semantic_similarity(hint, question)
        
        # Good hints are similar to answer but different from question
        score = sim_to_answer * (1 - sim_to_question)
        return score
    
    def save(self, path):
        """Save Word2Vec model."""
        if self.word2vec_model:
            self.word2vec_model.save(f"{path}_w2v_model")
        print(f"✓ Saved distractor/hint generator to {path}")


def prepare_distractor_dataset(df):
    """Prepare distractor dataset from wrong options."""
    records = []
    errors = {'skip_count': 0}
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preparing distractors"):
        try:
            article = str(row.get('article', ''))
            question = str(row.get('question', ''))
            answer = str(row.get('answer', '')).strip()
            
            # Get options
            options_raw = row.get('options')
            if pd.isna(options_raw):
                errors['skip_count'] += 1
                continue
            
            if isinstance(options_raw, str) and '|' in options_raw:
                options = options_raw.split('|')
            else:
                errors['skip_count'] += 1
                continue
            
            options = [str(opt).strip() for opt in options if opt]
            
            # Map answer to index
            if answer in ['A', 'B', 'C', 'D']:
                answer_idx = ord(answer) - ord('A')
            elif answer in ['0', '1', '2', '3']:
                answer_idx = int(answer)
            else:
                errors['skip_count'] += 1
                continue
            
            if answer_idx >= len(options):
                errors['skip_count'] += 1
                continue
            
            correct_option = options[answer_idx]
            
            # Collect wrong options as distractors
            wrong_options = [opt for i, opt in enumerate(options) if i != answer_idx]
            
            records.append({
                'article': article,
                'question': question,
                'correct_answer': correct_option,
                'distractor_candidates': wrong_options,
                'question_id': idx
            })
        except Exception as e:
            errors['skip_count'] += 1
            continue
    
    print(f"\n[DEBUG] Distractor dataset:")
    print(f"  Created: {len(records)} records")
    print(f"  Skipped: {errors['skip_count']} records")
    
    return pd.DataFrame(records)


print("✓ Distractor/Hint generation functions loaded")


# ============================================================================
# CELL 3: Load Dataset
# ============================================================================

import os
race_path = '/kaggle/input/datasets/ankitdhiman7/race-dataset/'
data_path = f'{race_path}train.csv'

# Load dataset
df = pd.read_csv(data_path)
print(f"✓ Loaded {len(df)} records")

# Combine A, B, C, D into options
if all(col in df.columns for col in ['A', 'B', 'C', 'D']):
    df['options'] = df[['A', 'B', 'C', 'D']].apply(lambda x: '|'.join(x.astype(str)), axis=1)
    print("✓ Created 'options' column from A, B, C, D")

# Use subset for training
df_subset = df.head(3000)
print(f"\nUsing {len(df_subset)} records for Model B training")


# ============================================================================
# CELL 4: Prepare Distractor Dataset
# ============================================================================

print("\n[Step 1] Preparing distractor dataset...")
distractor_df = prepare_distractor_dataset(df_subset)

if len(distractor_df) == 0:
    print("❌ ERROR: No distractor data created.")
else:
    print(f"✓ Prepared {len(distractor_df)} distractor records")


# ============================================================================
# CELL 5: Train Word2Vec
# ============================================================================

print("\n[Step 2] Training Word2Vec on corpus...")
import time
start_time = time.time()

# Combine all text for Word2Vec training
all_texts = (
    distractor_df['article'].tolist() +
    distractor_df['question'].tolist() +
    distractor_df['correct_answer'].tolist()
)

generator = DistractorHintGenerator(max_features=5000)
generator.train_word2vec(all_texts, vector_size=100, window=5, min_count=2)

elapsed = time.time() - start_time
print(f"✓ Word2Vec training completed in {elapsed:.2f}s")
print(f"  Vocabulary size: {len(generator.word2vec_model.wv)}")


# ============================================================================
# CELL 6: Generate & Rank Distractors
# ============================================================================

print("\n[Step 3] Generating and ranking distractors...")
import time
start_time = time.time()

distractor_results = []

with tqdm(total=len(distractor_df), desc="Ranking distractors") as pbar:
    for _, row in distractor_df.iterrows():
        try:
            candidates = row['distractor_candidates']
            
            # Rank distractors
            ranked = generator.rank_distractors(
                correct_answer=row['correct_answer'],
                distractor_candidates=candidates,
                question=row['question'],
                article=row['article'],
                alpha=0.6
            )
            
            # Get top 2 distractors
            top_distractors = [d[0] for d in ranked[:2]]
            scores = [d[1] for d in ranked[:2]]
            
            distractor_results.append({
                'question_id': row['question_id'],
                'correct_answer': row['correct_answer'],
                'distractor_1': top_distractors[0] if len(top_distractors) > 0 else '',
                'distractor_1_score': scores[0] if len(scores) > 0 else 0.0,
                'distractor_2': top_distractors[1] if len(top_distractors) > 1 else '',
                'distractor_2_score': scores[1] if len(scores) > 1 else 0.0,
            })
        except Exception as e:
            pass
        
        pbar.update(1)

elapsed = time.time() - start_time
print(f"✓ Distractor generation completed in {elapsed:.2f}s")

distractor_results_df = pd.DataFrame(distractor_results)
print(f"  Generated: {len(distractor_results_df)} distractor pairs")


# ============================================================================
# CELL 7: Extract & Score Hints
# ============================================================================

print("\n[Step 4] Extracting and scoring hints...")
start_time = time.time()

hint_results = []

with tqdm(total=len(distractor_df), desc="Extracting hints") as pbar:
    for _, row in distractor_df.iterrows():
        try:
            # Extract hint candidates
            hints = generator.extract_hint_candidates(
                article=row['article'],
                correct_answer=row['correct_answer'],
                max_hints=3
            )
            
            # Score hints
            hint_scores = []
            for hint in hints:
                score = generator.score_hint(
                    hint=hint,
                    correct_answer=row['correct_answer'],
                    question=row['question']
                )
                hint_scores.append((hint, score))
            
            # Sort by score
            hint_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Get top hint
            top_hint = hint_scores[0][0] if len(hint_scores) > 0 else ''
            top_hint_score = hint_scores[0][1] if len(hint_scores) > 0 else 0.0
            
            hint_results.append({
                'question_id': row['question_id'],
                'correct_answer': row['correct_answer'],
                'hint': top_hint,
                'hint_score': top_hint_score,
            })
        except Exception as e:
            pass
        
        pbar.update(1)

elapsed = time.time() - start_time
print(f"✓ Hint extraction completed in {elapsed:.2f}s")

hints_df = pd.DataFrame(hint_results)
print(f"  Extracted: {len(hints_df)} hints")


# ============================================================================
# CELL 8: Model B Evaluation Summary
# ============================================================================

print("\n" + "=" * 70)
print("MODEL B EVALUATION - DISTRACTOR & HINT GENERATION")
print("=" * 70)

print("\nDistractor Quality Stats:")
print(f"  Mean distractor 1 score: {distractor_results_df['distractor_1_score'].mean():.4f}")
print(f"  Mean distractor 2 score: {distractor_results_df['distractor_2_score'].mean():.4f}")

print("\nHint Quality Stats:")
print(f"  Mean hint score: {hints_df['hint_score'].mean():.4f}")
print(f"  Median hint score: {hints_df['hint_score'].median():.4f}")

print("\nSample Distractors:")
for i in range(min(3, len(distractor_results_df))):
    row = distractor_results_df.iloc[i]
    print(f"\n  Example {i+1}:")
    print(f"    Answer: {row['correct_answer']}")
    print(f"    Distractor 1: {row['distractor_1']} (score: {row['distractor_1_score']:.3f})")
    print(f"    Distractor 2: {row['distractor_2']} (score: {row['distractor_2_score']:.3f})")

print("\nSample Hints:")
for i in range(min(3, len(hints_df))):
    row = hints_df.iloc[i]
    print(f"\n  Example {i+1}:")
    print(f"    Answer: {row['correct_answer']}")
    print(f"    Hint: {row['hint']} (score: {row['hint_score']:.3f})")


# ============================================================================
# CELL 9: Save Model B Artifacts
# ============================================================================

print("\n[Final] Saving Model B artifacts...")

import os
os.makedirs('/kaggle/working/models/model_b/traditional/', exist_ok=True)

# Save models
generator.save('/kaggle/working/models/model_b/traditional/distractor_hint_generator')
joblib.dump(generator.word2vec_model, '/kaggle/working/models/model_b/traditional/word2vec_model.pkl')

# Save results
distractor_results_df.to_csv('/kaggle/working/models/model_b/traditional/distractor_results.csv', index=False)
hints_df.to_csv('/kaggle/working/models/model_b/traditional/hints_results.csv', index=False)

print("✓ All Model B artifacts saved to /kaggle/working/models/model_b/traditional/")
print("\nSaved files:")
print("  - distractor_hint_generator.pkl")
print("  - word2vec_model.pkl (also _w2v_model file)")
print("  - distractor_results.csv")
print("  - hints_results.csv")
print("\n✓ DOWNLOAD /kaggle/working/ when complete!")
