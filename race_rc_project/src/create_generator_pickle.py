"""
Quick script to load existing Word2Vec model and save DistractorHintGenerator pickle
(Skips expensive retraining - just wraps the loaded model)
"""
import joblib
import os
from gensim.models import Word2Vec

# Path setup
model_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_b', 'traditional')
w2v_path = os.path.join(model_dir, 'word2vec_model.pkl')
output_path = os.path.join(model_dir, 'distractor_hint_generator.pkl')

print(f"Loading Word2Vec model from: {w2v_path}")
w2v_model = joblib.load(w2v_path)
print(f"✓ Word2Vec model loaded ({len(w2v_model.wv)} words)")

# DistractorHintGenerator class (copied from notebook)
class DistractorHintGenerator:
    def __init__(self, word2vec_model):
        self.w2v_model = word2vec_model
    
    def compute_semantic_similarity(self, text1, text2):
        words1 = text1.lower().split()
        words2 = text2.lower().split()
        
        vectors1 = [self.w2v_model.wv[w] for w in words1 if w in self.w2v_model.wv]
        vectors2 = [self.w2v_model.wv[w] for w in words2 if w in self.w2v_model.wv]
        
        if not vectors1 or not vectors2:
            return 0.0
        
        import numpy as np
        avg_vec1 = np.mean(vectors1, axis=0)
        avg_vec2 = np.mean(vectors2, axis=0)
        
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity([avg_vec1], [avg_vec2])[0][0]
        return float(similarity)
    
    def compute_lexical_similarity(self, text1, text2):
        import string
        words1 = set(text1.lower().translate(str.maketrans('', '', string.punctuation)).split())
        words2 = set(text2.lower().translate(str.maketrans('', '', string.punctuation)).split())
        
        if len(words1) == 0 or len(words2) == 0:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

# Create generator and save
print("Creating DistractorHintGenerator wrapper...")
generator = DistractorHintGenerator(w2v_model)

print(f"Saving to: {output_path}")
joblib.dump(generator, output_path)
print(f"✓ distractor_hint_generator.pkl saved successfully!")
