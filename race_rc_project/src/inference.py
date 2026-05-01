"""
Unified inference API for Model A (Q&A verification) and Model B (Distractor & Hint generation).
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import joblib
from pathlib import Path
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class ModelAInference:
    """Inference for Question & Answer Generator/Verifier."""
    
    def __init__(self, model_paths):
        """
        model_paths: dict with keys like 'lr', 'svm', 'rf', 'ensemble'
        """
        self.models = {}
        self.feature_engineer = None
        self.load_models(model_paths)
    
    def load_models(self, model_paths):
        """Load trained models from disk."""
        for model_name, path in model_paths.items():
            try:
                self.models[model_name] = joblib.load(path)
                print(f"✓ Loaded {model_name} from {path}")
            except FileNotFoundError:
                print(f"⚠ Model {model_name} not found at {path}")
    
    def verify_answer(self, passage, question, option, method='ensemble'):
        """
        Verify if an option is correct given passage and question.
        Returns: (is_correct: bool, confidence: float, explanation: str)
        """
        if method not in self.models:
            return False, 0.0, f"Model {method} not loaded."
        
        model = self.models[method]
        
        # Feature extraction would happen here
        # For now, mock implementation
        try:
            confidence = model.predict_proba([[0.5, 0.3, 0.2, 0.1]])[0, 1]
            is_correct = confidence > 0.5
            explanation = f"Model confidence: {confidence:.2%}"
            return is_correct, confidence, explanation
        except Exception as e:
            return False, 0.0, f"Error: {str(e)}"
    
    def generate_question(self, passage, method='template'):
        """
        Generate a question from passage.
        Returns: (question: str, answer: str)
        """
        sentences = sent_tokenize(passage)
        if not sentences:
            return "", ""
        
        # Mock: return first sentence as question, dummy answer
        question = f"What is the main idea of: {sentences[0][:50]}...?"
        answer = sentences[0]
        
        return question, answer


class ModelBInference:
    """Inference for Distractor & Hint Generator."""
    
    def __init__(self, model_paths):
        """
        model_paths: dict with keys like 'distractor_ranker', 'hint_scorer'
        """
        self.models = {}
        self.feature_engineer = None
        self.load_models(model_paths)
    
    def load_models(self, model_paths):
        """Load trained models from disk."""
        for model_name, path in model_paths.items():
            try:
                self.models[model_name] = joblib.load(path)
                print(f"✓ Loaded {model_name} from {path}")
            except FileNotFoundError:
                print(f"⚠ Model {model_name} not found at {path}")
    
    def generate_distractors(self, passage, question, correct_answer, num_distractors=3):
        """
        Generate plausible but incorrect distractors.
        Returns: list of strings (distractors)
        """
        # Extract candidate phrases from passage
        candidates = self._extract_candidates(passage, correct_answer)
        
        if len(candidates) < num_distractors:
            # Fallback: pad with similar-length gibberish
            candidates.extend([f"Option {i}" for i in range(num_distractors)])
        
        # Rank candidates using distractor ranker
        if 'distractor_ranker' in self.models:
            scores = self._score_distractors(candidates, question, correct_answer)
            sorted_idx = np.argsort(scores)[::-1]  # Sort descending
            distractors = [candidates[i] for i in sorted_idx[:num_distractors]]
        else:
            distractors = candidates[:num_distractors]
        
        return distractors
    
    def _extract_candidates(self, passage, correct_answer, top_k=20):
        """Extract candidate phrases from passage (excluding correct answer)."""
        words = passage.split()
        candidates = []
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            if phrase != correct_answer and len(phrase) > 3:
                candidates.append(phrase)
        return candidates[:top_k]
    
    def _score_distractors(self, candidates, question, correct_answer):
        """Score distractors using trained model."""
        scores = np.random.rand(len(candidates))  # Mock scoring
        return scores
    
    def generate_hints(self, passage, question, correct_answer, num_hints=3):
        """
        Generate graduated hints (vague → specific) without revealing answer.
        Returns: list of strings (hints in order of specificity)
        """
        sentences = sent_tokenize(passage)
        
        if not sentences:
            return ["No hints available."]
        
        # Compute cosine similarity between sentences and question
        hint_scores = []
        for sent in sentences:
            # Mock: simple word overlap as proxy
            score = len(set(question.lower().split()) & set(sent.lower().split()))
            hint_scores.append((sent, score))
        
        # Sort by score and return top hints
        hint_scores.sort(key=lambda x: x[1], reverse=True)
        hints = [sent for sent, _ in hint_scores[:num_hints]]
        
        # Ensure hints don't directly contain the answer
        hints = [hint for hint in hints if correct_answer.lower() not in hint.lower()][:num_hints]
        
        if len(hints) < num_hints:
            hints.extend([f"Hint {i+1}" for i in range(num_hints - len(hints))])
        
        return hints[:num_hints]


class UnifiedInference:
    """Unified inference for both Model A and Model B."""
    
    def __init__(self, model_a_paths, model_b_paths):
        self.model_a = ModelAInference(model_a_paths)
        self.model_b = ModelBInference(model_b_paths)
    
    def generate_and_verify_mcq(self, passage):
        """
        End-to-end: generate question, answer, distractors, hints.
        Returns: dict with question, correct_answer, distractors, hints
        """
        # Generate question
        question, answer = self.model_a.generate_question(passage)
        
        # Generate distractors
        distractors = self.model_b.generate_distractors(passage, question, answer)
        
        # Generate hints
        hints = self.model_b.generate_hints(passage, question, answer, num_hints=3)
        
        return {
            'question': question,
            'correct_answer': answer,
            'distractors': distractors,
            'options': [answer] + distractors,  # Shuffle this in UI
            'hints': hints
        }
    
    def verify_user_answer(self, passage, question, correct_answer, user_option):
        """
        Verify user's selected option.
        Returns: dict with is_correct, confidence, explanation
        """
        is_correct, confidence, explanation = self.model_a.verify_answer(
            passage, question, user_option
        )
        
        return {
            'is_correct': is_correct,
            'confidence': confidence,
            'explanation': explanation,
            'correct_answer': correct_answer
        }


if __name__ == "__main__":
    print("Inference module loaded successfully.")
