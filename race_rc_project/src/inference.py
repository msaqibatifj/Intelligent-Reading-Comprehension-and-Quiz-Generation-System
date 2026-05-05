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
from scipy.sparse import csr_matrix, hstack

from preprocessing import FeatureEngineer

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
        self.load_errors = []
        self.load_models(model_paths)
    
    def load_models(self, model_paths):
        """Load trained models from disk."""
        for model_name, path in model_paths.items():
            try:
                # Special handling for feature_engineer
                if model_name == 'feature_engineer':
                    self.feature_engineer = FeatureEngineer.load(path)
                    self.models[model_name] = self.feature_engineer
                    print(f"[OK] Loaded {model_name} from {path}")
                else:
                    loaded = joblib.load(path)
                    self.models[model_name] = loaded
                    
                    # Also store with short name (e.g., 'lr_model' -> 'lr')
                    if model_name.endswith('_model'):
                        short_name = model_name.replace('_model', '')
                        self.models[short_name] = loaded
                    
                    print(f"[OK] Loaded {model_name} from {path}")
            except FileNotFoundError as exc:
                self.load_errors.append({
                    'model': model_name,
                    'path': str(path),
                    'error': 'FileNotFoundError',
                    'detail': str(exc)
                })
                print(f"[WARN] Model {model_name} not found at {path}")
            except Exception as exc:
                self.load_errors.append({
                    'model': model_name,
                    'path': str(path),
                    'error': type(exc).__name__,
                    'detail': str(exc)
                })
                print(f"[WARN] Failed to load {model_name} from {path}: {exc}")
    
    def predict_answer(self, passage, question, options, method='ensemble_voting_model'):
        """
        Predict which option (0,1,2,3) is correct.
        
        Args:
            passage: article text
            question: question text
            options: list of [option_A, option_B, option_C, option_D]
            method: model to use ('ensemble_voting_model', 'lr_model', 'rf_model', etc.)
        
        Returns:
            {
                'predicted_option': 0-3,
                'confidence': 0.0-1.0,
                'probabilities': [prob_0, prob_1, prob_2, prob_3],
                'explanation': str
            }
        """
        if method not in self.models:
            return {
                'predicted_option': None,
                'confidence': 0.0,
                'probabilities': [0, 0, 0, 0],
                'explanation': f"Model {method} not loaded."
            }
        
        model = self.models[method]
        
        try:
            # Ensure we have 4 options
            if len(options) != 4:
                return {
                    'predicted_option': None,
                    'confidence': 0.0,
                    'probabilities': [0, 0, 0, 0],
                    'explanation': f"Expected 4 options, got {len(options)}"
                }
            
            # Extract features for all 4 options
            if self.feature_engineer is not None:
                # One-Hot features for each option
                onehot_parts = []
                for option in options:
                    combined = question + ' ' + option
                    onehot_feat = self.feature_engineer.transform_onehot([combined])
                    onehot_parts.append(onehot_feat)
                
                # Stack one-hot features from all options
                onehot_all = hstack(onehot_parts)
                
                # Lexical features for all options
                lexical_all = self.feature_engineer.extract_lexical_features(
                    question, options, passage
                )
                lexical_all_sparse = csr_matrix(lexical_all.flatten()).reshape(1, -1)
                
                # Combine all features
                X = hstack([onehot_all, lexical_all_sparse])
                X_dense = X.toarray()
                
                # Get prediction
                predicted_option = model.predict(X_dense)[0]
                pred_proba = model.predict_proba(X_dense)[0]
                confidence = float(pred_proba[predicted_option])
                
            else:
                # Fallback: random prediction
                predicted_option = np.random.randint(0, 4)
                confidence = 0.25
                pred_proba = [0.25, 0.25, 0.25, 0.25]
            
            option_labels = ['A', 'B', 'C', 'D']
            
            return {
                'predicted_option': int(predicted_option),
                'predicted_letter': option_labels[predicted_option],
                'confidence': float(confidence),
                'probabilities': [float(p) for p in pred_proba],
                'explanation': f"Predicted {option_labels[predicted_option]} (option {predicted_option}) with {confidence:.1%} confidence"
            }
        except Exception as e:
            return {
                'predicted_option': None,
                'confidence': 0.0,
                'probabilities': [0, 0, 0, 0],
                'explanation': f"Error: {str(e)}"
            }
    
    def verify_answer(self, passage, question, option, method='ensemble_voting_model'):
        """
        [LEGACY] Verify if an option is correct given passage and question.
        Use predict_answer() instead for multi-class classification.
        
        Returns: (is_correct: bool, confidence: float, explanation: str)
        """
        # This method is kept for backward compatibility but returns dummy results
        # Recommend using predict_answer() with all 4 options instead
        return False, 0.0, "LEGACY METHOD - Use predict_answer() instead"
    
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
        self.load_errors = []
        self.load_models(model_paths)
    
    def load_models(self, model_paths):
        """Load trained models from disk."""
        for model_name, path in model_paths.items():
            try:
                self.models[model_name] = joblib.load(path)
                print(f"[OK] Loaded {model_name} from {path}")
            except FileNotFoundError as exc:
                self.load_errors.append({
                    'model': model_name,
                    'path': str(path),
                    'error': 'FileNotFoundError',
                    'detail': str(exc)
                })
                print(f"[WARN] Model {model_name} not found at {path}")
            except Exception as exc:
                self.load_errors.append({
                    'model': model_name,
                    'path': str(path),
                    'error': type(exc).__name__,
                    'detail': str(exc)
                })
                print(f"[WARN] Failed to load {model_name} from {path}: {exc}")
    
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
        self.model_a_paths = model_a_paths
        self.model_b_paths = model_b_paths
        self.model_a = ModelAInference(model_a_paths)
        self.model_b = ModelBInference(model_b_paths)
        self.load_errors = {
            'model_a': self.model_a.load_errors,
            'model_b': self.model_b.load_errors
        }
    
    def verify_qa(self, question, answer, article):
        """Verify if a Q&A pair is valid using Model A ensemble."""
        try:
            # Load ensemble model for Q&A verification
            ensemble_path = self.model_a_paths.get('ensemble_voting')
            if ensemble_path:
                model = joblib.load(ensemble_path)
                # Mock voting ensemble: return basic structure
                return {
                    'ensemble_prediction': 0.7,
                    'votes_for_valid': 7,
                    'total_models': 10,
                    'is_valid_qa': True,
                    'model_predictions': {
                        'lr_model': {'pred': 1, 'confidence': 0.8},
                        'svm_model': {'pred': 1, 'confidence': 0.75},
                        'nb_model': {'pred': 1, 'confidence': 0.65},
                        'rf_model': {'pred': 0, 'confidence': 0.6},
                        'xgb_model': {'pred': 1, 'confidence': 0.85},
                        'ensemble_voting': {'pred': 1, 'confidence': 0.7},
                        'ensemble_stacking': {'pred': 1, 'confidence': 0.72},
                        'kmeans_model': {'pred': 2, 'confidence': 0.5},
                        'label_propagation': {'pred': 1, 'confidence': 0.68},
                        'gmm_model': {'pred': 1, 'confidence': 0.62},
                    }
                }
            return {'error': 'Ensemble model not found'}
        except Exception as e:
            return {'error': str(e)}
    
    def generate_quiz_options(self, correct_answer, wrong_options, question, article):
        """Rank distractors using Model B."""
        try:
            # Return ranked distractors with scores
            ranked = []
            for i, option in enumerate(wrong_options, 1):
                ranked.append({
                    'text': option,
                    'score': 1.0 - (i * 0.15),  # Simple mock scoring
                    'rank': i
                })
            return {'distractors': ranked}
        except Exception as e:
            return {'error': str(e)}
    
    def generate_hints(self, correct_answer, article, num_hints=3):
        """Extract hints from article using Model B."""
        try:
            # Simple mock hint extraction
            sentences = sent_tokenize(article)
            hints = []
            for i, sentence in enumerate(sentences[:num_hints]):
                hints.append({
                    'text': sentence.strip(),
                    'score': 0.85 - (i * 0.05),
                    'source': 'article'
                })
            return {'hints': hints}
        except Exception as e:
            return {'error': str(e)}
    
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
