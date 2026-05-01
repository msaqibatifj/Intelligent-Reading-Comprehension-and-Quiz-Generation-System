"""
Integration tests for Model A and Model B inference.
Run with: pytest tests/test_inference.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from src.inference import ModelAInference, ModelBInference, UnifiedInference


class TestModelAInference:
    """Test Model A (Q&A Verification) inference."""
    
    def test_model_a_initialization(self):
        """Test that Model A can be initialized."""
        model_paths = {
            'lr': 'models/model_a/traditional/lr_model.pkl',
        }
        model_a = ModelAInference(model_paths)
        assert model_a is not None
    
    def test_verify_answer_basic(self):
        """Test basic answer verification."""
        model_paths = {
            'ensemble': 'models/model_a/traditional/ensemble_model.pkl',
        }
        model_a = ModelAInference(model_paths)
        
        passage = "The capital of France is Paris."
        question = "What is the capital of France?"
        option = "Paris"
        
        is_correct, confidence, explanation = model_a.verify_answer(
            passage, question, option
        )
        
        # Even if model isn't loaded, should return sensible values
        assert isinstance(is_correct, (bool, np.bool_))
        assert isinstance(confidence, (float, np.floating))
        assert isinstance(explanation, str)
    
    def test_generate_question(self):
        """Test question generation."""
        model_paths = {}
        model_a = ModelAInference(model_paths)
        
        passage = "Alice fell down the rabbit hole and found herself in a strange land."
        question, answer = model_a.generate_question(passage)
        
        assert isinstance(question, str)
        assert isinstance(answer, str)
        assert len(question) > 0
        assert len(answer) > 0


class TestModelBInference:
    """Test Model B (Distractor & Hint) inference."""
    
    def test_model_b_initialization(self):
        """Test that Model B can be initialized."""
        model_paths = {
            'distractor_ranker': 'models/model_b/traditional/distractor_ranker.pkl',
        }
        model_b = ModelBInference(model_paths)
        assert model_b is not None
    
    def test_generate_distractors(self):
        """Test distractor generation."""
        model_paths = {}
        model_b = ModelBInference(model_paths)
        
        passage = "The United Nations was established in 1945. It replaced the League of Nations."
        question = "When was the United Nations established?"
        correct_answer = "1945"
        
        distractors = model_b.generate_distractors(
            passage, question, correct_answer, num_distractors=3
        )
        
        assert isinstance(distractors, list)
        assert len(distractors) == 3
        for distractor in distractors:
            assert isinstance(distractor, str)
    
    def test_generate_hints(self):
        """Test hint generation."""
        model_paths = {}
        model_b = ModelBInference(model_paths)
        
        passage = "The Great Wall of China is one of the most famous monuments in the world. It was built to protect ancient Chinese states from invasions."
        question = "What was the Great Wall of China built for?"
        correct_answer = "protection from invasions"
        
        hints = model_b.generate_hints(
            passage, question, correct_answer, num_hints=3
        )
        
        assert isinstance(hints, list)
        assert len(hints) == 3
        for hint in hints:
            assert isinstance(hint, str)


class TestUnifiedInference:
    """Test unified end-to-end inference."""
    
    def test_unified_inference_initialization(self):
        """Test unified inference initialization."""
        model_a_paths = {}
        model_b_paths = {}
        
        unified = UnifiedInference(model_a_paths, model_b_paths)
        assert unified is not None
        assert unified.model_a is not None
        assert unified.model_b is not None
    
    def test_generate_and_verify_mcq(self):
        """Test end-to-end MCQ generation and verification."""
        model_a_paths = {}
        model_b_paths = {}
        
        unified = UnifiedInference(model_a_paths, model_b_paths)
        
        passage = "Python is a high-level programming language known for its simplicity and readability."
        
        mcq = unified.generate_and_verify_mcq(passage)
        
        # Validate MCQ structure
        assert 'question' in mcq
        assert 'correct_answer' in mcq
        assert 'distractors' in mcq
        assert 'options' in mcq
        assert 'hints' in mcq
        
        assert isinstance(mcq['question'], str)
        assert isinstance(mcq['correct_answer'], str)
        assert isinstance(mcq['distractors'], list)
        assert isinstance(mcq['hints'], list)
        assert len(mcq['distractors']) == 3
        assert len(mcq['hints']) == 3
    
    def test_verify_user_answer(self):
        """Test user answer verification."""
        model_a_paths = {}
        model_b_paths = {}
        
        unified = UnifiedInference(model_a_paths, model_b_paths)
        
        passage = "The Earth orbits the Sun."
        question = "What does Earth orbit?"
        correct_answer = "the Sun"
        user_option = "the Sun"
        
        result = unified.verify_user_answer(
            passage, question, correct_answer, user_option
        )
        
        assert 'is_correct' in result
        assert 'confidence' in result
        assert 'explanation' in result
        assert 'correct_answer' in result


class TestFeatureEngineering:
    """Test feature engineering components."""
    
    def test_word_overlap(self):
        """Test word overlap computation."""
        from src.preprocessing import FeatureEngineer
        
        fe = FeatureEngineer()
        
        text1 = "The quick brown fox"
        text2 = "The brown fox jumps"
        
        overlap = fe.compute_word_overlap(text1, text2)
        
        assert isinstance(overlap, float)
        assert 0 <= overlap <= 1
        assert overlap > 0  # Should have overlap
    
    def test_char_match_score(self):
        """Test character-level matching."""
        from src.preprocessing import FeatureEngineer
        
        fe = FeatureEngineer()
        
        text1 = "apple"
        text2 = "apple pie"
        
        match = fe.compute_char_match_score(text1, text2)
        
        assert isinstance(match, float)
        assert 0 <= match <= 1
    
    def test_lexical_features(self):
        """Test lexical feature extraction."""
        from src.preprocessing import FeatureEngineer
        
        fe = FeatureEngineer()
        
        question = "What is the capital of France?"
        options = ["Paris", "London", "Berlin"]
        passage = "The capital of France is Paris, a beautiful city on the Seine river."
        
        features = fe.extract_lexical_features(question, options, passage)
        
        assert features.shape == (3, 4)  # 3 options, 4 features each
        assert np.all(features >= 0)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
