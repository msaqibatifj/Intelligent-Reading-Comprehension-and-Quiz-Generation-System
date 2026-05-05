"""Debug Model A inference to see what's happening with feature extraction."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from inference import ModelAInference


def test_single_prediction():
    """Test a single prediction with debug output."""
    model_a_dir = ROOT / 'models' / 'model_a' / 'traditional'
    
    model_a_paths = {}
    for pkl in model_a_dir.glob('*.pkl'):
        key = pkl.stem
        model_a_paths[key] = str(pkl)
    
    print(f"Loading models...")
    model_a = ModelAInference(model_a_paths)
    
    print(f"\nFeature engineer loaded: {model_a.feature_engineer is not None}")
    print(f"Available models: {list(model_a.models.keys())}")
    
    # Test data
    passage = "The law of overlearning explains why cramming for an examination might not be as effective."
    question = "What is the law of overlearning?"
    option = "presenting research findings"
    
    print(f"\nTest input:")
    print(f"  Passage: {passage[:100]}...")
    print(f"  Question: {question}")
    print(f"  Option: {option}")
    
    # Try to predict
    print(f"\nTesting verify_answer with different models:")
    
    for method in ['lr', 'svm', 'rf', 'nb']:
        try:
            is_correct, confidence, explanation = model_a.verify_answer(
                passage, question, option, method=method
            )
            print(f"  {method}: is_correct={is_correct}, confidence={confidence:.4f}, explanation={explanation}")
        except Exception as e:
            print(f"  {method}: ERROR - {e}")
            traceback.print_exc()


if __name__ == '__main__':
    test_single_prediction()
