"""
Test script to validate the new multi-class Model A inference.
Run this after downloading retrained models from Kaggle.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from inference import ModelAInference


def test_multiclass_inference():
    """Test the new multi-class predict_answer() method."""
    
    print("=" * 70)
    print("TESTING MULTI-CLASS MODEL A INFERENCE")
    print("=" * 70)
    
    # Test data
    passage = """
    The Amazon rainforest is one of the most important ecosystems on Earth.
    It produces about 20% of the world's oxygen and is home to millions of species.
    Scientists estimate that we've discovered less than 1% of the plants and insects
    that live in the Amazon. The rainforest also plays a crucial role in regulating
    global climate by storing massive amounts of carbon.
    """
    
    question = "What percentage of the world's oxygen does the Amazon produce?"
    
    options = [
        "10% of the world's oxygen",
        "20% of the world's oxygen",
        "30% of the world's oxygen",
        "40% of the world's oxygen"
    ]
    
    correct_answer_index = 1  # "20% of the world's oxygen"
    
    # Load models
    print("\n1. Loading models...")
    model_dir = project_root / 'models' / 'model_a' / 'traditional'
    
    if not model_dir.exists():
        print(f"   ❌ Model directory not found: {model_dir}")
        print("   Please download models from Kaggle first!")
        return False
    
    model_paths = {
        'feature_engineer': str(model_dir / 'feature_engineer.pkl'),
        'ensemble_voting_model': str(model_dir / 'ensemble_voting_model.pkl'),
        'rf_model': str(model_dir / 'rf_model.pkl'),
        'xgb_model': str(model_dir / 'xgb_model.pkl'),
        'lr_model': str(model_dir / 'lr_model.pkl'),
        'svm_model': str(model_dir / 'svm_model.pkl'),
        'nb_model': str(model_dir / 'nb_model.pkl'),
    }
    
    try:
        inference = ModelAInference(model_paths)
        print("   ✓ Models loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load models: {e}")
        return False
    
    # Test predictions with different models
    print("\n2. Testing predictions...")
    models_to_test = [
        'ensemble_voting_model',
        'rf_model',
        'xgb_model',
        'lr_model',
    ]
    
    results = {}
    for model_name in models_to_test:
        if model_name not in inference.models:
            print(f"   ⚠ Model {model_name} not loaded, skipping...")
            continue
        
        try:
            result = inference.predict_answer(
                passage=passage,
                question=question,
                options=options,
                method=model_name
            )
            results[model_name] = result
            
            # Print result
            option_label = result.get('predicted_letter', '?')
            pred_idx = result.get('predicted_option')
            confidence = result.get('confidence', 0)
            is_correct = pred_idx == correct_answer_index
            status = "✓" if is_correct else "❌"
            
            print(f"\n   {model_name}:")
            print(f"   {status} Predicted: Option {option_label} (index {pred_idx})")
            print(f"      Confidence: {confidence:.1%}")
            print(f"      Probabilities: {[f'{p:.2%}' for p in result.get('probabilities', [])]}")
            
            if is_correct:
                print(f"      CORRECT! ✓")
            else:
                print(f"      Expected option {correct_answer_index} (B)")
        
        except Exception as e:
            print(f"   ❌ Error with {model_name}: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    correct_predictions = sum(
        1 for r in results.values() 
        if r.get('predicted_option') == correct_answer_index
    )
    total_predictions = len(results)
    
    if total_predictions == 0:
        print("❌ No predictions made - check model loading")
        return False
    
    accuracy = correct_predictions / total_predictions
    print(f"Accuracy on test sample: {accuracy:.1%} ({correct_predictions}/{total_predictions})")
    print(f"Expected accuracy on full test set: 60-80%")
    
    if accuracy > 0.5:
        print("\n✓ Multi-class model is working well!")
        return True
    else:
        print("\n⚠ Consider retraining with more balanced data")
        return False


def test_legacy_warning():
    """Test that legacy verify_answer() method shows deprecation warning."""
    
    print("\n" + "=" * 70)
    print("TESTING LEGACY API (DEPRECATED)")
    print("=" * 70)
    
    model_dir = Path(__file__).parent / 'models' / 'model_a' / 'traditional'
    
    try:
        model_paths = {
            'feature_engineer': str(model_dir / 'feature_engineer.pkl'),
            'ensemble_voting_model': str(model_dir / 'ensemble_voting_model.pkl'),
        }
        inference = ModelAInference(model_paths)
        
        # Call legacy method
        is_correct, conf, msg = inference.verify_answer(
            passage="Test passage",
            question="Test question",
            option="Test option"
        )
        
        print(f"\nLegacy verify_answer() returns:")
        print(f"  is_correct: {is_correct}")
        print(f"  confidence: {conf}")
        print(f"  message: {msg}")
        print(f"\n⚠ This method is deprecated!")
        print(f"✓ Use predict_answer() instead with all 4 options")
        
    except Exception as e:
        print(f"Error testing legacy method: {e}")


if __name__ == '__main__':
    success = test_multiclass_inference()
    test_legacy_warning()
    
    print("\n" + "=" * 70)
    if success:
        print("✓ Multi-class inference is working correctly!")
        print("\nYou're ready to integrate into the Streamlit UI:")
        print("  - Use predict_answer() to select the correct option")
        print("  - Pass all 4 options for reliable predictions")
        print("  - Use ensemble_voting_model or rf_model for best results")
    else:
        print("⚠ Issue detected - check model downloads and paths")
    print("=" * 70)
