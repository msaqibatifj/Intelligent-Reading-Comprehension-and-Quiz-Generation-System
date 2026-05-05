"""Run Model A and Model B tests on the test dataset.

This script:
- Loads processed test data (test_qa.csv, test_distractor.csv)
- Loads trained models from models/model_a/traditional/ and models/model_b/traditional/
- Runs inference on test samples
- Saves results to results/
- Prints performance metrics

Usage:
    python race_rc_project/scripts/test_models.py
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import traceback
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from inference import ModelAInference, ModelBInference
from evaluate import ModelAEvaluator, ModelBEvaluator


def load_models():
    """Load all trained models from disk."""
    model_a_dir = ROOT / 'models' / 'model_a' / 'traditional'
    model_b_dir = ROOT / 'models' / 'model_b' / 'traditional'
    
    # Model A paths
    model_a_paths = {}
    for pkl in model_a_dir.glob('*.pkl'):
        key = pkl.stem
        model_a_paths[key] = str(pkl)
    
    # Model B paths
    model_b_paths = {}
    for pkl in model_b_dir.glob('*.pkl'):
        key = pkl.stem
        model_b_paths[key] = str(pkl)
    
    print(f"Loading {len(model_a_paths)} Model A models...")
    model_a = ModelAInference(model_a_paths)
    
    print(f"Loading {len(model_b_paths)} Model B components...")
    model_b = ModelBInference(model_b_paths)
    
    return model_a, model_b


def load_test_data():
    """Load processed test data."""
    qa_path = ROOT / 'data' / 'processed' / 'test_qa.csv'
    dist_path = ROOT / 'data' / 'processed' / 'test_distractor.csv'
    
    if not qa_path.exists():
        print(f"Test QA data not found: {qa_path}")
        return None, None
    
    df_qa = pd.read_csv(qa_path)
    df_dist = pd.read_csv(dist_path) if dist_path.exists() else None
    
    print(f"Loaded {len(df_qa)} Q-A pairs from test set")
    if df_dist is not None:
        print(f"Loaded {len(df_dist)} distractor records from test set")
    
    return df_qa, df_dist


def test_model_a(model_a, df_qa, sample_size=500):
    """Test Model A on Q&A pairs."""
    print(f"\n{'='*60}")
    print("TESTING MODEL A (Q&A Verification)")
    print('='*60)
    
    evaluator = ModelAEvaluator()
    results = []
    
    # Test on a sample (full test would be slow)
    df_sample = df_qa.sample(n=min(sample_size, len(df_qa)), random_state=42)
    
    y_true = []
    y_pred = []
    
    for idx, row in df_sample.iterrows():
        try:
            article = row.get('article', '')
            question = row.get('question', '')
            option = row.get('option', '')
            label = row.get('label', 0)
            
            # Run inference with ensemble
            is_correct, confidence, explanation = model_a.verify_answer(
                article, question, option, method='ensemble_voting'
            )
            
            pred = 1 if is_correct else 0
            y_true.append(label)
            y_pred.append(pred)
            
            results.append({
                'question': question[:100],
                'option': option[:100],
                'true_label': label,
                'pred_label': pred,
                'confidence': confidence
            })
        except Exception as e:
            print(f"Error on row {idx}: {e}")
    
    if y_true and y_pred:
        metrics = evaluator.evaluate(np.array(y_true), np.array(y_pred))
        print(f"\nModel A Metrics (on {len(y_true)} samples):")
        print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
        print(f"  Precision: {metrics.get('precision', 0):.4f}")
        print(f"  Recall:    {metrics.get('recall', 0):.4f}")
        print(f"  F1-Score:  {metrics.get('f1', 0):.4f}")
        
        # Save results
        results_dir = ROOT / 'results'
        results_dir.mkdir(exist_ok=True)
        
        df_results = pd.DataFrame(results)
        out_path = results_dir / 'model_a_test_results.csv'
        df_results.to_csv(out_path, index=False)
        print(f"\nResults saved to {out_path}")
        
        return metrics
    else:
        print("No valid predictions made")
        return None


def test_model_b(model_b, df_dist, sample_size=100):
    """Test Model B on distractor/hint generation."""
    print(f"\n{'='*60}")
    print("TESTING MODEL B (Distractor & Hint Generation)")
    print('='*60)
    
    results = []
    df_sample = df_dist.sample(n=min(sample_size, len(df_dist)), random_state=42)
    
    for idx, row in df_sample.iterrows():
        try:
            article = row.get('article', '')
            question = row.get('question', '')
            correct_answer = row.get('correct_answer', '')
            
            # Test distractor generation
            try:
                distractors = model_b.generate_distractors(
                    article, question, correct_answer, num_distractors=3
                )
            except:
                distractors = ['N/A', 'N/A', 'N/A']
            
            # Test hint generation
            try:
                hints = model_b.generate_hints(
                    article, question, correct_answer, num_hints=3
                )
            except:
                hints = ['N/A', 'N/A', 'N/A']
            
            results.append({
                'question': question[:100],
                'correct_answer': correct_answer[:100],
                'distractor_1': str(distractors[0])[:100] if len(distractors) > 0 else 'N/A',
                'distractor_2': str(distractors[1])[:100] if len(distractors) > 1 else 'N/A',
                'distractor_3': str(distractors[2])[:100] if len(distractors) > 2 else 'N/A',
                'hint_1': str(hints[0])[:100] if len(hints) > 0 else 'N/A',
                'hint_2': str(hints[1])[:100] if len(hints) > 1 else 'N/A',
                'hint_3': str(hints[2])[:100] if len(hints) > 2 else 'N/A',
            })
        except Exception as e:
            print(f"Error on row {idx}: {e}")
            traceback.print_exc()
    
    print(f"\nModel B: Generated distractors and hints for {len(results)} samples")
    
    # Save results
    results_dir = ROOT / 'results'
    results_dir.mkdir(exist_ok=True)
    
    df_results = pd.DataFrame(results)
    out_path = results_dir / 'model_b_test_results.csv'
    df_results.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")
    
    # Print sample
    if len(results) > 0:
        print(f"\nSample Output (first result):")
        sample = results[0]
        print(f"  Question: {sample['question']}")
        print(f"  Correct: {sample['correct_answer']}")
        print(f"  Distractors: {sample['distractor_1']}, {sample['distractor_2']}, {sample['distractor_3']}")
        print(f"  Hints: {sample['hint_1']}, {sample['hint_2']}, {sample['hint_3']}")


def main():
    print("Loading models and test data...")
    
    model_a, model_b = load_models()
    df_qa, df_dist = load_test_data()
    
    if df_qa is None:
        print("Cannot proceed without test data")
        return
    
    # Test Model A
    metrics_a = test_model_a(model_a, df_qa, sample_size=500)
    
    # Test Model B
    if df_dist is not None:
        test_model_b(model_b, df_dist, sample_size=100)
    
    print(f"\n{'='*60}")
    print("TESTING COMPLETE")
    print('='*60)
    print("Check results/ directory for detailed results")


if __name__ == '__main__':
    main()
