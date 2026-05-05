"""Quick test of Model A and Model B on small sample."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from inference import ModelAInference, ModelBInference


def load_test_data(csv_path, sample_size=50):
    """Load test data."""
    df = pd.read_csv(csv_path, nrows=sample_size)
    return df


def test_model_a_quick(model_a, test_df, sample_size=50):
    """Quick test of Model A on a sample."""
    results = []
    
    for idx, row in test_df.head(sample_size).iterrows():
        passage = str(row.get('article', row.get('passage', '')))
        question = str(row.get('question', ''))
        option = str(row.get('option', ''))
        true_label = int(row.get('label', 0))
        
        # Test with LR model
        is_correct, confidence, _ = model_a.verify_answer(
            passage, question, option, method='lr'
        )
        pred_label = 1 if is_correct else 0
        
        results.append({
            'passage': passage[:100],
            'question': question[:100],
            'option': option[:100],
            'true_label': true_label,
            'pred_label': pred_label,
            'confidence': confidence
        })
    
    # Calculate metrics
    df_results = pd.DataFrame(results)
    accuracy = accuracy_score(df_results['true_label'], df_results['pred_label'])
    precision = precision_score(df_results['true_label'], df_results['pred_label'], zero_division=0)
    recall = recall_score(df_results['true_label'], df_results['pred_label'], zero_division=0)
    f1 = f1_score(df_results['true_label'], df_results['pred_label'], zero_division=0)
    
    print(f"\nModel A Metrics (on {sample_size} samples):")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    return df_results


if __name__ == '__main__':
    print("Quick Model A Test on 50 samples")
    print("=" * 60)
    
    # Load models
    model_a_dir = ROOT / 'models' / 'model_a' / 'traditional'
    model_a_paths = {pkl.stem: str(pkl) for pkl in model_a_dir.glob('*.pkl')}
    model_a = ModelAInference(model_a_paths)
    
    # Load test data - use processed test file
    test_csv = ROOT / 'data' / 'processed' / 'test_qa.csv'
    
    print(f"Loading test data from: {test_csv}")
    test_df = load_test_data(test_csv, sample_size=50)
    
    # Test
    results = test_model_a_quick(model_a, test_df, sample_size=50)
    
    # Show sample results
    print("\nSample Results:")
    print(results.head(10))
