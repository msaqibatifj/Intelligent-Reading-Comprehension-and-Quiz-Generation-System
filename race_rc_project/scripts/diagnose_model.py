"""Diagnose Model A performance issues."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from inference import ModelAInference


def diagnose_model_performance(test_size=500):
    """Analyze why model accuracy is low."""
    
    # Load models
    model_a_dir = ROOT / 'models' / 'model_a' / 'traditional'
    model_a_paths = {pkl.stem: str(pkl) for pkl in model_a_dir.glob('*.pkl')}
    model_a = ModelAInference(model_a_paths)
    
    # Load test data
    test_csv = ROOT / 'data' / 'processed' / 'test_qa.csv'
    test_df = pd.read_csv(test_csv, nrows=test_size)
    
    print("=" * 70)
    print("MODEL A DIAGNOSTIC ANALYSIS")
    print("=" * 70)
    
    # 1. Check class distribution
    true_labels = test_df['label'].values
    print(f"\n1. CLASS DISTRIBUTION IN TEST SET:")
    print(f"   Positive (label=1): {(true_labels == 1).sum()} ({(true_labels == 1).mean()*100:.1f}%)")
    print(f"   Negative (label=0): {(true_labels == 0).sum()} ({(true_labels == 0).mean()*100:.1f}%)")
    
    # 2. Test predictions
    print(f"\n2. TESTING PREDICTIONS ON {test_size} SAMPLES...")
    predictions = []
    confidences = []
    
    for idx, row in test_df.iterrows():
        passage = str(row['article'])
        question = str(row['question'])
        option = str(row['option'])
        
        is_correct, confidence, _ = model_a.verify_answer(
            passage, question, option, method='lr'
        )
        predictions.append(1 if is_correct else 0)
        confidences.append(confidence)
        
        if (idx + 1) % 100 == 0:
            print(f"   Processed {idx + 1}/{test_size} samples...")
    
    pred_labels = np.array(predictions)
    confidences = np.array(confidences)
    
    # 3. Confidence distribution
    print(f"\n3. PREDICTION CONFIDENCE DISTRIBUTION:")
    print(f"   Min: {confidences.min():.4f}")
    print(f"   Max: {confidences.max():.4f}")
    print(f"   Mean: {confidences.mean():.4f}")
    print(f"   Std: {confidences.std():.4f}")
    print(f"   Median: {np.median(confidences):.4f}")
    print(f"   Percentiles:")
    for p in [25, 50, 75, 90, 95]:
        print(f"     P{p}: {np.percentile(confidences, p):.4f}")
    
    # 4. Prediction distribution
    print(f"\n4. PREDICTION DISTRIBUTION:")
    print(f"   Predicted Positive: {(pred_labels == 1).sum()} ({(pred_labels == 1).mean()*100:.1f}%)")
    print(f"   Predicted Negative: {(pred_labels == 0).sum()} ({(pred_labels == 0).mean()*100:.1f}%)")
    
    # 5. Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    print(f"\n5. CONFUSION MATRIX:")
    print(f"   True Negatives:  {cm[0,0]}")
    print(f"   False Positives: {cm[0,1]}")
    print(f"   False Negatives: {cm[1,0]}")
    print(f"   True Positives:  {cm[1,1]}")
    
    # 6. Classification report
    print(f"\n6. CLASSIFICATION REPORT:")
    print(classification_report(true_labels, pred_labels, target_names=['Negative', 'Positive'], zero_division=0))
    
    # 7. Confidence by prediction type
    print(f"\n7. AVERAGE CONFIDENCE BY PREDICTION:")
    for pred_val in [0, 1]:
        mask = pred_labels == pred_val
        avg_conf = confidences[mask].mean() if mask.sum() > 0 else 0
        print(f"   Predicted {pred_val}: {avg_conf:.4f} (n={mask.sum()})")
    
    # 8. Confidence for correct vs incorrect
    correct = (pred_labels == true_labels)
    n_incorrect = (~correct).sum()
    print(f"\n8. AVERAGE CONFIDENCE BY CORRECTNESS:")
    print(f"   Correct predictions: {confidences[correct].mean():.4f} (n={correct.sum()})")
    print(f"   Incorrect predictions: {confidences[~correct].mean():.4f} (n={n_incorrect})")
    
    # 9. Analysis of high-confidence predictions
    high_conf_mask = confidences > 0.7
    print(f"\n9. HIGH CONFIDENCE (>0.7) PREDICTIONS:")
    print(f"   Total: {high_conf_mask.sum()}")
    if high_conf_mask.sum() > 0:
        high_conf_acc = (pred_labels[high_conf_mask] == true_labels[high_conf_mask]).mean()
        print(f"   Accuracy: {high_conf_acc:.4f}")
    
    # 10. Analysis of low-confidence predictions
    low_conf_mask = confidences < 0.3
    print(f"\n10. LOW CONFIDENCE (<0.3) PREDICTIONS:")
    print(f"    Total: {low_conf_mask.sum()}")
    if low_conf_mask.sum() > 0:
        low_conf_acc = (pred_labels[low_conf_mask] == true_labels[low_conf_mask]).mean()
        print(f"    Accuracy: {low_conf_acc:.4f}")
    
    # 11. Save diagnostic data
    results_dir = ROOT / 'results'
    results_dir.mkdir(exist_ok=True)
    
    diag_df = pd.DataFrame({
        'true_label': true_labels,
        'pred_label': pred_labels,
        'confidence': confidences,
        'correct': pred_labels == true_labels
    })
    diag_df.to_csv(results_dir / 'model_a_diagnostics.csv', index=False)
    print(f"\n11. Diagnostic data saved to: {results_dir / 'model_a_diagnostics.csv'}")


if __name__ == '__main__':
    diagnose_model_performance(test_size=500)
