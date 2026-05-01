"""
Evaluation metrics and reporting for Model A and Model B.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ModelAEvaluator:
    """Evaluation for Model A (Q&A verification)."""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate(self, y_true, y_pred, y_pred_proba=None):
        """
        Compute evaluation metrics.
        Returns: dict with all metrics
        """
        self.metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro'),
            'recall': recall_score(y_true, y_pred, average='macro'),
            'f1': f1_score(y_true, y_pred, average='macro'),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        }
        
        if y_pred_proba is not None:
            try:
                self.metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except:
                self.metrics['roc_auc'] = None
        
        return self.metrics
    
    def exact_match(self, y_true, y_pred):
        """Compute Exact Match score (same as accuracy for binary)."""
        return accuracy_score(y_true, y_pred)
    
    def print_report(self, y_true, y_pred):
        """Print detailed classification report."""
        return classification_report(y_true, y_pred, 
                                     target_names=['Incorrect', 'Correct'])
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Model A - Answer Verification Confusion Matrix')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()


class ModelBEvaluator:
    """Evaluation for Model B (Distractor & Hint generation)."""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate_distractor_ranking(self, y_true, y_pred, y_pred_proba=None):
        """
        Evaluate distractor ranking model (binary: correct vs. distractor).
        """
        self.metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro'),
            'recall': recall_score(y_true, y_pred, average='macro'),
            'f1': f1_score(y_true, y_pred, average='macro'),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        }
        
        if y_pred_proba is not None:
            try:
                self.metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except:
                self.metrics['roc_auc'] = None
        
        return self.metrics
    
    def evaluate_hint_extraction(self, hint_scores, y_true):
        """
        Evaluate hint extraction (assuming y_true is binary: relevant=1, irrelevant=0).
        """
        y_pred = (hint_scores > np.median(hint_scores)).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro'),
            'recall': recall_score(y_true, y_pred, average='macro'),
            'f1': f1_score(y_true, y_pred, average='macro'),
        }
        
        return metrics
    
    def evaluate_hint_scoring_regression(self, y_true, y_pred):
        """
        Evaluate hint scoring regression model.
        """
        metrics = {
            'r2_score': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': np.mean(np.abs(y_true - y_pred))
        }
        
        return metrics
    
    def human_evaluation_form(self, question_id, question, distractors, hints):
        """
        Generate a human evaluation form template (1-5 Likert scale).
        """
        form = {
            'question_id': question_id,
            'question': question,
            'distractors': distractors,
            'distractors_plausibility': None,  # 1-5 (1=very implausible, 5=very plausible)
            'distractors_incorrectness': None,  # 1-5 (1=correct, 5=definitely wrong)
            'distractors_diversity': None,  # 1-5 (1=all similar, 5=very diverse)
            'hints': hints,
            'hints_clarity': None,  # 1-5 (1=confusing, 5=very clear)
            'hints_progression': None,  # 1-5 (1=no progression, 5=perfect progression)
            'overall_quality': None,  # 1-5 (1=poor, 5=excellent)
        }
        return form


class UnifiedEvaluator:
    """Unified evaluation reporting."""
    
    def __init__(self):
        self.model_a_evaluator = ModelAEvaluator()
        self.model_b_evaluator = ModelBEvaluator()
    
    def generate_inference_report(self, inferences, y_true_a, y_true_b):
        """
        Generate comprehensive inference report.
        inferences: list of dicts with model predictions
        """
        report = {
            'model_a': self.model_a_evaluator.evaluate(y_true_a, inferences['model_a']),
            'model_b': self.model_b_evaluator.evaluate_distractor_ranking(y_true_b, inferences['model_b'])
        }
        
        return report
    
    def export_session_results(self, results_df, output_path):
        """Export session results to CSV."""
        results_df.to_csv(output_path, index=False)
        return output_path


if __name__ == "__main__":
    print("Evaluation module loaded successfully.")
