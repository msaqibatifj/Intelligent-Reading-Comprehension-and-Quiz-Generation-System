"""Evaluation helpers for text generation tasks (BLEU, ROUGE, METEOR)."""

import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


# Ensure required NLTK resources are available for METEOR tokenization/synonyms.
for resource in ["punkt", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)


class GenerationMetricsEvaluator:
    """Compute BLEU, ROUGE, and METEOR for generated text."""

    def __init__(self):
        self.rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1

    def _safe_tokens(self, text):
        return str(text).strip().split()

    def evaluate_generation(self, references, hypotheses):
        """Return corpus-level BLEU/ROUGE/METEOR averages."""
        if len(references) != len(hypotheses):
            raise ValueError("references and hypotheses must have the same length")

        if len(references) == 0:
            return {
                "bleu": 0.0,
                "rouge_1": 0.0,
                "rouge_2": 0.0,
                "rouge_l": 0.0,
                "meteor": 0.0,
            }

        bleu_scores = []
        meteor_scores = []
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []

        for ref, hyp in zip(references, hypotheses):
            ref_tokens = self._safe_tokens(ref)
            hyp_tokens = self._safe_tokens(hyp)

            bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=self.smoothing)
            bleu_scores.append(float(bleu))

            meteor = meteor_score([ref_tokens], hyp_tokens)
            meteor_scores.append(float(meteor))

            rouge = self.rouge.score(str(ref), str(hyp))
            rouge1_scores.append(float(rouge["rouge1"].fmeasure))
            rouge2_scores.append(float(rouge["rouge2"].fmeasure))
            rougeL_scores.append(float(rouge["rougeL"].fmeasure))

        n = float(len(references))
        return {
            "bleu": sum(bleu_scores) / n,
            "rouge_1": sum(rouge1_scores) / n,
            "rouge_2": sum(rouge2_scores) / n,
            "rouge_l": sum(rougeL_scores) / n,
            "meteor": sum(meteor_scores) / n,
        }


class ModelAEvaluator:
    """Model A evaluator based on text-generation metrics only."""

    def __init__(self):
        self.generator_metrics = GenerationMetricsEvaluator()

    def evaluate(self, references, hypotheses):
        return self.generator_metrics.evaluate_generation(references, hypotheses)


class ModelBEvaluator:
    """Model B evaluator based on text-generation metrics only."""

    def __init__(self):
        self.generator_metrics = GenerationMetricsEvaluator()

    def evaluate_distractors(self, references, hypotheses):
        return self.generator_metrics.evaluate_generation(references, hypotheses)

    def evaluate_hints(self, references, hypotheses):
        return self.generator_metrics.evaluate_generation(references, hypotheses)

    def human_evaluation_form(self, question_id, question, distractors, hints):
        """Generate a human evaluation form template (1-5 Likert scale)."""
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
    """Unified BLEU/ROUGE/METEOR reporting for both models."""

    def __init__(self):
        self.model_a_evaluator = ModelAEvaluator()
        self.model_b_evaluator = ModelBEvaluator()

    def generate_inference_report(self, model_a_refs, model_a_hyps, model_b_refs, model_b_hyps):
        """Generate combined text-generation metric report."""
        report = {
            'model_a': self.model_a_evaluator.evaluate(model_a_refs, model_a_hyps),
            'model_b': self.model_b_evaluator.evaluate_distractors(model_b_refs, model_b_hyps)
        }

        return report

    def export_session_results(self, results_df, output_path):
        """Export session results to CSV."""
        results_df.to_csv(output_path, index=False)
        return output_path


if __name__ == "__main__":
    print("Evaluation module loaded successfully.")
