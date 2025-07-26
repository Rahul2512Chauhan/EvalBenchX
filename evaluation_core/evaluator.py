import pandas as pd
from typing import List, Dict
from evaluate import load as load_metric
from bert_score import score as bert_score


class Evaluator:
    def __init__(self):
        self.bleu = load_metric("bleu")
        self.rouge = load_metric("rouge")

    def evaluate_bleu(self, references: List[str], predictions: List[str]) -> float:
        """
        BLEU expects:
        - predictions: List[str]
        - references: List[List[str]] ← wrap each reference in a list
        """
        formatted_refs = [[ref] for ref in references]
        result = self.bleu.compute(predictions=predictions, references=formatted_refs)
        return round(result["bleu"], 4)

    def evaluate_rouge(self, references: List[str], predictions: List[str]) -> Dict[str, float]:
        result = self.rouge.compute(predictions=predictions, references=references)
        return {
            "rouge1": round(result["rouge1"], 4),
            "rouge2": round(result["rouge2"], 4),
            "rougeL": round(result["rougeL"], 4),
        }

    def evaluate_bertscore(self, references: List[str], predictions: List[str]) -> float:
        P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
        return round(F1.mean().item(), 4)

    def evaluate_all(self, references: List[str], predictions: List[str]) -> Dict[str, float]:
        bleu = self.evaluate_bleu(references, predictions)
        rouge = self.evaluate_rouge(references, predictions)
        bert = self.evaluate_bertscore(references, predictions)
        return {
            "bleu": bleu,
            **rouge,
            "bertscore": bert,
        }

    def evaluate_from_csv(
        self,
        filepath: str,
        pred_col: str = "generated_answer",
        label_col: str = "ground_truth",
    ) -> Dict[str, float]:
        df = pd.read_csv(filepath)
        preds = df[pred_col].fillna("").astype(str).tolist()
        refs = df[label_col].fillna("").astype(str).tolist()
        return self.evaluate_all(references=refs, predictions=preds)


# Standalone functional interface
def evaluate_outputs(
    predictions: List[str],
    references: List[str],
    metrics: List[str],
) -> Dict[str, float]:
    """
    Evaluate model predictions using specified metrics.

    Args:
        predictions (List[str]): Model-generated answers
        references (List[str]): Ground truth answers
        metrics (List[str]): Metrics to compute (bleu, rouge, bertscore)

    Returns:
        Dict[str, float]: Metric results
    """
    eval_obj = Evaluator()
    results = {}

    if "bleu" in metrics:
        results["bleu"] = eval_obj.evaluate_bleu(references, predictions)

    if "rouge" in metrics:
        results.update(eval_obj.evaluate_rouge(references, predictions))

    if "bertscore" in metrics:
        results["bertscore"] = eval_obj.evaluate_bertscore(references, predictions)

    return results
