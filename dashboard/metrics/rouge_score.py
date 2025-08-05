# dashboard/metrics/rouge_score.py

from rouge_score import rouge_scorer


def compute_rouge(reference: str, prediction: str) -> dict:
    """
    Compute ROUGE scores between reference and prediction.

    Returns:
        dict: {'rouge1': float, 'rouge2': float, 'rougeL': float}
    """
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return {
            "rouge1": scores['rouge1'].fmeasure,
            "rouge2": scores['rouge2'].fmeasure,
            "rougeL": scores['rougeL'].fmeasure,
        }
    except Exception as e:
        print(f"❌ ROUGE computation error: {e}")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
