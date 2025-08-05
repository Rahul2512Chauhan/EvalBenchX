# dashboard/metrics/bertscore.py

from bert_score import score


def compute_bertscore(reference: str, prediction: str) -> float:
    """
    Compute BERTScore (F1) between reference and prediction.

    Args:
        reference (str): Ground truth answer.
        prediction (str): LLM-generated answer.

    Returns:
        float: BERTScore F1 (0.0 to 1.0)
    """
    try:
        P, R, F1 = score([prediction], [reference], lang="en", verbose=False)
        return F1[0].item()  # Convert tensor to float
    except Exception as e:
        print(f"❌ BERTScore computation error: {e}")
        return 0.0
