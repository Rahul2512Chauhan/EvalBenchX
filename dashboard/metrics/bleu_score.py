# dashboard/metrics/bleu_score.py

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize


def compute_bleu(reference: str, prediction: str) -> float:
    """
    Compute BLEU score between reference and prediction.

    Args:
        reference (str): Ground truth answer.
        prediction (str): LLM-generated answer.

    Returns:
        float: BLEU score (0.0–1.0). Returns 0.0 for invalid input.
    """
    try:
        # Tokenize inputs
        ref_tokens = word_tokenize(reference.strip().lower())
        pred_tokens = word_tokenize(prediction.strip().lower())

        # Return 0.0 if any input is empty
        if not ref_tokens or not pred_tokens:
            return 0.0

        # Smoothing to avoid zero on short sentences
        smoothing = SmoothingFunction().method1

        # Calculate BLEU score
        score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)

        # Validate score is float-like
        if isinstance(score, (int, float)):
            return float(score)

        return 0.0  # Fallback if score is malformed

    except Exception as e:
        print(f"[BLEU ERROR] {e}")
        return 0.0
