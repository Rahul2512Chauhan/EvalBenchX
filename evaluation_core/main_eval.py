import argparse
import os
import pandas as pd

from utils.data_loader import load_dataset
from evaluation_core.evaluator import evaluate_outputs

# ---------- 🎯 CLI Argument Parser ----------
def parser_args():
    parser = argparse.ArgumentParser(description="🔍 Evaluate LLM-generated outputs using different metrics.")
    parser.add_argument("data_path", type=str, help="Path to the original dataset CSV (with ground truth labels).")
    parser.add_argument("--generated-path", type=str, default="outputs/generated_answers.csv", help="Path to generated answers CSV.")
    parser.add_argument("--metrics", nargs="+", default=["bleu", "rouge"], help="Metrics to evaluate with: bleu, rouge, bertscore")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit evaluation to N rows")

    return parser.parse_args()

# ---------- 🚀 Main Evaluation Pipeline ----------
def main():
    args = parser_args()

    print("\n🧪 Starting evaluation with metrics:", ", ".join(args.metrics))

    # Load datasets with optional row limit
    df_original = load_dataset(args.data_path, max_rows=args.max_rows)
    df_generated = pd.read_csv(args.generated_path)

    # Check required columns
    if "generated_answer" not in df_generated.columns:
        raise ValueError("CSV must contain 'generated_answer' column.")

    if "question" not in df_original.columns or "ground_truth_answer" not in df_original.columns:
        raise ValueError("Original CSV must contain 'question' and 'ground_truth_answer' columns.")

    # Merge ground truth and generated answers
    df_eval = df_original.copy()
    df_eval["generated_answer"] = df_generated["generated_answer"].values[:len(df_eval)]


    # Evaluate
    scores = evaluate_outputs(
    predictions=df_eval["generated_answer"].tolist(),
    references=df_eval["ground_truth_answer"].tolist(),
    metrics=args.metrics
)


    # Print scores nicely
    print("\n📊 Evaluation Scores:")
    for metric, value in scores.items():
        if isinstance(value, float):
            print(f"🔹 {metric.upper()}: {value:.4f}")
        else:
            print(f"🔹 {metric.upper()}: {value}")

    # Save results
    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/evaluation_scores.csv"
    pd.DataFrame([scores]).to_csv(output_path, index=False)
    print(f"\n💾 Evaluation results saved to: {output_path}\n")

if __name__ == "__main__":
    main()
