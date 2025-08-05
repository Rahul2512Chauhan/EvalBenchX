import argparse
import os
import pandas as pd
from typing import List

from utils.data_loader import load_dataset
from evaluation_core.evaluator import evaluate_outputs
from models.llm_wrapper import get_llm_response  # LLM with LangSmith tracing

# ----------  CLI Argument Parser ----------
def parse_args():
    parser = argparse.ArgumentParser(description="🔍 Evaluate LLM-generated outputs with LangSmith tracing.")
    parser.add_argument("data_path", type=str, help="Path to original dataset CSV (must have question + ground_truth_answer).")
    parser.add_argument("--metrics", nargs="+", default=["bleu", "rouge"], help="Metrics to evaluate: bleu, rouge, bertscore.")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit number of rows to evaluate.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save outputs.")
    return parser.parse_args()

# ----------  Generate answers using LLM + LangSmith ----------
def generate_answers(questions: List[str]) -> List[str]:
    responses = []
    for q in questions:
        try:
            answer = get_llm_response(q)
        except Exception as e:
            print(f"[⚠️] Error generating response for: {q[:50]}... → {str(e)}")
            answer = ""
        responses.append(answer)
    return responses

# ----------  Main Evaluation Pipeline ----------
def main():
    args = parse_args()

    print(f"\n📂 Loading dataset from: {args.data_path}")
    df = load_dataset(args.data_path, max_rows=args.max_rows)

    if "question" not in df.columns or "ground_truth_answer" not in df.columns:
        raise ValueError("CSV must contain 'question' and 'ground_truth_answer' columns.")

    questions = df["question"].fillna("").astype(str).tolist()
    references = df["ground_truth_answer"].fillna("").astype(str).tolist()

    print(f"✍️ Generating answers for {len(questions)} questions using LLM...\n")
    predictions = generate_answers(questions)

    # Save generations
    os.makedirs(args.output_dir, exist_ok=True)
    gen_path = os.path.join(args.output_dir, "generated_answers.csv")
    pd.DataFrame({"question": questions, "generated_answer": predictions}).to_csv(gen_path, index=False)
    print(f"\n💾 Generated answers saved to: {gen_path}")

    # Run evaluations
    print(f"\n🧪 Evaluating with metrics: {', '.join(args.metrics)}")
    scores = evaluate_outputs(predictions=predictions, references=references, metrics=args.metrics)

    print("\n📊 Evaluation Scores:")
    for metric, value in scores.items():
        print(f"🔹 {metric.upper()}: {value:.4f}" if isinstance(value, float) else f"🔹 {metric.upper()}: {value}")

    # Save scores
    eval_path = os.path.join(args.output_dir, "evaluation_scores.csv")
    pd.DataFrame([scores]).to_csv(eval_path, index=False)
    print(f"\n💾 Evaluation results saved to: {eval_path}\n")

if __name__ == "__main__":
    main()
