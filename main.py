import argparse
from utils.data_loader import load_dataset
from utils.generator import generate_answers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="📊 EvalBenchX: LLM QA Evaluation Tool")

    parser.add_argument(
        "__path",
        type=str,
        help="Path to input dataset (.csv)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="load",
        choices=["load", "generate"],
        help="Choose 'load' to preview dataset or 'generate' to run LLM generation.",
    )

    parser.add_argument(
        "--prompt-type",
        type=str,
        default="rag",
        choices=["qa", "rag"],
        help="Prompt style: 'qa' (question only) or 'rag' (question + context).",
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit number of rows processed (useful for testing).",
    )

    args = parser.parse_args()

    # 📦 Phase 1: Load Dataset
    if args.mode == "load":
        print("🔍 Loading dataset...\n")
        df = load_dataset(args.__path)
        print("✅ Dataset loaded successfully!\n")
        print("📊 Columns:", list(df.columns))
        print("\n📝 Sample rows:\n", df.head())

    # ⚙️ Phase 2: Generate Answers
    elif args.mode == "generate":
        print(f"\n⚙️  Running LLM generation using '{args.prompt_type}' prompt...\n")
        generate_answers(
            dataset_path=args.__path,
            prompt_type=args.prompt_type,
            max_rows=args.max_rows
        )