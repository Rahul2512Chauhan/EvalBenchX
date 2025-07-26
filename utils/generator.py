"""
LLM Answer Generator
- Loads dataset
- Formats prompts
- Calls LLMs (via llm_wrapper)
- Appends results
- Saves new dataset
"""

import os
import pandas as pd
from tqdm import tqdm
from typing import Optional

from utils.data_loader import load_dataset
from models.llm_wrapper import get_completion
from configs.prompts import QA_PROMPT_TEMPLATE , RAG_PROMPT_TEMPLATE

def generate_answers(
    dataset_path: str,
    prompt_type: str = "rag",  # 'qa' or 'rag'
    output_path: str = "outputs/generated_answers.csv",
    max_rows: Optional[int] = None,  # limit rows for testing
    log_every: int = 10     # progress logging       
) -> pd.DataFrame:
    """
    Generate answers for each question using selected prompt type and LLM.
    """
    print(f"🚀 Generating answers using prompt: {prompt_type.upper()}")
    df = load_dataset(dataset_path)

    if max_rows:
        df = df.head(max_rows)

    answers = []

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
        question = row["question"]
        context = row.get("context", "")

        # Build the appropriate prompt
        if prompt_type == "rag":
            prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)
        elif prompt_type == "qa":
            prompt = QA_PROMPT_TEMPLATE.format(question=question)
        else:
            raise ValueError("Invalid prompt_type. Choose 'qa' or 'rag'.")

        # Call LLM to get answer
        response = get_completion(prompt)
        answers.append(response)

        # Periodic logging
        if (i + 1) % log_every == 0:
            print(f"✅ Generated {i + 1} answers...")
    
    #append the generated answers
    df["generated_answer"] = answers


    #sav to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path , index=False)
    print(f"\n💾 All responses saved to: {output_path}")

    return df