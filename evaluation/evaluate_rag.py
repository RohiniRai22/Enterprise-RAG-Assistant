
import json
import os
import asyncio
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from google import genai

from ragas.llms import llm_factory
from ragas.embeddings import GoogleEmbeddings

from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

from evaluation.rag_pipeline_adapter import run_rag_pipeline


# 1. Load environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")


# 2. Create asynchronous Gemini-compatible client
client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


# 3. Initialize RAGAS evaluation LLM
evaluator_llm = llm_factory(
    model = "gemini-3.5-flash-lite",
    #MODEL_NAME = "gemini-2.5-flash-lite"
    provider="openai",
    client=client,
    adapter="instructor"
)

print("RAGAS evaluator initialized:", evaluator_llm)
print("Async support:", evaluator_llm.is_async)


# 4. Initialize evaluation embeddings
embedding_client = genai.Client(
    api_key=API_KEY
)

evaluator_embeddings = GoogleEmbeddings(
    client=embedding_client,
    model="gemini-embedding-001"
)


# 5. Initialize evaluation metrics
metrics = {
    "faithfulness": Faithfulness(
        llm=evaluator_llm
    ),

    "answer_relevancy": AnswerRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    ),

    "context_precision": ContextPrecision(
        llm=evaluator_llm
    ),

    "context_recall": ContextRecall(
        llm=evaluator_llm
    ),
}


# 6. Dataset and output paths
BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "evaluation_dataset.json"

RESULTS_DIR = BASE_DIR / "evaluation_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = RESULTS_DIR / "rag_evaluation_results.csv"


# 7. Load evaluation dataset
def load_dataset():

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# 8. Main evaluation function
async def main():

    dataset = load_dataset()
    results = []

    print("\nSTARTING RAG EVALUATION")
    print("=" * 60)

    for index, item in enumerate(dataset, start=1):

        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"\nQuestion {index}: {question}")

        # Run existing RAG pipeline
        rag_result = run_rag_pipeline(question)

        contexts = rag_result["contexts"]
        answer = rag_result["answer"]

        print("\nGenerated Answer:")
        print(answer)

        print("\nRetrieved Context Count:", len(contexts))

        row = {
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "contexts": " | ".join(contexts),
        }

        # Prepare metric inputs
        metric_inputs = {

            "faithfulness": {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
            },

            "answer_relevancy": {
                "user_input": question,
                "response": answer,
            },

            "context_precision": {
                "user_input": question,
                "reference": ground_truth,
                "retrieved_contexts": contexts,
            },

            "context_recall": {
                "user_input": question,
                "retrieved_contexts": contexts,
                "reference": ground_truth,
            },
        }

        # Calculate RAGAS evaluation scores
        for metric_name, metric in metrics.items():

            try:

                score = await metric.ascore(
                    **metric_inputs[metric_name]
                )

                row[metric_name] = score.value

                print(f"{metric_name}: {score.value}")

            except Exception as error:

                row[metric_name] = None

                print(
                    f"Error calculating {metric_name}: {error}"
                )

        results.append(row)

    # 9. Save results to CSV
    dataframe = pd.DataFrame(results)

    dataframe.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("RAG EVALUATION COMPLETED")

    print("\nResults saved at:")
    print(OUTPUT_PATH)

    print("\nFINAL EVALUATION RESULTS")
    print(dataframe.to_string(index=False))


# 10. Execute program
if __name__ == "__main__":
    asyncio.run(main())
