import argparse
import json
import os
import warnings
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

warnings.filterwarnings("ignore", category=DeprecationWarning)

# RAGAS imports
try:
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics import (
        BleuScore,
        ContextPrecision,
        Faithfulness,
        ResponseRelevancy,
        RougeScore,
    )

    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


def evaluate_response_quality(
    question: str,
    answer: str,
    contexts: List[str],
) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics.

    Args:
        question: The user's original question.
        answer: The model's generated answer.
        contexts: List of retrieved context strings used to produce the answer.

    Returns:
        Dictionary mapping metric names to scores (floats between 0 and 1).
        Returns {"error": "<message>"} if evaluation cannot proceed.
    """
    if not RAGAS_AVAILABLE:
        return {
            "error": "RAGAS library is not available. "
            "Install with: pip install ragas"
        }

    if not question or not question.strip():
        return {"error": "Empty or malformed question provided."}
    if not answer or not answer.strip():
        return {"error": "Empty or malformed answer provided."}
    if not contexts:
        return {"error": "No contexts provided for evaluation."}

    try:
        import os

        openai_api_key = os.environ.get(
            "CHROMA_OPENAI_API_KEY"
        ) or os.environ.get("OPENAI_API_KEY", "")

        # Initialise the evaluator LLM and embeddings
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_api_key)
        )
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=openai_api_key,
            )
        )

        # Instantiate metrics
        response_relevancy = ResponseRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )
        faithfulness = Faithfulness(llm=evaluator_llm)
        context_precision = ContextPrecision(llm=evaluator_llm)
        bleu_score = BleuScore()
        rouge_score = RougeScore()

        # Build evaluation sample
        sample = SingleTurnSample(
            user_input=question,
            retrieved_contexts=contexts,
            response=answer,
            reference=answer,
        )

        dataset = EvaluationDataset(samples=[sample])

        metrics = [
            response_relevancy,
            faithfulness,
            context_precision,
            bleu_score,
            rouge_score,
        ]

        results = evaluate(dataset=dataset, metrics=metrics)

        # Convert result to plain dict
        scores: Dict[str, float] = {}
        result_df = results.to_pandas()
        for col in result_df.columns:
            val = result_df[col].iloc[0]
            if isinstance(val, (int, float)):
                scores[col] = round(float(val), 4)

        return (
            scores
            if scores
            else {"error": "No metric scores returned by RAGAS."}
        )

    except Exception as exc:
        return {"error": f"Evaluation failed: {str(exc)}"}


def load_evaluation_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load evaluation questions from a JSON dataset file."""
    if not dataset_path or not dataset_path.strip():
        raise ValueError("Dataset path is required.")

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("Dataset JSON must be a list of objects.")

    normalized_rows: List[Dict[str, Any]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(
                f"Dataset entry at index {index} must be an object."
            )

        question = row.get("question", "")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                f"Dataset entry at index {index} has invalid question."
            )

        normalized_rows.append(
            {
                "question": question.strip(),
                "reference": str(row.get("reference", "")).strip(),
                "mission": str(row.get("mission", "")).strip(),
                "category": str(row.get("category", "")).strip(),
            }
        )

    return normalized_rows


def _aggregate_metric_values(
    question_results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Aggregate per-metric values across batch results."""
    values_by_metric: Dict[str, List[float]] = {}

    for item in question_results:
        scores = item.get("scores", {})
        if not isinstance(scores, dict):
            continue

        for metric_name, metric_value in scores.items():
            if isinstance(metric_value, (int, float)):
                values_by_metric.setdefault(metric_name, []).append(
                    float(metric_value)
                )

    summary: Dict[str, Dict[str, float]] = {}
    for metric_name, values in values_by_metric.items():
        if not values:
            continue

        summary[metric_name] = {
            "count": float(len(values)),
            "mean": round(mean(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }

    return summary


def run_batch_evaluation(
    dataset_path: str,
    openai_key: str,
    chroma_dir: str,
    collection_name: str,
    model: str = "gpt-3.5-turbo",
    n_results: int = 3,
    mission_filter: str = "",
) -> Dict[str, Any]:
    """Run end-to-end batch evaluation over a JSON question dataset."""
    if not RAGAS_AVAILABLE:
        return {
            "error": "RAGAS library is not available. "
            "Install with: pip install ragas"
        }

    try:
        dataset_items = load_evaluation_dataset(dataset_path)
    except Exception as exc:
        return {"error": f"Failed to load evaluation dataset: {exc}"}

    if openai_key and openai_key.strip():
        os.environ["CHROMA_OPENAI_API_KEY"] = openai_key.strip()

    try:
        import llm_client
        import rag_client

        collection, success, error = rag_client.initialize_rag_system(
            chroma_dir,
            collection_name,
        )
        if not success:
            return {"error": f"RAG initialization failed: {error}"}

        per_question_results: List[Dict[str, Any]] = []

        for item in dataset_items:
            question = item["question"]
            active_filter = mission_filter or item.get("mission", "")

            docs_result = rag_client.retrieve_documents(
                collection=collection,
                query=question,
                n_results=n_results,
                mission_filter=active_filter or None,
            )
            if not docs_result or docs_result.get("error"):
                retrieval_error = (
                    docs_result.get("error")
                    if isinstance(docs_result, dict)
                    else "Retrieval failed."
                )
                per_question_results.append(
                    {
                        "question": question,
                        "mission": item.get("mission", ""),
                        "category": item.get("category", ""),
                        "scores": {"error": retrieval_error},
                    }
                )
                continue

            context_items = docs_result.get("context_items", [])
            contexts_list = [
                row.get("document", "")
                for row in context_items
                if row.get("document")
            ]
            metadatas = [row.get("metadata", {}) for row in context_items]
            distances = [row.get("distance") for row in context_items]

            if not contexts_list:
                contexts_list = ["No context retrieved."]

            context = rag_client.format_context(
                contexts_list,
                metadatas,
                distances,
            )
            answer = llm_client.generate_response(
                openai_key=openai_key,
                user_message=question,
                context=context,
                conversation_history=[],
                model=model,
            )

            scores = evaluate_response_quality(
                question=question,
                answer=answer,
                contexts=contexts_list,
            )

            per_question_results.append(
                {
                    "question": question,
                    "mission": item.get("mission", ""),
                    "category": item.get("category", ""),
                    "reference": item.get("reference", ""),
                    "answer": answer,
                    "scores": scores,
                }
            )

        return {
            "dataset_path": dataset_path,
            "question_count": len(dataset_items),
            "results": per_question_results,
            "metric_summary": _aggregate_metric_values(per_question_results),
        }
    except Exception as exc:
        return {"error": f"Batch evaluation failed: {exc}"}


def _print_batch_summary(batch_result: Dict[str, Any]) -> None:
    """Print per-question and aggregate metric summaries for batch runs."""
    if batch_result.get("error"):
        print(f"ERROR: {batch_result['error']}")
        return

    print("Batch Evaluation Results")
    print("=" * 80)
    print(f"Dataset: {batch_result.get('dataset_path', 'N/A')}")
    print(f"Questions: {batch_result.get('question_count', 0)}")
    print("-" * 80)

    for index, item in enumerate(batch_result.get("results", []), start=1):
        print(f"[{index}] Question: {item.get('question', '')}")
        print(f"    Mission: {item.get('mission', '')}")
        print(f"    Category: {item.get('category', '')}")
        scores = item.get("scores", {})
        if isinstance(scores, dict):
            for metric_name, value in scores.items():
                print(f"    {metric_name}: {value}")
        print("-" * 80)

    print("Metric Summary")
    print("=" * 80)
    metric_summary = batch_result.get("metric_summary", {})
    for metric_name, aggregate in metric_summary.items():
        print(f"{metric_name} -> {aggregate}")


def main() -> None:
    """CLI entrypoint for batch RAGAS evaluation."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end RAGAS batch evaluation on NASA dataset"
    )
    parser.add_argument(
        "--dataset-path",
        default="test_questions.json",
        help="Path to evaluation dataset JSON",
    )
    parser.add_argument(
        "--openai-key",
        default=os.getenv("OPENAI_API_KEY", ""),
        help="OpenAI API key",
    )
    parser.add_argument(
        "--chroma-dir",
        default="./chroma_db_openai",
        help="ChromaDB persist directory",
    )
    parser.add_argument(
        "--collection-name",
        default="nasa_space_missions_text",
        help="ChromaDB collection name",
    )
    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo",
        help="OpenAI model used for generated answers",
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=3,
        help="Top-k retrieval count per question",
    )
    parser.add_argument(
        "--mission-filter",
        default="",
        help="Optional mission filter for all dataset questions",
    )

    args = parser.parse_args()

    batch_result = run_batch_evaluation(
        dataset_path=args.dataset_path,
        openai_key=args.openai_key,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection_name,
        model=args.model,
        n_results=args.n_results,
        mission_filter=args.mission_filter,
    )
    _print_batch_summary(batch_result)


if __name__ == "__main__":
    main()
