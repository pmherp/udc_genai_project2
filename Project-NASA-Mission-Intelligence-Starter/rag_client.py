import os
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")

    # Look for ChromaDB directories
    for directory in current_dir.glob("**/chroma_db_*"):
        try:
            # Initialize database client
            client = chromadb.PersistentClient(path=str(directory))

            # Retrieve list of available collections from the database
            collections = client.list_collections()

            for collection in collections:
                key = f"{directory.name}_{collection.name}"

                try:
                    real_collection = client.get_collection(collection.name)
                    doc_count = real_collection.count()
                except Exception:
                    doc_count = "Unknown"

                backends[key] = {
                    "directory": str(directory),
                    "collection_name": collection.name,
                    "display_name": f"{directory.name} - {collection.name}",
                    "document_count": doc_count,
                }
        except Exception as e:
            # Handle connection or access errors gracefully
            backends[str(directory)] = {
                "directory": str(directory),
                "collection": "Unavailable",
                "display_name": f"Error: {str(e)[:50]}...",
                "document_count": "N/A",
            }

    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend"""
    try:
        openai_api_key = (
            os.getenv("CHROMA_OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("OPENAI_KEY")
        )
        if not openai_api_key:
            return (
                None,
                False,
                "Missing OpenAI API key. Set CHROMA_OPENAI_API_KEY, "
                "OPENAI_API_KEY, or OPENAI_KEY.",
            )

        client = chromadb.PersistentClient(path=chroma_dir)
        collection = client.get_collection(
            collection_name,
            embedding_function=OpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name="text-embedding-3-small",
            ),
        )

        return collection, True, None

    except Exception as e:
        return None, False, str(e)


def retrieve_documents(
    collection,
    query: str,
    n_results: int = 3,
    mission_filter: Optional[str] = None,
) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""
    filter_criteria = None

    if mission_filter and mission_filter.lower() != "all":
        filter_criteria = {"mission": mission_filter}

    try:
        results = collection.query(
            query_texts=[query],  # <- use query_texts as a list
            n_results=n_results,
            where=filter_criteria,
        )
        results["context_items"] = prepare_context_items(results)
        return results
    except Exception as e:
        return {"error": f"Failed to retrieve documents: {str(e)}"}


def prepare_context_items(results: Dict) -> List[Dict]:
    """Build ranked context items with deterministic deduplication."""
    documents_nested = results.get("documents", [[]])
    metadatas_nested = results.get("metadatas", [[]])
    distances_nested = results.get("distances", [[]])

    documents = documents_nested[0] if documents_nested else []
    metadatas = metadatas_nested[0] if metadatas_nested else []
    distances = distances_nested[0] if distances_nested else []

    items: List[Dict] = []
    for index, doc in enumerate(documents):
        if not isinstance(doc, str) or not doc.strip():
            continue

        metadata = metadatas[index] if index < len(metadatas) else {}
        if not isinstance(metadata, dict):
            metadata = {}

        distance = distances[index] if index < len(distances) else None
        items.append(
            {
                "document": doc.strip(),
                "metadata": metadata,
                "distance": distance,
            }
        )

    deduplicated: Dict[str, Dict] = {}
    for item in items:
        key = " ".join(item["document"].split())
        existing = deduplicated.get(key)
        if existing is None:
            deduplicated[key] = item
            continue

        existing_distance = existing.get("distance")
        current_distance = item.get("distance")
        if isinstance(current_distance, (int, float)) and (
            not isinstance(existing_distance, (int, float))
            or current_distance < existing_distance
        ):
            deduplicated[key] = item

    ranked_items = list(deduplicated.values())
    ranked_items.sort(
        key=lambda item: (
            item.get("distance")
            if isinstance(item.get("distance"), (int, float))
            else float("inf"),
            str(item.get("metadata", {}).get("source", "")),
        )
    )
    return ranked_items


def format_context(
    documents: List[str],
    metadatas: List[Dict],
    distances: Optional[List[float]] = None,
) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""

    context_parts = ["### Context Section"]

    context_items: List[Dict] = []
    for index, doc in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        distance = (
            distances[index] if distances and index < len(distances) else None
        )
        context_items.append(
            {
                "document": doc,
                "metadata": metadata if isinstance(metadata, dict) else {},
                "distance": distance,
            }
        )

    unique_context_items: Dict[str, Dict] = {}
    for item in context_items:
        if not isinstance(item["document"], str):
            continue
        normalized = " ".join(item["document"].split())
        if not normalized:
            continue
        if normalized not in unique_context_items:
            unique_context_items[normalized] = item

    sorted_items = list(unique_context_items.values())
    sorted_items.sort(
        key=lambda item: (
            item["distance"]
            if isinstance(item.get("distance"), (int, float))
            else float("inf"),
            str(item.get("metadata", {}).get("source", "")),
        )
    )

    for idx, item in enumerate(sorted_items, start=1):
        doc = item["document"]
        metadata = item["metadata"]
        mission_value = metadata.get("mission", "Unknown Mission")
        mission = mission_value.replace("_", " ").title()
        source = metadata.get("source", "Unknown Source")
        category_value = metadata.get(
            "document_category",
            metadata.get("category", "Unknown Category"),
        )
        category = category_value.replace("_", " ").title()
        distance = item.get("distance")
        score_text = (
            "n/a"
            if not isinstance(distance, (int, float))
            else "{:.4f}".format(distance)
        )

        source_header = " | ".join(
            [
                f"[{idx}] Mission: {mission}",
                f"Source: {source}",
                f"Category: {category}",
                f"Distance: {score_text}",
            ]
        )
        context_parts.append(source_header)

        truncated_doc = doc if len(doc) <= 500 else doc[:500] + "..."
        context_parts.append(truncated_doc)

    return "\n".join(context_parts)
