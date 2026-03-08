import os
from datetime import datetime
from pathlib import Path

import llm_client
import rag_client
from dotenv import load_dotenv


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    load_dotenv(dotenv_path=repo_root / ".env")

    openai_key = (
        os.getenv("CHROMA_OPENAI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_KEY")
    )
    if not openai_key:
        raise RuntimeError("No OpenAI key found in .env/environment")

    os.environ.setdefault("CHROMA_OPENAI_API_KEY", openai_key)
    os.environ.setdefault("OPENAI_API_KEY", openai_key)
    os.environ.setdefault("OPENAI_KEY", openai_key)

    collection, ok, error = rag_client.initialize_rag_system(
        "./chroma_db_openai", "nasa_space_missions_text"
    )
    if not ok:
        raise RuntimeError(f"RAG init failed: {error}")

    prompts = [
        "When was Apollo 11 mission?",
        "During Apollo 13, what emergency triggered the mission abort?",
        (
            "Which technical factors are discussed "
            "in the Challenger disaster documents?"
        ),
    ]

    records = []
    for prompt in prompts:
        result = rag_client.retrieve_documents(collection, prompt, n_results=4)
        context_items = (
            result.get("context_items", []) if isinstance(result, dict) else []
        )

        context = rag_client.format_context(
            [
                item.get("document", "")
                for item in context_items
                if item.get("document")
            ],
            [item.get("metadata", {}) for item in context_items],
            [item.get("distance") for item in context_items],
        )

        response = llm_client.generate_response(
            openai_key=openai_key,
            user_message=prompt,
            context=context,
            conversation_history=[],
            model="gpt-3.5-turbo",
        )

        sources = []
        for item in context_items[:4]:
            metadata = (
                item.get("metadata", {})
                if isinstance(item.get("metadata", {}), dict)
                else {}
            )
            mission = metadata.get("mission", "unknown")
            source = metadata.get("source", "unknown")
            sources.append("{}:{}".format(mission, source))

        records.append(
            {
                "question": prompt,
                "expected": response,
                "sources": sources,
            }
        )

    output_file = repo_root / "evaluation_dataset.txt"
    lines = [
        "Evaluation Dataset for Reproduction",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        "",
        (
            "Legen Sie eine Textdatei namens evaluation_dataset.txt bei, "
            "die einige Testfragen enthält,"
        ),
        "die verwendet wurden, sowie die erwarteten Antworten des Systems.",
        "",
    ]

    for index, item in enumerate(records, start=1):
        lines.append(f"Test Case {index}")
        lines.append(f"Question: {item['question']}")
        lines.append("Expected System Answer:")
        lines.append(item["expected"])
        lines.append("Retrieved Sources:")
        if item["sources"]:
            for source in item["sources"]:
                lines.append(f"- {source}")
        else:
            lines.append("- none")
        lines.append("-" * 80)

    output_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output_file.name} with {len(records)} test cases")


if __name__ == "__main__":
    main()
