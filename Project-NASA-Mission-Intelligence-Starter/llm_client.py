from typing import Dict, List

from openai import OpenAI

SYSTEM_PROMPT = """
You are a NASA mission intelligence expert for historic missions,
including Apollo 11, Apollo 13, and Challenger.

Follow these rules strictly:
1. Use only the retrieved context provided in this request.
2. Do not invent facts, numbers, quotes, or sources.
3. Cite supporting sources when stating factual claims,
    using clear references to the provided source names.
4. If evidence is missing or insufficient, state exactly:
    "The available documents do not contain sufficient information
    to fully answer this question."
5. If you find conflicting information, don't attempt to resolve it.
    Instead, report the conflict clearly,
6. Preserve historical accuracy and use precise technical language.
7. Keep answers well structured for multi-part questions.

Conversation handling:
- Use prior turns to maintain continuity.
- Prefer the current request and current retrieved context
    when conflicts appear.
- Limit reasoning to context relevant to the user question.
""".strip()

INSUFFICIENT_CONTEXT_MESSAGE = (
    "The available documents do not contain sufficient information "
    "to fully answer this question."
)


def _context_is_insufficient(context: str) -> bool:
    """Return True if retrieved context is missing or placeholder-only."""
    if not context or not context.strip():
        return True

    normalized = context.strip().lower()
    placeholder_markers = [
        "no relevant documents found",
        "no context retrieved",
        "no retrieved context was provided",
    ]

    if any(marker in normalized for marker in placeholder_markers):
        return True

    header_only = "### context section"
    if normalized == header_only:
        return True

    condensed_lines = [
        line.strip() for line in normalized.splitlines() if line.strip()
    ]
    if condensed_lines == [header_only]:
        return True

    return False


def generate_response(
    openai_key: str,
    user_message: str,
    context: str,
    conversation_history: List[Dict],
    model: str = "gpt-3.5-turbo",
) -> str:
    """Generate response using OpenAI with context"""

    if not openai_key or not openai_key.strip():
        return "Error generating response: OpenAI API key is missing."
    if not user_message or not user_message.strip():
        return "Error generating response: User message is empty."

    if _context_is_insufficient(context):
        return INSUFFICIENT_CONTEXT_MESSAGE

    context_block = context.strip() if context else ""
    context_prompt = (
        "Use this retrieved context when relevant:\n" f"{context_block}"
        if context_block
        else "No retrieved context was provided."
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": context_prompt},
    ]

    for entry in conversation_history or []:
        role = entry.get("role") if isinstance(entry, dict) else None
        content = entry.get("content") if isinstance(entry, dict) else None
        valid_role = role in {"user", "assistant", "system"}
        if valid_role and isinstance(content, str):
            if content.strip():
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message.strip()})

    try:
        client = OpenAI(api_key=openai_key, timeout=30.0, max_retries=2)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )

        if not response.choices:
            return "Error generating response: No response choices returned."

        message_content = response.choices[0].message.content
        if not message_content:
            return "Error generating response: Empty response content."

        return message_content.strip()
    except Exception as exc:
        return f"Error generating response: {exc}"
