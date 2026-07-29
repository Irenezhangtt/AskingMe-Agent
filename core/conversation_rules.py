"""Deterministic conversation-level rules that do not require an LLM."""

import re


def is_conversation_memory_query(message: str) -> bool:
    """Return whether the user is recalling or continuing the current conversation."""
    text = re.sub(r"\s+", " ", (message or "").strip().lower())
    if not text:
        return False

    fixed_phrases = [
        "do you remember",
        "remember what i just",
        "what were we just discussing",
        "what did we just discuss",
        "summarize our previous conversation",
        "summarize what we just discussed",
        "repeat what i just said",
        "repeat my last question",
        "continue our previous topic",
        "continue where we left off",
        "previous message",
        "previous question",
    ]
    if any(phrase in text for phrase in fixed_phrases):
        return True

    # Covers colloquial variants of asking what the user said or asked earlier.
    time_markers = ("just", "previously", "before", "earlier", "last")
    speech_markers = ("ask", "say", "said", "discuss", "mention")
    recall_markers = ("what", "which", "content", "question", "topic")
    return (
        any(marker in text for marker in time_markers)
        and any(marker in text for marker in speech_markers)
        and any(marker in text for marker in recall_markers)
    )
