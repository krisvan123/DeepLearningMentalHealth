"""Helper utilities for MindCare chatbot."""

import os
from pathlib import Path
from typing import Dict, Tuple

from config.config import (
    CLASS_LABELS,
    MODEL_PATH,
    RESPONSE_DIR,
    RESPONSE_MAP,
    TOKENIZER_PATH,
)


def validate_user_input(text: str, max_chars: int = 4000) -> Tuple[bool, str]:
    """Validate user input.

    Returns:
        (is_valid, error_message or clean_text)
    """
    if text is None:
        return False, "Message cannot be empty."

    stripped = text.strip()
    if not stripped:
        return False, "Message cannot be blank."

    if len(stripped) > max_chars:
        return (
            False,
            f"Message is too long ({len(stripped)} characters). Please keep it under {max_chars} characters.",
        )

    return True, stripped


def safe_read_text(path: Path) -> str:
    """Read file content with UTF-8 encoding and fallback encodings."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def check_system_status() -> Dict[str, bool]:
    """Check existence of all core system components.

    Returns a status dictionary with boolean flags.
    """
    model_exists = MODEL_PATH.exists()
    tokenizer_exists = TOKENIZER_PATH.exists()
    response_dir_exists = RESPONSE_DIR.exists() and RESPONSE_DIR.is_dir()

    missing_responses = []
    if response_dir_exists:
        for label, filename in RESPONSE_MAP.items():
            if not (RESPONSE_DIR / filename).exists():
                missing_responses.append(label)

    responses_ready = response_dir_exists and (len(missing_responses) == 0)

    ready = model_exists and tokenizer_exists and responses_ready

    return {
        "model_exists": model_exists,
        "tokenizer_exists": tokenizer_exists,
        "response_dir_exists": response_dir_exists,
        "responses_ready": responses_ready,
        "missing_responses": missing_responses,
        "ready": ready,
    }
