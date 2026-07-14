from __future__ import annotations

import re


TASK_PROMPT = "Send a warm hello-world message for a first M365 + Foundry RLE collaboration demo."
SUCCESS_HINT = "The message must include both 'hello' and 'M365'."


def normalize_message(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).lower()


def evaluate_message(message: str) -> tuple[bool, str]:
    normalized = normalize_message(message)
    has_hello = "hello" in normalized
    has_m365 = "m365" in normalized or "microsoft 365" in normalized
    if has_hello and has_m365:
        return True, "Accepted: the message includes a hello greeting and an M365 reference."

    missing = []
    if not has_hello:
        missing.append("hello")
    if not has_m365:
        missing.append("M365")
    return False, f"Missing required term(s): {', '.join(missing)}. {SUCCESS_HINT}"