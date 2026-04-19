"""Basic spam detector model for testing classification flow."""


def predict(input_data):
    """Classify email text as spam or ham."""
    # Handle dict input payload and normalize to lowercase text.
    text = str((input_data or {}).get("input", "")).strip().lower()

    spam_keywords = ("free", "win", "winner", "claim", "urgent", "click", "offer")
    if any(word in text for word in spam_keywords):
        return "spam"
    return "ham"
