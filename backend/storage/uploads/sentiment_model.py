"""Simple keyword-based sentiment model for platform testing."""


def predict(input_data):
    """Classify input text as positive, negative, or neutral."""
    # Accept dict input like {"input": "text"} and fallback safely.
    text = str((input_data or {}).get("input", "")).strip().lower()

    positive_keywords = ("good", "great", "awesome", "love", "excellent", "amazing")
    negative_keywords = ("bad", "terrible", "hate", "awful", "horrible", "worst")

    if any(word in text for word in positive_keywords):
        return "positive"
    if any(word in text for word in negative_keywords):
        return "negative"
    return "neutral"
