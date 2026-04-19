"""Deterministic Q&A model for platform testing."""


def predict(input_data):
    """Return fixed answers for known questions."""
    # Normalize user input from dict format used by the platform.
    question = str((input_data or {}).get("input", "")).strip().lower()

    answers = {
        "what is 2+2?": "4",
        "what is the capital of france?": "Paris",
        "what color is the sky on a clear day?": "blue",
        "how many days are in a week?": "7",
        "who wrote hamlet?": "Shakespeare",
    }

    return answers.get(question, "I don't know")
