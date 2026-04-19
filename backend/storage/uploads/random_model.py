"""Random-output model to test consistency metrics."""

import random


def predict(input_data):
    """Return a random label each call."""
    # Input is accepted for interface compatibility; output is intentionally random.
    _ = (input_data or {}).get("input", "")
    return random.choice(["yes", "no", "maybe"])
