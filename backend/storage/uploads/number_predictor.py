"""Simple numeric model that doubles the input value."""


def predict(input_data):
    """Return input * 2 as a number."""
    # Read numeric input from dict format {"input": value}.
    value = (input_data or {}).get("input", 0)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0

    doubled = number * 2
    # Keep integer output clean when possible.
    return int(doubled) if doubled.is_integer() else doubled
