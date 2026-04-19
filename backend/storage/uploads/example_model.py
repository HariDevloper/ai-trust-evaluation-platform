def predict(input_data):
    text = str(input_data.get("input", "")).lower()
    if "2+2" in text:
        return "4"
    if "capital of france" in text:
        return "Paris"
    return "unknown"
