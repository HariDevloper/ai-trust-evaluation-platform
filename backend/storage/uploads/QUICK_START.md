# Quick Start: Ready-to-Use Test Assets

All files in this folder are upload-ready for the platform.

## Model and Dataset Pairings

1. `sentiment_model.py` + `sentiment_dataset.csv`
2. `qa_model.py` + `qa_dataset.csv`
3. `number_predictor.py` + `numbers_dataset.csv`
4. `spam_detector.py` + `spam_dataset.csv`
5. `random_model.py` + `unlabeled_dataset.csv`

## How to Test

1. Start backend and frontend.
2. In the app, upload one CSV dataset from this folder.
3. Upload the matching `.py` model from this folder (not `.pkl`).
4. Run evaluation to see results in the dashboard.

## Notes

- All models implement `predict(input_data)`.
- All models accept dict payloads like `{\"input\": \"some value\"}`.
- Outputs are plain strings or numbers for simple debugging.
