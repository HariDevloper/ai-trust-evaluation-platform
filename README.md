# AI Trust & Performance Evaluation Platform

MVP platform for evaluating AI/ML models on performance and trust metrics.

## Structure

- `backend/` FastAPI API server and evaluation modules
- `backend/storage/uploads` uploaded/example datasets and models
- `backend/storage/jobs` job results JSON files
- `frontend/` React + TypeScript dashboard

## Backend Setup

```bash
cd backend
python -m pip install -r requirements.txt
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### Backend API

- `POST /api/upload-dataset` (multipart file or JSON payload)
- `POST /api/upload-model` (multipart file or JSON payload)
- `POST /api/run-test`
- `GET /api/status/{job_id}`
- `GET /api/results/{job_id}`
- `GET /api/jobs`

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Set `frontend/.env.example` values in `.env` if needed:

```bash
VITE_API_BASE_URL=http://localhost:8000
```

## Example Data

- CSV dataset: `backend/storage/uploads/example_dataset.csv`
- JSON dataset: `backend/storage/uploads/example_dataset.json`
- Local model: `backend/storage/uploads/example_model.py`
- Sample result: `backend/storage/jobs/sample_result.json`

## Ready-to-Use Test Models and Datasets

All files are available in `backend/storage/uploads/` and can be uploaded directly in the UI.

- `sentiment_model.py` ↔ `sentiment_dataset.csv`
- `qa_model.py` ↔ `qa_dataset.csv`
- `number_predictor.py` ↔ `numbers_dataset.csv`
- `spam_detector.py` ↔ `spam_dataset.csv`
- `random_model.py` ↔ `unlabeled_dataset.csv`

See `backend/storage/uploads/QUICK_START.md` for step-by-step testing instructions.
