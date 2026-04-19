"""Main FastAPI application for AI Trust and Performance Evaluation Platform."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Thread
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from dataset_analyzer import DatasetAnalyzer
from error_handler import ErrorHandler, PlatformError
from evaluation_engine import EvaluationEngine
from model_adapter import AdapterFactory
from scoring_engine import ScoringEngine
from storage_manager import StorageManager
from test_selector import TestSelector
from trust_calculator import TrustCalculator

app = FastAPI(
    title="AI Trust and Performance Evaluation Platform",
    description="Comprehensive evaluation platform for machine learning models",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

storage_manager = StorageManager()
dataset_analyzer = DatasetAnalyzer()
test_selector = TestSelector()
scoring_engine = ScoringEngine()
trust_calculator = TrustCalculator()


class DatasetUploadRequest(BaseModel):
    name: str
    data: List[Dict[str, Any]]


class ModelUploadRequest(BaseModel):
    name: str
    type: str = "local"
    config: Dict[str, Any]


class EvaluationRequest(BaseModel):
    dataset_id: str
    model_id: str
    has_labels: bool = True
    sample_limit: Optional[int] = None


@app.get("/")
async def root():
    return {"name": "AI Trust and Performance Evaluation Platform", "version": "1.0.0", "status": "operational"}


@app.post("/api/upload-dataset")
async def upload_dataset(
    request: Request,
    file: UploadFile | None = File(default=None),
    name: str | None = Form(default=None),
):
    try:
        if file is not None:
            with NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(await file.read())
                temp_file.flush()
                dataset_id = storage_manager.save_dataset_file(temp_file.name, file.filename or "dataset.json")
            data = storage_manager.load_dataset(dataset_id) or []
            analysis = dataset_analyzer.analyze(data)
            return {"dataset_id": dataset_id, "dataset_name": file.filename, "analysis": analysis}

        payload = await request.json()
        body = DatasetUploadRequest(**payload)
        is_valid, error = ErrorHandler.validate_dataset(body.data)
        if not is_valid:
            ErrorHandler.raise_dataset_error(error or "Invalid dataset")
        dataset_id = storage_manager.save_dataset(body.data, body.name)
        analysis = dataset_analyzer.analyze(body.data)
        return {"dataset_id": dataset_id, "dataset_name": body.name, "analysis": analysis}
    except PlatformError as exc:
        raise HTTPException(status_code=400, detail=exc.to_dict()) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc


@app.post("/api/upload-model")
async def upload_model(
    request: Request,
    file: UploadFile | None = File(default=None),
    name: str | None = Form(default=None),
    model_type: str = Form(default="local"),
):
    try:
        if file is not None:
            with NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(await file.read())
                temp_file.flush()
                model_id = storage_manager.save_model_file(temp_file.name, file.filename or "model.py", model_type)
            return {"model_id": model_id, "model_name": name or file.filename, "type": model_type}

        payload = await request.json()
        body = ModelUploadRequest(**payload)
        is_valid, error = ErrorHandler.validate_model_config({"type": body.type, "config": body.config})
        if not is_valid:
            ErrorHandler.raise_model_error(error or "Invalid model configuration")
        model_id = storage_manager.save_model_config(body.config, body.name, body.type)
        return {"model_id": model_id, "model_name": body.name, "type": body.type}
    except PlatformError as exc:
        raise HTTPException(status_code=400, detail=exc.to_dict()) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc


def _run_job(job_id: str, dataset_id: str, model_id: str, has_labels: bool, sample_limit: Optional[int]):
    try:
        storage_manager.update_job_status(job_id, "running")
        dataset = storage_manager.load_dataset(dataset_id)
        model_config = storage_manager.load_model_config(model_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        if not model_config:
            raise ValueError(f"Model {model_id} not found")

        analysis = dataset_analyzer.analyze(dataset)
        adapter = AdapterFactory.create_adapter(model_config)
        engine = EvaluationEngine(adapter, dataset)
        eval_results = engine.run_evaluation(analysis["input_fields"], analysis["output_field"], sample_limit)

        performance: Dict[str, float] = scoring_engine.compute_latency_metrics(
            eval_results["latencies"],
            eval_results["throughput"],
        )
        if has_labels and analysis.get("output_field") and eval_results.get("ground_truth"):
            if analysis["task_type"] == "classification":
                performance.update(
                    scoring_engine.compute_classification_metrics(
                        eval_results["predictions"],
                        eval_results["ground_truth"],
                    )
                )
            elif analysis["task_type"] == "regression":
                performance.update(
                    scoring_engine.compute_regression_metrics(
                        eval_results["predictions"],
                        eval_results["ground_truth"],
                    )
                )

        trust = trust_calculator.calculate_trust_score(performance, eval_results, analysis.get("sensitive_fields", []))
        job = storage_manager.load_job(job_id) or {}
        job["status"] = "completed"
        job["results"] = {
            "performance": performance,
            "trust": {
                "consistency": trust["consistency"],
                "bias_score": trust["bias_score"],
                "hallucination_rate": trust["hallucination_rate"],
                "toxicity_score": trust["toxicity_score"],
            },
            "trust_score": trust["trust_score"],
            "detailed_outputs": eval_results["detailed_outputs"],
            "latencies": eval_results["latencies"],
            "errors": eval_results["errors"],
            "sample_count": eval_results["sample_count"],
        }
        storage_manager.save_job(job_id, job)
    except Exception as exc:  # noqa: BLE001
        storage_manager.update_job_status(job_id, "failed", str(exc))


@app.post("/api/run-test")
async def run_test(request: EvaluationRequest):
    try:
        dataset = storage_manager.load_dataset(request.dataset_id)
        model_config = storage_manager.load_model_config(request.model_id)
        if not dataset:
            raise HTTPException(status_code=404, detail={"error": f"Dataset {request.dataset_id} not found"})
        if not model_config:
            raise HTTPException(status_code=404, detail={"error": f"Model {request.model_id} not found"})

        analysis = dataset_analyzer.analyze(dataset)
        test_config = test_selector.select_tests(analysis, has_labels=request.has_labels)
        job_id = storage_manager.create_job_record(
            request.dataset_id,
            request.model_id,
            analysis,
            model_config,
            test_config["selected_tests"],
        )

        thread = Thread(
            target=_run_job,
            args=(job_id, request.dataset_id, request.model_id, request.has_labels, request.sample_limit),
            daemon=True,
        )
        thread.start()
        return {"job_id": job_id, "status": "queued", "selected_tests": test_config["selected_tests"]}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    job = storage_manager.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail={"error": "Job not found"})
    return {
        "job_id": job_id,
        "status": job["status"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "error": job.get("error"),
    }


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    job = storage_manager.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail={"error": "Job not found"})
    return {
        "job_id": job_id,
        "status": job["status"],
        "config": job["config"],
        "results": job.get("results"),
        "error": job.get("error"),
    }


@app.get("/api/jobs")
async def list_jobs(status: Optional[str] = None, limit: int = 100):
    jobs = storage_manager.list_jobs(status=status, limit=limit)
    return {"jobs": jobs, "total": len(jobs)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
