"""
Main FastAPI application for AI Trust and Performance Evaluation Platform.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from typing import List, Dict, Any, Optional

from dataset_analyzer import DatasetAnalyzer
from model_adapter import AdapterFactory
from test_selector import TestSelector
from evaluation_engine import EvaluationEngine
from scoring_engine import ScoringEngine
from trust_calculator import TrustCalculator
from storage_manager import StorageManager
from error_handler import ErrorHandler, PlatformError

# Initialize app
app = FastAPI(
    title="AI Trust and Performance Evaluation Platform",
    description="Comprehensive evaluation platform for machine learning models",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
storage_manager = StorageManager()
dataset_analyzer = DatasetAnalyzer()
test_selector = TestSelector()
scoring_engine = ScoringEngine()
trust_calculator = TrustCalculator()

# Pydantic models
class DatasetUploadRequest(BaseModel):
    name: str
    data: List[Dict[str, Any]]

class ModelUploadRequest(BaseModel):
    name: str
    type: str
    config: Dict[str, Any]

class EvaluationRequest(BaseModel):
    dataset_id: str
    model_id: str
    has_labels: bool = True
    sample_limit: Optional[int] = None

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "AI Trust and Performance Evaluation Platform",
        "version": "1.0.0",
        "status": "operational"
    }

@app.post("/api/upload-dataset")
async def upload_dataset(request: DatasetUploadRequest):
    """Upload and analyze a dataset."""
    try:
        is_valid, error = ErrorHandler.validate_dataset(request.data)
        if not is_valid:
            ErrorHandler.raise_dataset_error(error)
        
        dataset_id = storage_manager.save_dataset(request.data, request.name)
        analysis = dataset_analyzer.analyze(request.data)
        
        return {
            "dataset_id": dataset_id,
            "dataset_name": request.name,
            "analysis": analysis
        }
    
    except PlatformError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.post("/api/upload-model")
async def upload_model(request: ModelUploadRequest):
    """Upload and validate model configuration."""
    try:
        is_valid, error = ErrorHandler.validate_model_config(request.config)
        if not is_valid:
            ErrorHandler.raise_model_error(error)
        
        model_id = storage_manager.save_model_config(request.config, request.name)
        
        return {
            "model_id": model_id,
            "model_name": request.name,
            "type": request.type
        }
    
    except PlatformError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.post("/api/run-test")
async def run_test(request: EvaluationRequest):
    """Start evaluation job."""
    try:
        dataset = storage_manager.load_dataset(request.dataset_id)
        model_config = storage_manager.load_model_config(request.model_id)
        
        if not dataset:
            raise ValueError(f"Dataset {request.dataset_id} not found")
        if not model_config:
            raise ValueError(f"Model {request.model_id} not found")
        
        analysis = dataset_analyzer.analyze(dataset)
        adapter = AdapterFactory.create_adapter(model_config)
        
        test_config = test_selector.select_tests(
            analysis,
            has_labels=request.has_labels
        )
        
        job_id = storage_manager.create_job_record(
            request.dataset_id,
            request.model_id,
            analysis,
            model_config,
            test_config["selected_tests"]
        )
        
        engine = EvaluationEngine(adapter, dataset)
        eval_results = engine.run_evaluation(
            analysis['input_fields'],
            analysis['output_field'],
            request.sample_limit
        )
        
        metrics = {}
        
        if analysis['output_field'] and eval_results['ground_truth']:
            if analysis['task_type'] == 'classification':
                metrics['classification'] = scoring_engine.compute_classification_metrics(
                    eval_results['predictions'],
                    eval_results['ground_truth']
                )
            elif analysis['task_type'] == 'regression':
                metrics['regression'] = scoring_engine.compute_regression_metrics(
                    eval_results['predictions'],
                    eval_results['ground_truth']
                )
        
        trust_report = trust_calculator.calculate_trust_score(
            metrics.get('classification', metrics.get('regression', {})),
            {
                'consistency_score': 0.95,
                'toxicity_score': 0.02
            }
        )
        
        job_data = storage_manager.load_job(job_id)
        job_data['status'] = 'completed'
        job_data['results'] = {
            'evaluation': eval_results,
            'metrics': metrics,
            'trust_report': trust_report
        }
        storage_manager.save_job(job_id, job_data)
        
        return {
            "job_id": job_id,
            "status": "completed",
            "results": job_data['results']
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Get job status."""
    try:
        job = storage_manager.load_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail={"error": "Job not found"})
        
        return {
            "job_id": job_id,
            "status": job['status'],
            "created_at": job['created_at'],
            "updated_at": job['updated_at']
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    """Get full evaluation results."""
    try:
        job = storage_manager.load_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail={"error": "Job not found"})
        
        return {
            "job_id": job_id,
            "status": job['status'],
            "config": job['config'],
            "results": job['results']
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get("/api/jobs")
async def list_jobs(status: Optional[str] = None, limit: int = 100):
    """List all evaluation jobs."""
    try:
        jobs = storage_manager.list_jobs(status=status, limit=limit)
        return {"jobs": jobs, "total": len(jobs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its results."""
    try:
        success = storage_manager.delete_job(job_id)
        if not success:
            raise HTTPException(status_code=404, detail={"error": "Job not found"})
        
        return {"job_id": job_id, "deleted": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)