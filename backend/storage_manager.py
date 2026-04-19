from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional


class StorageManager:
    def __init__(self, base_path: str | None = None):
        root = Path(base_path) if base_path else Path(__file__).resolve().parent / "storage"
        self.base_path = root
        self.uploads_path = self.base_path / "uploads"
        self.jobs_path = self.base_path / "jobs"
        self.models_path = self.base_path / "models"
        self.datasets_path = self.base_path / "datasets"
        self._lock = Lock()
        for path in (self.uploads_path, self.jobs_path, self.models_path, self.datasets_path):
            path.mkdir(parents=True, exist_ok=True)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        with self._lock:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _read_json(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save_dataset(self, dataset: List[Dict[str, Any]], name: str) -> str:
        dataset_id = str(uuid.uuid4())
        self._write_json(self.datasets_path / f"{dataset_id}.json", {"id": dataset_id, "name": name, "data": dataset})
        return dataset_id

    def save_dataset_file(self, source_path: str, original_name: str) -> str:
        dataset_id = str(uuid.uuid4())
        ext = Path(original_name).suffix or ".dat"
        target = self.uploads_path / f"{dataset_id}{ext}"
        shutil.copyfile(source_path, target)
        self._write_json(
            self.datasets_path / f"{dataset_id}.json",
            {"id": dataset_id, "name": original_name, "file_path": str(target)},
        )
        return dataset_id

    def load_dataset(self, dataset_id: str) -> Optional[List[Dict[str, Any]]]:
        payload = self._read_json(self.datasets_path / f"{dataset_id}.json")
        if not payload:
            return None
        if "data" in payload:
            return payload["data"]
        file_path = payload.get("file_path")
        if not file_path:
            return None
        path = Path(file_path)
        if not path.exists():
            return None
        if path.suffix.lower() == ".json":
            content = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(content, dict) and "samples" in content:
                return content["samples"]
            if isinstance(content, list):
                return content
            return []
        if path.suffix.lower() == ".csv":
            import pandas as pd

            return pd.read_csv(path).to_dict(orient="records")
        return None

    def save_model_config(self, config: Dict[str, Any], name: str, model_type: str = "local") -> str:
        model_id = str(uuid.uuid4())
        payload = {"id": model_id, "name": name, "type": model_type, "config": config}
        self._write_json(self.models_path / f"{model_id}.json", payload)
        return model_id

    def save_model_file(self, source_path: str, original_name: str, model_type: str = "local") -> str:
        model_id = str(uuid.uuid4())
        ext = Path(original_name).suffix or ".bin"
        target = self.uploads_path / f"{model_id}{ext}"
        shutil.copyfile(source_path, target)
        payload = {"id": model_id, "name": original_name, "type": model_type, "config": {"path": str(target)}}
        self._write_json(self.models_path / f"{model_id}.json", payload)
        return model_id

    def load_model_config(self, model_id: str) -> Optional[Dict[str, Any]]:
        return self._read_json(self.models_path / f"{model_id}.json")

    def create_job_record(
        self,
        dataset_id: str,
        model_id: str,
        analysis: Dict[str, Any],
        model_config: Dict[str, Any],
        selected_tests: List[str],
    ) -> str:
        job_id = str(uuid.uuid4())
        now = self._now()
        payload = {
            "job_id": job_id,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "config": {
                "dataset_id": dataset_id,
                "model_id": model_id,
                "dataset_analysis": analysis,
                "model_type": model_config.get("type", "local"),
                "task_type": analysis.get("task_type", "unknown"),
                "selected_tests": selected_tests,
            },
            "results": None,
            "error": None,
        }
        self._write_json(self.jobs_path / f"{job_id}.json", payload)
        return job_id

    def load_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._read_json(self.jobs_path / f"{job_id}.json")

    def save_job(self, job_id: str, payload: Dict[str, Any]) -> None:
        payload["updated_at"] = self._now()
        self._write_json(self.jobs_path / f"{job_id}.json", payload)

    def update_job_status(self, job_id: str, status: str, error: str | None = None) -> None:
        job = self.load_job(job_id)
        if not job:
            return
        job["status"] = status
        if error:
            job["error"] = error
        self.save_job(job_id, job)

    def list_jobs(self, status: str | None = None, limit: int = 100) -> List[Dict[str, Any]]:
        jobs: List[Dict[str, Any]] = []
        for path in sorted(self.jobs_path.glob("*.json"), reverse=True):
            payload = self._read_json(path)
            if not payload:
                continue
            if status and payload.get("status") != status:
                continue
            jobs.append(
                {
                    "job_id": payload.get("job_id"),
                    "status": payload.get("status"),
                    "created_at": payload.get("created_at"),
                    "updated_at": payload.get("updated_at"),
                    "task_type": payload.get("config", {}).get("task_type"),
                    "trust_score": (payload.get("results") or {}).get("trust_score"),
                }
            )
            if len(jobs) >= limit:
                break
        return jobs

    def delete_job(self, job_id: str) -> bool:
        path = self.jobs_path / f"{job_id}.json"
        if not path.exists():
            return False
        path.unlink()
        return True
