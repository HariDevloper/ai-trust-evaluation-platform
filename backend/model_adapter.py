from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict

import requests


class BaseAdapter:
    def predict(self, input_data: Dict[str, Any]) -> Any:  # pragma: no cover - interface
        raise NotImplementedError


class APIAdapter(BaseAdapter):
    def __init__(self, endpoint: str, method: str = "POST", headers: Dict[str, str] | None = None, timeout: int = 30):
        self.endpoint = endpoint
        self.method = method.upper()
        self.headers = headers or {}
        self.timeout = timeout

    def predict(self, input_data: Dict[str, Any]) -> Any:
        response = requests.request(
            self.method,
            self.endpoint,
            headers=self.headers,
            json={"input": input_data},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
        if isinstance(payload, dict):
            return payload.get("output", payload.get("prediction", payload))
        return payload


class LocalAdapter(BaseAdapter):
    def __init__(self, model_path: str, callable_name: str = "predict"):
        self.model_path = Path(model_path)
        self.callable_name = callable_name
        self.predict_fn = self._load_predictor()

    def _load_predictor(self):
        if self.model_path.suffix == ".json":
            lookup = json.loads(self.model_path.read_text(encoding="utf-8"))

            def _predict(data: Dict[str, Any]) -> Any:
                key = str(data)
                return lookup.get(key, lookup.get(data.get("input"), "unknown"))

            return _predict

        spec = importlib.util.spec_from_file_location("local_model_module", str(self.model_path))
        if not spec or not spec.loader:
            raise ValueError(f"Unable to load local model module from {self.model_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        predict_fn = getattr(module, self.callable_name, None)
        if not callable(predict_fn):
            raise ValueError(f"Callable '{self.callable_name}' not found in local model module")
        return predict_fn

    def predict(self, input_data: Dict[str, Any]) -> Any:
        return self.predict_fn(input_data)


class AdapterFactory:
    @staticmethod
    def create_adapter(model_config: Dict[str, Any]) -> BaseAdapter:
        model_type = (model_config.get("type") or "local").lower()
        config = model_config.get("config", model_config)
        if model_type == "api":
            endpoint = config.get("endpoint") or config.get("url")
            if not endpoint:
                raise ValueError("API model requires 'endpoint' or 'url'")
            return APIAdapter(
                endpoint=endpoint,
                method=config.get("method", "POST"),
                headers=config.get("headers", {}),
                timeout=int(config.get("timeout", 30)),
            )
        if model_type == "local":
            model_path = config.get("path") or config.get("model_path")
            if not model_path:
                raise ValueError("Local model requires 'path' or 'model_path'")
            return LocalAdapter(model_path=model_path, callable_name=config.get("callable_name", "predict"))
        raise ValueError(f"Unsupported model type: {model_type}")
