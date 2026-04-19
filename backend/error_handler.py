from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class PlatformError(Exception):
    code: str
    message: str
    details: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details or {}}


class ErrorHandler:
    @staticmethod
    def validate_dataset(dataset: List[Dict[str, Any]]) -> Tuple[bool, str | None]:
        if not isinstance(dataset, list):
            return False, "Dataset must be a list of objects"
        if not dataset:
            return False, "Dataset cannot be empty"
        if not isinstance(dataset[0], dict):
            return False, "Dataset rows must be objects"
        return True, None

    @staticmethod
    def validate_model_config(config: Dict[str, Any]) -> Tuple[bool, str | None]:
        if not isinstance(config, dict):
            return False, "Model config must be an object"
        model_type = (config.get("type") or "local").lower()
        target_config = config.get("config", config)
        if model_type == "api" and not (target_config.get("endpoint") or target_config.get("url")):
            return False, "API model config requires endpoint/url"
        if model_type == "local" and not (target_config.get("path") or target_config.get("model_path")):
            return False, "Local model config requires path/model_path"
        return True, None

    @staticmethod
    def raise_dataset_error(message: str) -> None:
        raise PlatformError(code="DATASET_VALIDATION_ERROR", message=message)

    @staticmethod
    def raise_model_error(message: str) -> None:
        raise PlatformError(code="MODEL_VALIDATION_ERROR", message=message)
