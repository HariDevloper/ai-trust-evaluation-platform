from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score


class ScoringEngine:
    def compute_classification_metrics(self, predictions: List[Any], ground_truth: List[Any]) -> Dict[str, float]:
        y_true = [str(v) for v in ground_truth]
        y_pred = [str(v) for v in predictions]
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }

    def compute_regression_metrics(self, predictions: List[Any], ground_truth: List[Any]) -> Dict[str, float]:
        y_true = np.array(ground_truth, dtype=float)
        y_pred = np.array(predictions, dtype=float)
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mse": mse,
            "rmse": rmse,
        }

    def compute_latency_metrics(self, latencies: List[float], throughput: float) -> Dict[str, float]:
        if not latencies:
            return {"latency_avg": 0.0, "latency_min": 0.0, "latency_max": 0.0, "throughput": throughput}
        arr = np.array(latencies, dtype=float)
        return {
            "latency_avg": float(np.mean(arr)),
            "latency_min": float(np.min(arr)),
            "latency_max": float(np.max(arr)),
            "throughput": float(throughput),
        }
