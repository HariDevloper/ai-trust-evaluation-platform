from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


class EvaluationEngine:
    def __init__(self, adapter, dataset: List[Dict[str, Any]]):
        self.adapter = adapter
        self.dataset = dataset

    def run_evaluation(
        self,
        input_fields: List[str],
        output_field: Optional[str] = None,
        sample_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        predictions: List[Any] = []
        ground_truth: List[Any] = []
        detailed_outputs: List[Dict[str, Any]] = []
        latencies: List[float] = []
        errors = 0

        rows = self.dataset[:sample_limit] if sample_limit else self.dataset
        started = time.perf_counter()
        for index, row in enumerate(rows):
            payload = {field: row.get(field) for field in input_fields}
            if "input" in row and "input" not in payload:
                payload["input"] = row["input"]

            row_started = time.perf_counter()
            error_message = None
            try:
                prediction = self.adapter.predict(payload)
            except Exception as exc:  # noqa: BLE001
                prediction = None
                error_message = str(exc)
                errors += 1
            row_latency = time.perf_counter() - row_started
            latencies.append(row_latency)

            expected = row.get(output_field) if output_field else None
            predictions.append(prediction)
            if output_field:
                ground_truth.append(expected)

            detailed_outputs.append(
                {
                    "index": index,
                    "input": payload,
                    "expected": expected,
                    "prediction": prediction,
                    "latency": row_latency,
                    "error": error_message,
                }
            )

        total_duration = time.perf_counter() - started
        throughput = (len(rows) / total_duration) if total_duration > 0 else 0.0

        return {
            "predictions": predictions,
            "ground_truth": ground_truth,
            "latencies": latencies,
            "sample_count": len(rows),
            "errors": errors,
            "throughput": throughput,
            "detailed_outputs": detailed_outputs,
        }
