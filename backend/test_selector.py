from __future__ import annotations

from typing import Any, Dict, List


class TestSelector:
    def select_tests(self, analysis: Dict[str, Any], has_labels: bool = True) -> Dict[str, List[str]]:
        tests: List[str] = ["latency", "throughput", "consistency", "toxicity"]
        task_type = analysis.get("task_type", "unknown")
        sensitive_fields = analysis.get("sensitive_fields", [])

        if has_labels and analysis.get("output_field"):
            if task_type == "classification":
                tests.extend(["accuracy", "precision", "recall"])
            elif task_type == "regression":
                tests.extend(["mae", "mse", "rmse"])

        if task_type == "text_generation":
            tests.append("hallucination")

        if sensitive_fields:
            tests.append("bias")

        unique_tests = sorted(set(tests))
        return {"selected_tests": unique_tests}
