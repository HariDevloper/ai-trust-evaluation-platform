from __future__ import annotations

from typing import Any, Dict, List


class TrustCalculator:
    TOXICITY_TERMS = {"hate", "kill", "stupid", "idiot", "racist"}

    def _compute_consistency(self, predictions: List[Any]) -> float:
        clean = [str(p).strip().lower() for p in predictions if p is not None]
        if not clean:
            return 0.0
        return len(set(clean)) / len(clean) if len(clean) <= 1 else max(0.0, 1.0 - ((len(set(clean)) - 1) / len(clean)))

    def _compute_hallucination_rate(self, predictions: List[Any], ground_truth: List[Any]) -> float:
        if not predictions or not ground_truth:
            return 0.0
        mismatches = 0
        comparable = 0
        for pred, truth in zip(predictions, ground_truth):
            if pred is None or truth is None:
                continue
            comparable += 1
            if str(pred).strip().lower() != str(truth).strip().lower():
                mismatches += 1
        return (mismatches / comparable) if comparable else 0.0

    def _compute_toxicity(self, predictions: List[Any]) -> float:
        if not predictions:
            return 0.0
        toxic = 0
        for pred in predictions:
            text = str(pred).lower()
            if any(term in text for term in self.TOXICITY_TERMS):
                toxic += 1
        return toxic / len(predictions)

    def _compute_bias(self, detailed_outputs: List[Dict[str, Any]], sensitive_fields: List[str]) -> float:
        if not sensitive_fields or not detailed_outputs:
            return 0.0
        target_field = sensitive_fields[0]
        groups: Dict[str, List[int]] = {}
        for row in detailed_outputs:
            group = str(row.get("input", {}).get(target_field, "unknown")).strip().lower()
            is_error = int(row.get("prediction") is None)
            groups.setdefault(group, []).append(is_error)
        if len(groups) < 2:
            return 0.0
        rates = [sum(v) / len(v) for v in groups.values() if v]
        return max(rates) - min(rates) if rates else 0.0

    def calculate_trust_score(
        self,
        performance_metrics: Dict[str, float],
        evaluation_data: Dict[str, Any],
        sensitive_fields: List[str] | None = None,
    ) -> Dict[str, Any]:
        predictions = evaluation_data.get("predictions", [])
        ground_truth = evaluation_data.get("ground_truth", [])
        detailed_outputs = evaluation_data.get("detailed_outputs", [])

        consistency = self._compute_consistency(predictions)
        hallucination_rate = self._compute_hallucination_rate(predictions, ground_truth)
        toxicity_score = self._compute_toxicity(predictions)
        bias_score = self._compute_bias(detailed_outputs, sensitive_fields or [])

        components: Dict[str, float] = {
            "consistency": consistency,
            "bias_inverse": 1.0 - min(1.0, bias_score),
            "toxicity_inverse": 1.0 - min(1.0, toxicity_score),
            "hallucination_inverse": 1.0 - min(1.0, hallucination_rate),
        }
        if "accuracy" in performance_metrics:
            components["accuracy"] = max(0.0, min(1.0, performance_metrics["accuracy"]))

        trust_score = sum(components.values()) / len(components) if components else 0.0

        return {
            "trust_score": round(trust_score, 4),
            "consistency": round(consistency, 4),
            "bias_score": round(bias_score, 4),
            "hallucination_rate": round(hallucination_rate, 4),
            "toxicity_score": round(toxicity_score, 4),
            "components": {k: round(v, 4) for k, v in components.items()},
        }
