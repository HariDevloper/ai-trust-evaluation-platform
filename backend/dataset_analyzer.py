from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class DatasetAnalysis:
    task_type: str
    input_fields: List[str]
    output_field: Optional[str]
    sample_count: int
    confidence: float
    sensitive_fields: List[str]
    field_types: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "input_fields": self.input_fields,
            "output_field": self.output_field,
            "sample_count": self.sample_count,
            "confidence": self.confidence,
            "sensitive_fields": self.sensitive_fields,
            "field_types": self.field_types,
        }


class DatasetAnalyzer:
    """Analyzes datasets and infers task/setup metadata."""

    OUTPUT_CANDIDATES = ("label", "labels", "target", "reference", "output", "answer", "expected")
    SENSITIVE_KEYWORDS = ("gender", "sex", "race", "ethnicity", "religion", "age")

    def analyze(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not dataset:
            return DatasetAnalysis(
                task_type="unknown",
                input_fields=[],
                output_field=None,
                sample_count=0,
                confidence=0.0,
                sensitive_fields=[],
                field_types={},
            ).to_dict()

        fields = list(dataset[0].keys())
        output_field = self._detect_output_field(fields)
        input_fields = [field for field in fields if field != output_field]
        field_types = self._infer_field_types(dataset, fields)
        task_type, confidence = self._detect_task_type(dataset, output_field, field_types)
        sensitive_fields = [field for field in fields if any(k in field.lower() for k in self.SENSITIVE_KEYWORDS)]

        return DatasetAnalysis(
            task_type=task_type,
            input_fields=input_fields,
            output_field=output_field,
            sample_count=len(dataset),
            confidence=confidence,
            sensitive_fields=sensitive_fields,
            field_types=field_types,
        ).to_dict()

    def _detect_output_field(self, fields: List[str]) -> Optional[str]:
        lowered = {f.lower(): f for f in fields}
        for candidate in self.OUTPUT_CANDIDATES:
            if candidate in lowered:
                return lowered[candidate]
        return None

    def _infer_field_types(self, dataset: List[Dict[str, Any]], fields: List[str]) -> Dict[str, str]:
        field_types: Dict[str, str] = {}
        sample_size = min(20, len(dataset))
        for field in fields:
            values = [dataset[i].get(field) for i in range(sample_size)]
            numeric_count = 0
            text_count = 0
            for value in values:
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric_count += 1
                elif isinstance(value, str):
                    text_count += 1

            if numeric_count >= max(1, sample_size // 2):
                field_types[field] = "numeric"
            elif text_count >= max(1, sample_size // 2):
                field_types[field] = "text"
            else:
                field_types[field] = "mixed"
        return field_types

    def _detect_task_type(
        self,
        dataset: List[Dict[str, Any]],
        output_field: Optional[str],
        field_types: Dict[str, str],
    ) -> tuple[str, float]:
        if not output_field:
            if any(t == "text" for t in field_types.values()):
                return "text_generation", 0.5
            return "unknown", 0.3

        values = [row.get(output_field) for row in dataset[:100] if row.get(output_field) is not None]
        if not values:
            return "unknown", 0.2

        unique_values = len(set(str(v) for v in values))
        numeric_values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        text_values = [v for v in values if isinstance(v, str)]

        if numeric_values and len(numeric_values) >= len(values) * 0.8:
            if unique_values <= 20:
                return "classification", 0.8
            return "regression", 0.9

        if text_values and len(text_values) >= len(values) * 0.8:
            avg_len = sum(len(v) for v in text_values) / len(text_values)
            if avg_len > 24:
                return "text_generation", 0.9
            return "classification", 0.7

        return "unknown", 0.4
