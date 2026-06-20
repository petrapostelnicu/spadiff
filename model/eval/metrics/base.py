from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from PIL import Image


@dataclass
class MetricResult:
    """Result of a metric computation.

    Attributes:
        aggregate: Aggregated metric value(s) - either a single float or dict.
        per_image: Optional list of per-image values.
    """

    aggregate: float | dict[str, float]
    per_image: list[float] | None = None


class BaseMetric(ABC):
    """Abstract base class for evaluation metrics.

    All metrics must implement this interface to be used in experiments.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the metric name."""

    @property
    @abstractmethod
    def higher_is_better(self) -> bool:
        """Return True if higher values indicate better quality."""

    @abstractmethod
    def compute(
        self,
        generated_images: list[Image.Image],
        reference_images: list[Image.Image] | None = None,
        prompts: list[str] | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """Compute the metric value.

        Args:
            generated_images: List of generated images to evaluate.
            reference_images: Optional list of reference images (for FID, etc.).
            prompts: Optional list of prompts (for CLIP score, etc.).
            **kwargs: Additional metric-specific arguments.

        Returns:
            MetricResult containing aggregate and optional per-image values.
        """

    def reset(self) -> None:
        """Reset any internal state.

        Override in subclasses that maintain state.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
