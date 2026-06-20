import logging
from typing import Any

import numpy as np
import torch
from PIL import Image

from .base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


class ImageRewardMetric(BaseMetric):
    """ImageReward metric for predicting human preference.

    ImageReward is trained on human preference data to score
    text-to-image generation quality. Higher scores indicate
    images more aligned with human preferences.

    Requires the image-reward package
    """

    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 16,
    ):
        """Initialize ImageReward metric.

        Args:
            device: Device for computation.
            batch_size: Batch size for processing.
        """
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _load_model(self) -> None:
        """Load ImageReward model."""
        if self._model is not None:
            return

        try:
            import ImageReward as IR
        except ImportError:
            raise ImportError(
                "ImageReward is required. "
                "Install with: pip install image-reward"
            )

        logger.info("Loading ImageReward model...")
        self._model = IR.load("ImageReward-v1.0", device=self.device)
        logger.info("ImageReward model loaded successfully")

    @property
    def name(self) -> str:
        return "image_reward"

    @property
    def higher_is_better(self) -> bool:
        return True

    @torch.no_grad()
    def _score_single(self, image: Image.Image, prompt: str) -> float:
        """Compute ImageReward for a single image-prompt pair."""
        score = self._model.score(prompt, image)
        return float(score)

    def compute(
        self,
        generated_images: list[Image.Image],
        reference_images: list[Image.Image] | None = None,
        prompts: list[str] | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """Compute ImageReward for generated images.

        Args:
            generated_images: List of generated images.
            reference_images: Not used.
            prompts: List of text prompts (required).

        Returns:
            MetricResult with aggregate stats and per-image scores.

        Raises:
            ValueError: If prompts is None or length doesn't match.
        """
        if prompts is None:
            raise ValueError("ImageReward requires prompts")
        if len(prompts) != len(generated_images):
            raise ValueError(
                f"Number of prompts ({len(prompts)}) must match "
                f"number of images ({len(generated_images)})"
            )

        self._load_model()

        all_scores = []
        logger.debug(f"Computing ImageReward for {len(generated_images)} images")

        for img, prompt in zip(generated_images, prompts):
            score = self._score_single(img.convert("RGB"), prompt)
            all_scores.append(score)

        mean_score = float(np.mean(all_scores))
        std_score = float(np.std(all_scores))

        logger.info(f"ImageReward: {mean_score:.4f} ± {std_score:.4f}")

        return MetricResult(
            aggregate={"image_reward_mean": mean_score, "image_reward_std": std_score},
            per_image=all_scores,
        )
