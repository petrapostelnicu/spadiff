import logging
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


class MANIQAMetric(BaseMetric):
    """MAN-IQA: Multi-dimension Attention Network for IQA (CVPR 2022).

    No-reference image quality metric that assesses perceptual quality
    without needing a reference image. Detects graininess, blur, noise,
    and compression artifacts. Range: [0, 1], higher is better.
    """

    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 32,
    ):
        """Initialize MAN-IQA metric.

        Args:
            device: Device for computation.
            batch_size: Batch size for processing.
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self._metric = None

    def _load_model(self) -> None:
        if self._metric is not None:
            return
        import pyiqa

        logger.info("Loading MAN-IQA model (pyiqa)...")
        self._metric = pyiqa.create_metric('maniqa', device=self.device)
        logger.info("MAN-IQA model loaded successfully")

    @property
    def name(self) -> str:
        return "maniqa"

    @property
    def higher_is_better(self) -> bool:
        return True

    @torch.no_grad()
    def _compute_batch(self, images: list[Image.Image]) -> list[float]:
        """Compute MAN-IQA scores for a batch of images."""
        # Convert PIL images to tensors in [0, 1]
        to_tensor = transforms.ToTensor()
        tensors = torch.stack([to_tensor(img.convert("RGB")) for img in images])
        tensors = tensors.to(self.device)

        scores = []
        for img_tensor in tensors:
            score = self._metric(img_tensor.unsqueeze(0))
            scores.append(score.item())

        return scores

    def compute(
        self,
        generated_images: list[Image.Image],
        reference_images: list[Image.Image] | None = None,
        prompts: list[str] | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """Compute MAN-IQA scores for generated images.

        Args:
            generated_images: List of generated images.
            reference_images: Not used (no-reference metric).
            prompts: Not used.

        Returns:
            MetricResult with aggregate stats and per-image scores.
        """
        self._load_model()

        all_scores = []
        logger.debug(f"Computing MAN-IQA scores for {len(generated_images)} images")

        for i in range(0, len(generated_images), self.batch_size):
            batch = generated_images[i:i + self.batch_size]
            scores = self._compute_batch(batch)
            all_scores.extend(scores)

        mean_score = float(np.mean(all_scores))
        std_score = float(np.std(all_scores))

        logger.info(f"MAN-IQA: {mean_score:.4f} ± {std_score:.4f}")

        return MetricResult(
            aggregate={"maniqa_mean": mean_score, "maniqa_std": std_score},
            per_image=all_scores,
        )
