import logging
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchmetrics.multimodal import CLIPScore
from torchvision import transforms

from .base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


class CLIPScoreMetric(BaseMetric):
    """CLIP Score metric for measuring text-image alignment.

    Uses torchmetrics CLIPScore to compute cosine similarity between
    CLIP embeddings of generated images and their corresponding text prompts.

    Higher CLIP score indicates better prompt adherence.
    """

    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 32,
        model_name: str = "openai/clip-vit-base-patch16",
    ):
        """Initialize CLIP Score metric.

        Args:
            device: Device for computation.
            batch_size: Batch size for processing.
            model_name: HuggingFace model name for CLIP.
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self._model_name = model_name
        self._clip_score = None
        self._transform = transforms.ToTensor()

    def _load_model(self) -> None:
        """Load CLIP model."""
        if self._clip_score is not None:
            return

        logger.info(f"Loading CLIP model ({self._model_name}) for CLIP score...")
        self._clip_score = CLIPScore(model_name_or_path=self._model_name)
        self._clip_score = self._clip_score.to(self.device)
        logger.info("CLIP model loaded successfully")

    @property
    def name(self) -> str:
        return "clip_score"

    @property
    def higher_is_better(self) -> bool:
        return True

    @torch.no_grad()
    def _compute_batch(
        self,
        images: list[Image.Image],
        prompts: list[str],
    ) -> list[float]:
        """Compute CLIP scores for a batch."""
        # Convert images to tensors (C, H, W) with values in [0, 255]
        image_tensors = torch.stack([
            self._transform(img.convert("RGB")) for img in images
        ])
        # Scale to [0, 255] as expected by torchmetrics CLIPScore
        image_tensors = (image_tensors * 255).to(torch.uint8)
        image_tensors = image_tensors.to(self.device)

        # Compute score for each image-text pair individually
        scores = []
        for img, prompt in zip(image_tensors, prompts):
            self._clip_score.reset()
            score = self._clip_score(img.unsqueeze(0), [prompt])
            # Divide by 100 to get cosine similarity in [0, 1] range
            scores.append(score.item() / 100.0)

        return scores

    def compute(
        self,
        generated_images: list[Image.Image],
        reference_images: list[Image.Image] | None = None,
        prompts: list[str] | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """Compute CLIP scores for generated images.

        Args:
            generated_images: List of generated images.
            reference_images: Not used.
            prompts: List of text prompts (required).

        Returns:
            MetricResult with aggregate stats and per-image scores.

        Raises:
            ValueError: If prompts is None or length doesn't match images.
        """
        if prompts is None:
            raise ValueError("CLIP score requires prompts")
        if len(prompts) != len(generated_images):
            raise ValueError(
                f"Number of prompts ({len(prompts)}) must match "
                f"number of images ({len(generated_images)})"
            )

        self._load_model()

        all_scores = []
        logger.debug(f"Computing CLIP scores for {len(generated_images)} images")

        for i in range(0, len(generated_images), self.batch_size):
            batch_images = generated_images[i:i + self.batch_size]
            batch_prompts = prompts[i:i + self.batch_size]
            scores = self._compute_batch(batch_images, batch_prompts)
            all_scores.extend(scores)

        mean_score = float(np.mean(all_scores))
        std_score = float(np.std(all_scores))

        logger.info(f"CLIP Score: {mean_score:.4f} ± {std_score:.4f}")

        return MetricResult(
            aggregate={"clip_score_mean": mean_score, "clip_score_std": std_score},
            per_image=all_scores,
        )
