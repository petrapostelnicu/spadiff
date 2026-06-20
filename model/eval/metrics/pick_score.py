import logging
from typing import Any

import numpy as np
import torch
from PIL import Image

from .base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


class PickScoreMetric(BaseMetric):
    """PickScore metric for predicting human preference.

    PickScore is trained on human preference data to predict which
    images humans would prefer. Higher scores indicate images more
    likely to be preferred by humans.
    """

    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 16,
    ):
        """Initialize PickScore metric.

        Args:
            device: Device for computation.
            batch_size: Batch size for processing.
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self._model = None
        self._processor = None

    def _load_model(self) -> None:
        """Load PickScore model."""
        if self._model is not None:
            return

        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError:
            raise ImportError(
                "transformers is required for PickScore. "
                "Install with: pip install transformers"
            )

        logger.info("Loading PickScore model...")
        processor_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_name = "yuvalkirstain/PickScore_v1"

        self._processor = AutoProcessor.from_pretrained(processor_name)
        self._model = AutoModel.from_pretrained(model_name).eval().to(self.device)
        logger.info("PickScore model loaded successfully")

    @property
    def name(self) -> str:
        return "pick_score"

    @property
    def higher_is_better(self) -> bool:
        return True

    @torch.no_grad()
    def _compute_batch(
        self,
        images: list[Image.Image],
        prompts: list[str],
    ) -> list[float]:
        """Compute PickScore for a batch."""
        # Process inputs
        image_inputs = self._processor(
            images=[img.convert("RGB") for img in images],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self.device)

        text_inputs = self._processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self.device)

        # Get embeddings
        image_embs = self._model.get_image_features(**image_inputs)
        text_embs = self._model.get_text_features(**text_inputs)

        # Normalize
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        # Compute scores with logit_scale
        scores = self._model.logit_scale.exp() * (image_embs * text_embs).sum(dim=-1)
        return scores.tolist()

    def compute(
        self,
        generated_images: list[Image.Image],
        reference_images: list[Image.Image] | None = None,
        prompts: list[str] | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """Compute PickScore for generated images.

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
            raise ValueError("PickScore requires prompts")
        if len(prompts) != len(generated_images):
            raise ValueError(
                f"Number of prompts ({len(prompts)}) must match "
                f"number of images ({len(generated_images)})"
            )

        self._load_model()

        all_scores = []
        logger.debug(f"Computing PickScore for {len(generated_images)} images")

        for i in range(0, len(generated_images), self.batch_size):
            batch_images = generated_images[i:i + self.batch_size]
            batch_prompts = prompts[i:i + self.batch_size]
            scores = self._compute_batch(batch_images, batch_prompts)
            all_scores.extend(scores)

        mean_score = float(np.mean(all_scores))
        std_score = float(np.std(all_scores))

        logger.info(f"PickScore: {mean_score:.4f} ± {std_score:.4f}")

        return MetricResult(
            aggregate={"pick_score_mean": mean_score, "pick_score_std": std_score},
            per_image=all_scores,
        )
