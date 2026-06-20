import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


def crop_region_from_mask(
    image: Image.Image,
    mask: torch.Tensor | np.ndarray,
    padding: int = 10,
    min_size: int = 32,
) -> Image.Image | None:
    """Crop image region based on mask with padding.

    Args:
        image: PIL Image to crop.
        mask: Binary mask tensor (H, W).
        padding: Padding around bounding box.
        min_size: Minimum crop dimension.

    Returns:
        Cropped PIL Image or None if mask is empty or too small.
    """
    mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask

    # Find bounding box of mask
    rows = np.any(mask_np > 0, axis=1)
    cols = np.any(mask_np > 0, axis=0)

    if not rows.any() or not cols.any():
        return None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # Scale coordinates if mask size differs from image size
    img_w, img_h = image.size
    if mask_np.shape[0] != img_h or mask_np.shape[1] != img_w:
        scale_y = img_h / mask_np.shape[0]
        scale_x = img_w / mask_np.shape[1]
        x_min = int(x_min * scale_x)
        x_max = int(x_max * scale_x)
        y_min = int(y_min * scale_y)
        y_max = int(y_max * scale_y)

    # Add padding
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(img_w, x_max + padding)
    y_max = min(img_h, y_max + padding)

    # Ensure minimum crop size
    if x_max - x_min < min_size or y_max - y_min < min_size:
        return None

    return image.crop((x_min, y_min, x_max, y_max))


class RegionalCLIPScoreMetric(BaseMetric):
    """Regional CLIP Score metric for measuring text-image alignment per region.

    For each region in the image, crops the area based on the segmentation mask
    and computes CLIP score between the cropped region and the regional caption.
    """

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "openai/clip-vit-base-patch16",
        max_regions_per_image: int = 50,
    ):
        """Initialize Regional CLIP Score metric.

        Args:
            device: Device for computation.
            model_name: HuggingFace model name for CLIP.
            max_regions_per_image: Maximum regions to evaluate per image.
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._model_name = model_name
        self.max_regions_per_image = max_regions_per_image

        self._clip_score = None
        self._transform = transforms.ToTensor()

        # Accumulators
        self._scores: list[float] = []

    def _load_model(self) -> None:
        """Load CLIP model."""
        if self._clip_score is not None:
            return

        from torchmetrics.multimodal import CLIPScore

        logger.info(f"Loading CLIP model ({self._model_name}) for regional CLIP score...")
        self._clip_score = CLIPScore(model_name_or_path=self._model_name)
        self._clip_score = self._clip_score.to(self.device)
        logger.info("CLIP model loaded successfully")

    @property
    def name(self) -> str:
        return "regional_clip_score"

    @property
    def higher_is_better(self) -> bool:
        return True

    def reset(self) -> None:
        """Reset accumulated scores."""
        self._scores = []

    @torch.no_grad()
    def _compute_clip_score(self, image: Image.Image, caption: str) -> float:
        """Compute CLIP score for a single image-caption pair."""
        # Convert image to tensor
        image_tensor = self._transform(image.convert("RGB"))
        # Scale to [0, 255] as expected by torchmetrics CLIPScore
        image_tensor = (image_tensor * 255).to(torch.uint8)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        self._clip_score.reset()
        score = self._clip_score(image_tensor, [caption])
        # Divide by 100 to get cosine similarity in [0, 1] range
        return score.item() / 100.0

    def update(
        self,
        generated_image: Image.Image,
        regional_labels: torch.Tensor,
        regional_captions: list[str],
    ) -> None:
        """Update metrics with a single image.

        Args:
            generated_image: Generated image to evaluate.
            regional_labels: Segmentation masks (N, H, W).
            regional_captions: List of captions for each region.
        """
        self._load_model()

        num_regions = min(
            len(regional_captions),
            regional_labels.shape[0],
            self.max_regions_per_image
        )

        for region_idx in range(num_regions):
            caption = regional_captions[region_idx]
            mask = regional_labels[region_idx]

            # Crop region
            cropped = crop_region_from_mask(generated_image, mask)
            if cropped is None:
                continue

            # Compute CLIP score
            score = self._compute_clip_score(cropped, caption)
            self._scores.append(score)

    def aggregate(self) -> MetricResult:
        """Compute aggregated scores from accumulated statistics."""
        if not self._scores:
            return MetricResult(
                aggregate={
                    "regional_clip_score_mean": 0.0,
                    "regional_clip_score_std": 0.0,
                    "regional_clip_score_total": 0,
                }
            )

        mean_score = float(np.mean(self._scores))
        std_score = float(np.std(self._scores))

        return MetricResult(
            aggregate={
                "regional_clip_score_mean": mean_score,
                "regional_clip_score_std": std_score,
                "regional_clip_score_total": len(self._scores),
            },
            per_image=self._scores,
        )

    def compute(
        self,
        generated_images: list[Image.Image],
        reference_images: list[Image.Image] | None = None,
        prompts: list[str] | None = None,
        regional_labels_list: list[torch.Tensor] | None = None,
        regional_captions_list: list[list[str]] | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """Compute regional CLIP scores for a batch of images.

        Args:
            generated_images: List of generated images.
            reference_images: Not used.
            prompts: Not used (use regional_captions_list instead).
            regional_labels_list: List of segmentation masks per image.
            regional_captions_list: List of regional captions per image.

        Returns:
            MetricResult with regional CLIP score statistics.
        """
        if regional_labels_list is None or regional_captions_list is None:
            raise ValueError(
                "regional_clip_score metric requires regional_labels_list and regional_captions_list"
            )

        if len(generated_images) != len(regional_labels_list):
            raise ValueError(
                f"Number of images ({len(generated_images)}) must match "
                f"number of label sets ({len(regional_labels_list)})"
            )

        self.reset()

        for img, labels, captions in zip(
            generated_images, regional_labels_list, regional_captions_list
        ):
            self.update(img, labels, captions)

        return self.aggregate()


@dataclass
class RegionQualityResult:
    """Aggregated results for region quality metrics."""
    spatial_score: float
    color_score: float
    shape_score: float
    texture_score: float

    # Detailed counts for debugging
    spatial_yes: int = 0
    spatial_total: int = 0
    color_yes: int = 0
    color_total: int = 0
    shape_yes: int = 0
    shape_total: int = 0
    texture_yes: int = 0
    texture_total: int = 0


class RegionQualityMetric(BaseMetric):
    """Region-wise quality metric using VLM (Qwen2-VL) for VQA-based evaluation.

    Evaluates generated images from two perspectives:

    1. Spatial Score: For each segmentation mask, crop the corresponding region
       and ask the VLM whether the target entity is located within this area.
       Score = ratio of "Yes" answers to total number of entities.

    2. Attribute Scores (Color, Shape, Texture): For each region, crop the area
       and ask the VLM whether the entity satisfies the described attributes.
       Each attribute type is evaluated separately.
    """

    VALID_METRICS = {"spatial", "color", "shape", "texture"}

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "cuda",
        max_regions_per_image: int = 50,
        metrics: list[str] | None = None,
    ):
        """Initialize Region Quality metric.

        Args:
            model_name: Qwen2-VL model name from HuggingFace.
            device: Device for computation.
            max_regions_per_image: Maximum regions to evaluate per image.
            metrics: List of metrics to compute. Options: "spatial", "color", "shape", "texture".
                     If None, computes all metrics.
        """
        self.model_name = model_name
        self.device = device
        self.max_regions_per_image = max_regions_per_image

        # Validate and store requested metrics
        if metrics is None:
            self.metrics = self.VALID_METRICS
        else:
            invalid = set(metrics) - self.VALID_METRICS
            if invalid:
                raise ValueError(f"Invalid metrics: {invalid}. Valid options: {self.VALID_METRICS}")
            self.metrics = set(metrics)

        self._model = None
        self._processor = None

        # Accumulators for streaming updates
        self._spatial_yes = 0
        self._spatial_total = 0
        self._color_yes = 0
        self._color_total = 0
        self._shape_yes = 0
        self._shape_total = 0
        self._texture_yes = 0
        self._texture_total = 0

    def _load_model(self) -> None:
        """Load Qwen2-VL model."""
        if self._model is not None:
            return

        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        logger.info(f"Loading {self.model_name}...")

        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"Available VRAM: {total_vram:.1f} GB")

        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        self._processor = AutoProcessor.from_pretrained(self.model_name)

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            logger.info(f"Model loaded! Using {allocated:.1f} GB VRAM")

    @property
    def name(self) -> str:
        return "region_quality"

    @property
    def higher_is_better(self) -> bool:
        return True

    def reset(self) -> None:
        """Reset accumulated statistics."""
        self._spatial_yes = 0
        self._spatial_total = 0
        self._color_yes = 0
        self._color_total = 0
        self._shape_yes = 0
        self._shape_total = 0
        self._texture_yes = 0
        self._texture_total = 0

    def _ask_vlm(self, image: Image.Image, question: str, max_tokens: int = 16) -> str:
        """Ask VLM a yes/no question about an image.

        Args:
            image: PIL Image to analyze.
            question: Question to ask.
            max_tokens: Maximum tokens for response.

        Returns:
            VLM response string.
        """
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return output.strip().lower()

    def _is_yes_response(self, response: str) -> bool:
        """Check if VLM response contains 'yes' (matches reference's 'Yes' in res or 'yes' in res)."""
        return "yes" in response.lower()

    def update(
        self,
        generated_image: Image.Image,
        regional_labels: torch.Tensor,
        regional_captions: list[str],
        short_regional_captions: list[str] | None = None,
    ) -> None:
        """Update metrics with a single image.

        Mirrors CreatiLayout's `score_layoutsam_benchmark.py` scoring scheme:
        - Spatial uses the short region caption ("description").
        - Attribute questions reference both the short description and the
          detailed caption; the VLM is told to answer "Yes" when the attribute
          is not mentioned, so the denominator is the total region count.
        - Attribute scores are 0 when the entity is not detected (spatial=No).

        Args:
            generated_image: Generated image to evaluate.
            regional_labels: Segmentation masks (N, H, W).
            regional_captions: Detailed caption per region.
            short_regional_captions: Short caption per region. If None, uses
                ``regional_captions`` for both roles (e.g. class-name datasets).
        """
        self._load_model()

        if short_regional_captions is None:
            short_regional_captions = regional_captions

        num_regions = min(
            len(regional_captions),
            len(short_regional_captions),
            regional_labels.shape[0],
            self.max_regions_per_image,
        )

        for region_idx in range(num_regions):
            description = short_regional_captions[region_idx]
            detail_description = regional_captions[region_idx]
            mask = regional_labels[region_idx]

            cropped = crop_region_from_mask(generated_image, mask)
            if cropped is None:
                continue

            spatial_yes = False
            if self.metrics & {"spatial", "color", "shape", "texture"}:
                spatial_question = (
                    f'Is the subject "{description}" present in the image? '
                    f'Strictly answer with "Yes" or "No", without any irrelevant words.'
                )
                spatial_yes = self._is_yes_response(self._ask_vlm(cropped, spatial_question))

            if "spatial" in self.metrics:
                if spatial_yes:
                    self._spatial_yes += 1
                self._spatial_total += 1

            # Attributes: only credited when the entity is present. Denominator
            # is the total region count (matches the reference). The VLM is
            # instructed to answer "Yes" when the attribute is unspecified.

            if "color" in self.metrics:
                if spatial_yes:
                    q = (
                        f'Is the subject in "{description}" in the image consistent with the color '
                        f'described in the detailed description: "{detail_description}"? '
                        f'Strictly answer with "Yes" or "No", without any irrelevant words. '
                        f'If the color is not mentioned in the detailed description, the answer is "Yes".'
                    )
                    if self._is_yes_response(self._ask_vlm(cropped, q)):
                        self._color_yes += 1
                self._color_total += 1

            if "texture" in self.metrics:
                if spatial_yes:
                    q = (
                        f'Is the subject in "{description}" in the image consistent with the texture '
                        f'described in the detailed description: "{detail_description}"? '
                        f'Strictly answer with "Yes" or "No", without any irrelevant words. '
                        f'If the texture is not mentioned in the detailed description, the answer is "Yes".'
                    )
                    if self._is_yes_response(self._ask_vlm(cropped, q)):
                        self._texture_yes += 1
                self._texture_total += 1

            if "shape" in self.metrics:
                if spatial_yes:
                    q = (
                        f'Is the subject in "{description}" in the image consistent with the shape '
                        f'described in the detailed description: "{detail_description}"? '
                        f'Strictly answer with "Yes" or "No", without any irrelevant words. '
                        f'If the shape is not mentioned in the detailed description, the answer is "Yes".'
                    )
                    if self._is_yes_response(self._ask_vlm(cropped, q)):
                        self._shape_yes += 1
                self._shape_total += 1

    def aggregate(self) -> MetricResult:
        """Compute aggregated scores from accumulated statistics."""
        results = {}

        if "spatial" in self.metrics:
            spatial_score = self._spatial_yes / max(self._spatial_total, 1)
            results["region_spatial_score"] = spatial_score
            results["region_spatial_total"] = self._spatial_total

        if "color" in self.metrics:
            color_score = self._color_yes / max(self._color_total, 1)
            results["region_color_score"] = color_score
            results["region_color_total"] = self._color_total

        if "shape" in self.metrics:
            shape_score = self._shape_yes / max(self._shape_total, 1)
            results["region_shape_score"] = shape_score
            results["region_shape_total"] = self._shape_total

        if "texture" in self.metrics:
            texture_score = self._texture_yes / max(self._texture_total, 1)
            results["region_texture_score"] = texture_score
            results["region_texture_total"] = self._texture_total

        return MetricResult(aggregate=results)

    def compute(
        self,
        generated_images: list[Image.Image],
        reference_images: list[Image.Image] | None = None,
        prompts: list[str] | None = None,
        regional_labels_list: list[torch.Tensor] | None = None,
        regional_captions_list: list[list[str]] | None = None,
        short_regional_captions_list: list[list[str]] | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """Compute region quality metrics for a batch of images.

        Args:
            generated_images: List of generated images.
            reference_images: Not used.
            prompts: Not used (use regional_captions_list instead).
            regional_labels_list: List of segmentation masks per image.
            regional_captions_list: Detailed regional caption per image.
            short_regional_captions_list: Short regional caption per image.
                If None, the detailed captions are reused.

        Returns:
            MetricResult with spatial, color, shape, and texture scores.
        """
        if regional_labels_list is None or regional_captions_list is None:
            raise ValueError(
                "region_quality metric requires regional_labels_list and regional_captions_list"
            )

        if len(generated_images) != len(regional_labels_list):
            raise ValueError(
                f"Number of images ({len(generated_images)}) must match "
                f"number of label sets ({len(regional_labels_list)})"
            )

        self.reset()

        if short_regional_captions_list is None:
            short_regional_captions_list = regional_captions_list

        for img, labels, captions, short_captions in zip(
            generated_images,
            regional_labels_list,
            regional_captions_list,
            short_regional_captions_list,
        ):
            self.update(img, labels, captions, short_captions)

        return self.aggregate()
