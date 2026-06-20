import logging
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image

from .base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


class SegmentationConsistencyMetric(BaseMetric):
    """Segmentation consistency metric using SAM2.

    Uses SAM2 to re-segment generated images with ground truth bounding box
    and mask prompts, then computes per-region IoU against ground truth masks.
    The aggregate mIoU is the mean across all regions across all images.
    """

    def __init__(
        self,
        sam2_checkpoint: str,
        sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        device: str = "cuda",
        resolution: int = 512,
    ):
        self.device = device
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config
        self.resolution = resolution
        self._predictor = None

        self._total_iou = 0.0
        self._total_regions = 0
        self._per_image_ious: list[float] = []

    @property
    def name(self) -> str:
        return "segmentation_consistency"

    @property
    def higher_is_better(self) -> bool:
        return True

    def _load_model(self) -> None:
        if self._predictor is not None:
            return

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        logger.info("Loading SAM2 model...")
        model = build_sam2(self.sam2_config, self.sam2_checkpoint)
        self._predictor = SAM2ImagePredictor(model)
        logger.info("SAM2 model loaded successfully")

    def reset(self) -> None:
        self._total_iou = 0.0
        self._total_regions = 0
        self._per_image_ious = []

    def update(
        self,
        generated_image: Image.Image,
        label: torch.Tensor,
        boxes: list[np.ndarray],
    ) -> None:
        """Process a single image and accumulate IoU statistics.

        Args:
            generated_image: Generated PIL image.
            label: Ground truth masks tensor (num_regions, H, W).
            boxes: List of normalized [x0, y0, x1, y1] box arrays, one per region.
        """
        self._load_model()

        img_resized = generated_image.resize(
            (self.resolution, self.resolution), resample=Image.BICUBIC
        )
        img_np = np.array(img_resized)

        gt_masks = label.cpu().numpy()  # (num_regions, H, W)
        num_regions = gt_masks.shape[0]

        if num_regions == 0:
            return

        # Resize masks to 256x256 for SAM2 mask prompt input
        label_tensor = label[None, ...].float()  # (1, num_regions, H, W)
        resized = F.interpolate(label_tensor, size=[256, 256], mode='nearest-exact')
        resized_masks = resized[0, ...].long().cpu().numpy()  # (num_regions, 256, 256)

        sam_masks = []
        sam_boxes = []
        for i in range(num_regions):
            sam_masks.append(resized_masks[i:i + 1])  # (1, 256, 256)
            sam_boxes.append(boxes[i] * self.resolution)

        # Run SAM2 prediction
        with torch.inference_mode():
            self._predictor.set_image_batch([img_np] * num_regions)
            masks_batch, _, _ = self._predictor.predict_batch(
                point_coords_batch=None,
                point_labels_batch=None,
                box_batch=sam_boxes,
                mask_input_batch=sam_masks,
                multimask_output=False,
            )

        pred_masks = np.stack(
            [m[0].astype(np.bool_) for m in masks_batch], axis=0
        )  # (num_regions, H, W)

        # Compute per-region IoU
        target = torch.from_numpy(gt_masks).long()
        preds = torch.from_numpy(pred_masks).long()

        intersection = torch.sum(preds & target, dim=[1, 2])
        target_sum = torch.sum(target, dim=[1, 2])
        pred_sum = torch.sum(preds, dim=[1, 2])
        union = target_sum + pred_sum - intersection
        iou = torch.where(union != 0, intersection.float() / union.float(), torch.tensor(0.0))

        self._total_iou += float(iou.sum())
        self._total_regions += num_regions
        self._per_image_ious.append(float(iou.mean()))

    def aggregate(self) -> MetricResult:
        """Return final MetricResult from accumulated statistics."""
        if self._total_regions == 0:
            return MetricResult(
                aggregate={"miou_mean": 0.0, "miou_std": 0.0},
                per_image=None,
            )

        mean_iou = self._total_iou / self._total_regions
        std_iou = float(np.std(self._per_image_ious)) if self._per_image_ious else 0.0

        logger.info(f"Segmentation Consistency (mIoU): {mean_iou:.4f} +/- {std_iou:.4f}")

        return MetricResult(
            aggregate={"miou_mean": float(mean_iou), "miou_std": std_iou},
            per_image=self._per_image_ious,
        )

    def compute(
        self,
        generated_images: list[Image.Image],
        reference_images: list[Image.Image] | None = None,
        prompts: list[str] | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """Compute segmentation consistency

        Args:
            generated_images: List of generated PIL images.
            reference_images: Not used.
            prompts: Not used.
            labels: List of tensors, each (num_regions, H, W) with binary masks.
            boxes: List of box arrays, each a list of normalized [x0, y0, x1, y1].

        Returns:
            MetricResult with aggregate mIoU mean/std and per-image IoU values.
        """
        labels = kwargs.get("labels")
        boxes = kwargs.get("boxes")

        if labels is None or boxes is None:
            raise ValueError("segmentation_consistency requires 'labels' and 'boxes' kwargs")
        if len(labels) != len(generated_images):
            raise ValueError(
                f"Number of labels ({len(labels)}) must match "
                f"number of images ({len(generated_images)})"
            )

        self.reset()
        for img, label, box_list in zip(generated_images, labels, boxes):
            self.update(img, label, box_list)
        return self.aggregate()
