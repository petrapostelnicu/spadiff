import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .base import MetricResult

logger = logging.getLogger(__name__)


# Predefined configs and checkpoints matching seg2any
PRESETS = {
    "coco_stuff": {
        "config": "deeplabv3_r101-d8_4xb4-320k_coco-stuff164k-512x512",
        "checkpoint": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_512x512_4x4_320k_coco-stuff164k/deeplabv3_r101-d8_512x512_4x4_320k_coco-stuff164k_20210709_155402-3cbca14d.pth",
    },
}


def compute_semantic_miou(
    gen_image_dir: str,
    seg_map_dir: str,
    dataset: str = "coco_stuff",
    num_groups: int = 1,
    work_dir: str = None,
    mmseg_config_dir: str = None,
) -> MetricResult:
    """Compute semantic mIoU — identical to `mim test mmseg`.

    Builds an mmengine Runner from the mmseg config and runs the test loop.
    The config handles image loading, preprocessing, inference, and IoU
    computation.

    Args:
        gen_image_dir: Directory containing group_0/, group_1/, etc.
        seg_map_dir: Directory containing ground truth segmentation PNGs.
        dataset: Preset name ("coco_stuff").
        num_groups: Number of image groups to evaluate.
        work_dir: Directory for mmseg logs/output. Defaults to a temp dir.
        mmseg_config_dir: Directory containing mmseg config files.
                          If None, uses model/eval/config.
    """
    from mmengine.config import Config
    from mmengine.runner import Runner

    if dataset not in PRESETS:
        raise ValueError(f"Unknown dataset: {dataset}. Use one of {list(PRESETS.keys())}")

    preset = PRESETS[dataset]

    # Find config file
    if mmseg_config_dir:
        config_path = os.path.join(mmseg_config_dir, f"{preset['config']}.py")
    else:
        repo_root = Path(__file__).parent.parent.parent.parent
        config_path = str(
            repo_root / "model" / "eval" / "config" / f"{preset['config']}.py"
        )

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = Config.fromfile(config_path)

    # Override data paths for each group (same as --cfg-options in mim test)
    for i in range(num_groups):
        group_dir = os.path.join(gen_image_dir, f"group_{i}")
        cfg.test_dataloader.dataset.datasets[i].data_prefix.img_path = group_dir
        cfg.test_dataloader.dataset.datasets[i].data_prefix.seg_map_path = seg_map_dir

    # Only keep the groups we actually have
    cfg.test_dataloader.dataset.datasets = cfg.test_dataloader.dataset.datasets[:num_groups]

    cfg.load_from = preset['checkpoint']

    if work_dir:
        cfg.work_dir = work_dir
    else:
        import tempfile
        cfg.work_dir = tempfile.mkdtemp(prefix='semantic_miou_')

    runner = Runner.from_cfg(cfg)
    metrics = runner.test()

    mean_iou = float(metrics['mIoU'])
    logger.info(f"Semantic mIoU: {mean_iou:.4f}")

    return MetricResult(
        aggregate={"semantic_miou_mean": mean_iou},
        per_image=None,
    )
