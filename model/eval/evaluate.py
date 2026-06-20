import sys
import os
from pathlib import Path

# Add repo root to sys.path so that 'model.*' imports work
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import argparse
import json
import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from omegaconf import OmegaConf

from model.dataset.collate_fn import collate_fn
from model.utils.utils import instantiate_from_config


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    return parser


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gen_image_dir = Path(args.project.gen_image_dir)
    output_dir = Path(args.project.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_groups = args.eval.num_images_per_prompt
    resolution = args.resolution

    # Load validation dataset
    val_dataset = instantiate_from_config(args.data.val)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
    )

    # --- Pass 1: Collect generated images, reference images, and prompts ---
    print("###### Loading images ######")
    generated_images = {i: [] for i in range(num_groups)}
    reference_images = []
    prompts = []
    skipped = 0

    for batch in tqdm(val_dataloader, desc="Loading"):
        image_name = batch["image_name"][0]

        # Check that generated images exist (at least group_0)
        gen_path_0 = gen_image_dir / "group_0" / image_name
        if not gen_path_0.exists():
            skipped += 1
            continue

        for group_idx in range(num_groups):
            gen_path = gen_image_dir / f"group_{group_idx}" / image_name
            generated_images[group_idx].append(Image.open(gen_path).convert("RGB"))

        reference_images.append(Image.open(batch["image_path"][0]).convert("RGB"))

        caption = batch["global_caption"]
        if isinstance(caption, list):
            caption = caption[0]
        prompts.append(caption)

    if skipped > 0:
        print(f"  Skipped {skipped} images (generated files not found)")
    print(f"  Loaded {len(reference_images)} image pairs")

    has_prompts = len(prompts) > 0 and prompts[0] is not None
    gen_imgs = generated_images[0]
    results = {}

    # --- FID (per group and averaged) using clean_fid ---
    print("###### Computing FID ######")
    from cleanfid import fid as clean_fid

    # Get reference image directory from dataset
    ref_image_dir = getattr(args.data.val.params, 'image_root', None)
    if ref_image_dir is None:
        print("  WARNING: No image_root in val dataset, skipping FID")
        results["fid"] = None
    else:
        fid_scores = []
        for group_idx in range(num_groups):
            group_dir = gen_image_dir / f"group_{group_idx}"
            if not group_dir.exists():
                continue
            fid_score = clean_fid.compute_fid(
                str(ref_image_dir),
                str(group_dir),
                dataset_res=resolution,
                batch_size=32,
            )
            fid_scores.append(fid_score)
            results[f"fid_group_{group_idx}"] = round(fid_score, 4)
            print(f"  FID group_{group_idx}: {fid_score:.4f}")

        results["fid_mean"] = round(sum(fid_scores) / len(fid_scores), 4)
        results["fid_std"] = round(float(np.std(fid_scores)), 4)
        print(f"  FID mean: {results['fid_mean']} ± {results['fid_std']}")

    # --- CLIP Score ---
    if has_prompts:
        print("###### Computing CLIP Score ######")
        from model.eval.metrics import CLIPScoreMetric

        clip_metric = CLIPScoreMetric(device=device, batch_size=32)
        clip_result = clip_metric.compute(generated_images=gen_imgs, prompts=prompts)
        results.update({k: round(v, 4) for k, v in clip_result.aggregate.items()})
        print(f"  {clip_result.aggregate}")
        del clip_metric
        torch.cuda.empty_cache()

    # --- PickScore ---
    if has_prompts:
        print("###### Computing PickScore ######")
        from model.eval.metrics import PickScoreMetric

        pick_metric = PickScoreMetric(device=device)
        pick_result = pick_metric.compute(generated_images=gen_imgs, prompts=prompts)
        results.update({k: round(v, 4) for k, v in pick_result.aggregate.items()})
        print(f"  {pick_result.aggregate}")
        del pick_metric
        torch.cuda.empty_cache()

    # --- ImageReward ---
    if has_prompts:
        print("###### Computing ImageReward ######")
        from model.eval.metrics import ImageRewardMetric

        ir_metric = ImageRewardMetric(device=device)
        ir_result = ir_metric.compute(generated_images=gen_imgs, prompts=prompts)
        results.update({k: round(v, 4) for k, v in ir_result.aggregate.items()})
        print(f"  {ir_result.aggregate}")
        del ir_metric
        torch.cuda.empty_cache()

    # --- MAN-IQA (no-reference image quality) ---
    use_maniqa = getattr(args.eval, "maniqa", True)
    if use_maniqa:
        print("###### Computing MAN-IQA ######")
        from model.eval.metrics import MANIQAMetric

        maniqa_metric = MANIQAMetric(device=device, batch_size=32)
        maniqa_result = maniqa_metric.compute(generated_images=gen_imgs)
        results.update({k: round(v, 4) for k, v in maniqa_result.aggregate.items()})
        print(f"  {maniqa_result.aggregate}")
        del maniqa_metric
        torch.cuda.empty_cache()

    # Free image lists before segmentation consistency pass
    del generated_images, reference_images, gen_imgs
    torch.cuda.empty_cache()

    # --- Segmentation Consistency / mIoU  ---
    sam2_checkpoint = getattr(args.eval, "sam2_checkpoint", None)
    use_segmentation_consistency = getattr(args.eval, "segmentation_consistency", True)
    if sam2_checkpoint is not None and use_segmentation_consistency:
        print("###### Computing Segmentation Consistency (mIoU) ######")
        from model.eval.metrics import SegmentationConsistencyMetric

        sam2_config = getattr(args.eval, "sam2_config", "configs/sam2.1/sam2.1_hiera_l.yaml")
        seg_metric = SegmentationConsistencyMetric(
            sam2_checkpoint=sam2_checkpoint,
            sam2_config=sam2_config,
            device=device,
            resolution=resolution,
        )

        for batch in tqdm(val_dataloader, desc="mIoU"):
            image_name = batch["image_name"][0]
            gen_path = gen_image_dir / "group_0" / image_name
            if not gen_path.exists():
                continue
            gen_img = Image.open(gen_path).convert("RGB")
            seg_metric.update(gen_img, batch["label"][0], batch["boxes"][0])

        seg_result = seg_metric.aggregate()
        results.update({k: round(v, 4) for k, v in seg_result.aggregate.items()})
        print(f"  {seg_result.aggregate}")
        del seg_metric
        torch.cuda.empty_cache()
    elif not use_segmentation_consistency:
        print("###### Skipping mIoU (eval.segmentation_consistency=False) ######")
    else:
        print("###### Skipping mIoU (no eval.sam2_checkpoint specified) ######")

    # --- Regional Quality Metrics (CLIP and/or VLM-based) ---
    # Options: "clip", "spatial", "color", "shape", "texture"
    regional_metrics = list(getattr(args.eval, "regional_metrics", []))
    if regional_metrics:
        # Separate CLIP from VLM metrics
        use_regional_clip = "clip" in regional_metrics
        vlm_metrics = [m for m in regional_metrics if m in {"spatial", "color", "shape", "texture"}]

        # Regional CLIP Score
        if use_regional_clip:
            print("###### Computing Regional CLIP Score ######")
            from model.eval.metrics import RegionalCLIPScoreMetric

            regional_clip_metric = RegionalCLIPScoreMetric(device=device)

            for batch in tqdm(val_dataloader, desc="Regional CLIP"):
                image_name = batch["image_name"][0]
                gen_path = gen_image_dir / "group_0" / image_name
                if not gen_path.exists():
                    continue

                gen_img = Image.open(gen_path).convert("RGB")
                regional_labels = batch["label"][0]  # (N, H, W)
                regional_captions = batch["regional_captions"][0]  # List[str]

                if regional_labels is not None and regional_captions is not None:
                    regional_clip_metric.update(gen_img, regional_labels, regional_captions)

            regional_clip_result = regional_clip_metric.aggregate()
            results["regional_clip_score_mean"] = round(
                regional_clip_result.aggregate["regional_clip_score_mean"], 4
            )
            results["regional_clip_score_std"] = round(
                regional_clip_result.aggregate["regional_clip_score_std"], 4
            )
            print(f"  Regional CLIP Score: {results['regional_clip_score_mean']} ± {results['regional_clip_score_std']}")
            del regional_clip_metric
            torch.cuda.empty_cache()

        # VLM-based Region Quality (spatial, color, shape, texture)
        if vlm_metrics:
            print(f"###### Computing Region Quality ({', '.join(vlm_metrics)}) ######")
            from model.eval.metrics import RegionQualityMetric

            region_model = getattr(args.eval, "regional_metrics_vlm_model", "Qwen/Qwen2-VL-7B-Instruct")
            region_metric = RegionQualityMetric(
                model_name=region_model,
                device=device,
                metrics=vlm_metrics,
            )

            for batch in tqdm(val_dataloader, desc="Region Quality"):
                image_name = batch["image_name"][0]
                gen_path = gen_image_dir / "group_0" / image_name
                if not gen_path.exists():
                    continue

                gen_img = Image.open(gen_path).convert("RGB")
                regional_labels = batch["label"][0]  # (N, H, W)
                regional_captions = batch["regional_captions"][0]  # List[str]
                short_regional_captions = batch.get("short_regional_captions", [None])[0]

                if regional_labels is not None and regional_captions is not None:
                    region_metric.update(
                        gen_img, regional_labels, regional_captions, short_regional_captions
                    )

            region_result = region_metric.aggregate()
            # Only include score metrics (not counts) for requested metrics
            for metric_name in vlm_metrics:
                score_key = f"region_{metric_name}_score"
                if score_key in region_result.aggregate:
                    results[score_key] = round(region_result.aggregate[score_key], 4)
                    print(f"  {metric_name.capitalize()} Score: {results[score_key]}")
            del region_metric
            torch.cuda.empty_cache()
    else:
        print("###### Skipping Regional Metrics (eval.regional_metrics is empty) ######")

    # --- Semantic mIoU (mmsegmentation-based, for COCO-Stuff) ---
    # This matches seg2any's evaluation approach using pretrained segmentation models
    use_semantic_miou = getattr(args.eval, "semantic_miou", False)
    semantic_miou_dataset = getattr(args.eval, "semantic_miou_dataset", "coco_stuff")
    seg_map_dir = getattr(args.eval, "seg_map_dir", None)

    if use_semantic_miou:
        if seg_map_dir is None:
            print("###### Skipping Semantic mIoU (no eval.seg_map_dir specified) ######")
        else:
            print(f"###### Computing Semantic mIoU ({semantic_miou_dataset}) ######")
            from model.eval.metrics.semantic_miou import compute_semantic_miou

            seg_result = compute_semantic_miou(
                gen_image_dir=str(gen_image_dir),
                seg_map_dir=seg_map_dir,
                dataset=semantic_miou_dataset,
                num_groups=num_groups,
                work_dir=str(output_dir),
            )
            results.update({k: round(v, 4) for k, v in seg_result.aggregate.items()})
            print(f"  {seg_result.aggregate}")
            torch.cuda.empty_cache()

    # --- Save results ---
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n###### Evaluation Results ######")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print(f"\nSaved to: {results_path}")


if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    unknown = [s.lstrip('-') for s in unknown]
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    print('###### cli input evaluation setup: ######\n', cli)
    config = OmegaConf.merge(*configs, cli)
    main(config)
