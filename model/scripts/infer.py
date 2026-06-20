import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Add package paths for development.
project_root = Path(__file__).resolve().parent.parent
repo_root = project_root.parent
sys.path.insert(0, str(repo_root))

from model.src.models import FluxTransformer2DModel
from model.src.models.adaptive_mask import AdaptiveMaskModule
from model.src.pipelines import FluxRegionalPipeline
from model.utils.visualizer import Visualizer


visualizer = Visualizer()

IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def discover_samples(input_dir: str) -> list[tuple[str, str, str]]:
    """Return [(stem, image_path, json_path), ...] for every pair found.

    A pair = matching stems for one of the accepted image extensions and a
    sibling `.json`. Raises if a JSON is missing for an image.
    """
    samples = []
    for fname in sorted(os.listdir(input_dir)):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in IMAGE_EXTS:
            continue
        img_path = os.path.join(input_dir, fname)
        json_path = os.path.join(input_dir, f"{stem}.json")
        if not os.path.isfile(json_path):
            raise FileNotFoundError(
                f"Sample '{stem}': image '{img_path}' has no matching JSON at '{json_path}'."
            )
        samples.append((stem, img_path, json_path))
    if not samples:
        raise FileNotFoundError(f"No image/JSON pairs found in {input_dir}")
    return samples


def load_seg_map_and_prompt(seg_map_path: str, seg_anno_path: str, cond_scale_factor: int):
    """Single-sample equivalent of dataset/collate output (one element, no batch dim)."""
    seg_map = Image.open(seg_map_path).convert("RGB")
    img_w, img_h = seg_map.size
    seg_map = np.array(seg_map)

    with open(seg_anno_path, "r") as f:
        seg_anno = json.load(f)

    s = cond_scale_factor * 16
    cond_resolution = [img_h // s * 16, img_w // s * 16]

    cond_transforms = transforms.Compose([
        transforms.Resize(cond_resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    global_caption = seg_anno["caption"]

    color_to_text = {tuple(region["color"]): region["text"] for region in seg_anno["segments_info"]}

    label, regional_captions = [], []
    for color in (tuple(c.tolist()) for c in np.unique(seg_map.reshape(-1, 3), axis=0)):
        if color not in color_to_text:
            continue
        mask = ((seg_map[..., 0] == color[0])
                & (seg_map[..., 1] == color[1])
                & (seg_map[..., 2] == color[2]))
        label.append(mask)
        regional_captions.append(color_to_text[color])

    if not label:
        raise ValueError(f"No region colors in {seg_map_path} matched any 'segments_info' entry in {seg_anno_path}.")

    label = torch.from_numpy(np.stack(label, axis=0)).long()  # (n, h, w)

    cond_pixel_values = np.zeros([label.shape[-2], label.shape[-1], 3], dtype=np.uint8)
    cond_pixel_values = visualizer.draw_contours(
        cond_pixel_values,
        label.cpu().numpy(),
        thickness=1,
        colors=[(255, 255, 255)] * len(regional_captions),
    )
    cond_pixel_values = cond_transforms(Image.fromarray(cond_pixel_values))

    return {
        "label": label,
        "regional_captions": regional_captions,
        "global_caption": global_caption,
        "cond_pixel_values": cond_pixel_values,
        "image_width": cond_resolution[1] * cond_scale_factor,
        "image_height": cond_resolution[0] * cond_scale_factor,
    }


def _parse_args():
    p = argparse.ArgumentParser()
    # Model + checkpoints.
    p.add_argument("--pretrained_model_name_or_path", type=str,
                   default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--resume_from_checkpoint", type=str, required=True,
                   help="Directory containing 'cond/' subfolder (+ optional "
                        "adaptive_mask_module.pt, cond2img_scale.pt) OR direct "
                        "path to a cond .safetensors file.")
    # Inputs.
    p.add_argument("--input_dir", type=str, required=True,
                   help="Folder containing <name>.{png,jpg,jpeg} + <name>.json pairs.")
    p.add_argument("--output_path", type=str, default="./result")
    # Seeds.
    p.add_argument("--seed", type=int, default=42,
                   help="Base seed; the actual seeds used are "
                        "[seed, seed+1, ..., seed+num_seeds-1] (default: 42).")
    p.add_argument("--num_seeds", type=int, default=10,
                   help="Number of incrementing seeds to generate per sample "
                        "(default: 10).")
    # Generation knobs (mirror generate_image.py + pipeline_flux defaults).
    p.add_argument("--conditional_integration_method", type=str,
                   choices=["none", "unified", "decoupled"], default="unified")
    p.add_argument("--attention_mask_method", type=str,
                   choices=["none", "base", "hard", "adaptive"], default="adaptive")
    p.add_argument("--zero_init_cond2img", action="store_true",
                   help="Allocate per-channel cond2img_scale params (decoupled mode); "
                        "trained values will be loaded from cond2img_scale.pt if present.")
    p.add_argument("--hard_attn_block_range", type=int, nargs=2, default=[19, 37],
                   metavar=("START", "END"))
    p.add_argument("--is_filter_cond_token", action="store_true", default=True,
                   help="(default on) Drop zero-content cond tokens before attention.")
    p.add_argument("--no_filter_cond_token", dest="is_filter_cond_token",
                   action="store_false")
    p.add_argument("--cond2image_attention_weight", type=float, default=1.0,
                   help="Multiplicative attention weight at img↔cond cells. "
                        "Training default 1.0. Now propagates continuously via "
                        "AdaptiveMaskSpec.soft_bias.")
    p.add_argument("--cond_scale_factor", type=int, default=2)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--num_inference_steps", type=int, default=32)
    p.add_argument("--num_images_per_prompt", type=int, default=1)
    p.add_argument("--max_sequence_length", type=int, default=512)
    p.add_argument("--regional_max_sequence_length", type=int, default=50)
    p.add_argument("--weight_dtype", type=str,
                   choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--adaptive_mask_hidden_dim", type=int, default=256)
    p.add_argument("--adaptive_mask_variant", type=str, default="full")
    return p.parse_args()


def build_pipeline(args):
    """Load FLUX + seg adapter once. Returns the pipeline."""
    weight_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.weight_dtype]

    print("[infer] Loading transformer...")
    flux_transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
        conditional_integration_method=args.conditional_integration_method,
        zero_init_cond2img=False,
    )

    if args.zero_init_cond2img:
        all_blocks = (list(flux_transformer.transformer_blocks)
                      + list(flux_transformer.single_transformer_blocks))
        for block in all_blocks:
            if block.conditional_integration_method == "decoupled":
                block.zero_init_cond2img = True
                block.cond2img_scale = torch.nn.Parameter(
                    torch.zeros(flux_transformer.inner_dim, dtype=weight_dtype)
                )

    use_adaptive_mask = args.attention_mask_method == "adaptive"
    if use_adaptive_mask:
        num_layers = (flux_transformer.config.num_layers
                      + flux_transformer.config.num_single_layers)
        flux_transformer.adaptive_mask_module = AdaptiveMaskModule(
            temb_dim=flux_transformer.inner_dim,
            num_layers=num_layers,
            num_heads=flux_transformer.config.num_attention_heads,
            hidden_dim=args.adaptive_mask_hidden_dim,
            variant=args.adaptive_mask_variant,
        ).to(weight_dtype)

    print("[infer] Loading pipeline...")
    pipeline = FluxRegionalPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=flux_transformer,
        torch_dtype=weight_dtype,
    )

    print("[infer] Loading LoRA + adapter tensors...")
    ckpt = args.resume_from_checkpoint
    if os.path.isfile(ckpt) and ckpt.endswith(".safetensors"):
        pipeline.load_lora_weights(ckpt, adapter_name="cond")
        pipeline.set_adapters(["cond"])
        ckpt_dir = os.path.dirname(ckpt)
    elif os.path.exists(os.path.join(ckpt, "cond")):
        pipeline.load_lora_weights(
            os.path.join(ckpt, "cond"),
            adapter_name="cond",
            weight_name="pytorch_lora_weights.safetensors",
        )
        pipeline.set_adapters(["cond"])
        ckpt_dir = ckpt
    else:
        pipeline.load_lora_weights(ckpt, adapter_name="default")
        pipeline.set_adapters("default")
        ckpt_dir = ckpt

    if args.zero_init_cond2img:
        scale_path = os.path.join(ckpt_dir, "cond2img_scale.pt")
        if os.path.exists(scale_path):
            print(f"[infer] Loading cond2img_scale from {scale_path}...")
            cond2img_state = torch.load(scale_path, map_location="cpu")
            for name, param in flux_transformer.named_parameters():
                if name in cond2img_state:
                    param.data.copy_(cond2img_state[name].to(param.dtype))
        else:
            print(f"[WARNING] zero_init_cond2img=True but no cond2img_scale.pt at {scale_path}")

    if use_adaptive_mask:
        amm_path = os.path.join(ckpt_dir, "adaptive_mask_module.pt")
        if os.path.exists(amm_path):
            print(f"[infer] Loading adaptive_mask_module from {amm_path}...")
            flux_transformer.adaptive_mask_module.load_state_dict(
                torch.load(amm_path, map_location="cpu")
            )
        else:
            print(f"[WARNING] attention_mask_method=adaptive but no adaptive_mask_module.pt "
                  f"at {amm_path}")

    pipeline.set_progress_bar_config(disable=False)
    pipeline = pipeline.to("cuda")
    return pipeline


def generate_for_sample(pipeline, args, stem, seg_map_path, seg_anno_path, seeds):
    """Generate `len(seeds)` images for one sample, one per seed."""
    os.makedirs(args.output_path, exist_ok=True)

    batch = load_seg_map_and_prompt(
        seg_map_path=seg_map_path,
        seg_anno_path=seg_anno_path,
        cond_scale_factor=args.cond_scale_factor,
    )

    print(f"[infer] [{stem}] global: {batch['global_caption']!r}")
    print(f"[infer] [{stem}] regions ({len(batch['regional_captions'])}): "
          f"{[c[:40] for c in batch['regional_captions']]}")

    saved = []
    for seed in seeds:
        generator = [torch.Generator("cuda").manual_seed(seed + i)
                     for i in range(args.num_images_per_prompt)]
        print(f"[infer] [{stem}] seed={seed} -> generating "
              f"{args.num_images_per_prompt} image(s)...")
        images = pipeline(
            global_prompt=batch["global_caption"],
            regional_prompts=batch["regional_captions"],
            regional_labels=batch["label"],
            cond=(batch["cond_pixel_values"] + 1) / 2.0,
            attention_mask_method=args.attention_mask_method,
            conditional_integration_method=args.conditional_integration_method,
            is_filter_cond_token=args.is_filter_cond_token,
            cond2image_attention_weight=args.cond2image_attention_weight,
            hard_attn_block_range=args.hard_attn_block_range,
            height=batch["image_height"],
            width=batch["image_width"],
            cond_scale_factor=args.cond_scale_factor,
            num_images_per_prompt=args.num_images_per_prompt,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            max_sequence_length=args.max_sequence_length,
            regional_max_sequence_length=args.regional_max_sequence_length,
        ).images

        for i, image in enumerate(images):
            suffix = f"_{i}" if args.num_images_per_prompt > 1 else ""
            save_path = os.path.join(args.output_path, f"{stem}_seed{seed}{suffix}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
            print(f"[infer] Wrote {save_path}")
            saved.append(save_path)
    return saved


def main():
    args = _parse_args()
    samples = discover_samples(args.input_dir)
    seeds = [args.seed + i for i in range(args.num_seeds)]
    print(f"[infer] Found {len(samples)} sample(s) in {args.input_dir}")
    print(f"[infer] Using {len(seeds)} seed(s) per sample: {seeds}")

    pipeline = build_pipeline(args)

    for stem, img_path, json_path in samples:
        try:
            generate_for_sample(pipeline, args, stem, img_path, json_path, seeds)
        except Exception as e:
            print(f"[infer] [{stem}] FAILED: {e}")


if __name__ == "__main__":
    main()
