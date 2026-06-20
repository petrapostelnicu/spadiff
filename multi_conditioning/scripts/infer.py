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

_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from multi_conditioning.src.models import FluxTransformer2DModel
from multi_conditioning.src.models.adaptive_mask import AdaptiveMaskModule
from multi_conditioning.src.pipelines import FluxRegionalPipeline
from multi_conditioning.utils.visualizer import Visualizer


visualizer = Visualizer()

IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def discover_samples(input_dir: str) -> list[tuple[str, str, str]]:
    """Return [(stem, image_path, json_path), ...] for every pair found.

    A 'sample' is any `<stem>.{png,jpg,jpeg}` in `input_dir` with a matching
    `<stem>.json`. Omini control images live in the same folder under the
    naming convention `<stem>_<omini_name>.<ext>` (see `find_omini_image`),
    so they are filtered out here by skipping stems that contain `_`-suffixed
    omini names.
    """
    samples = []
    for fname in sorted(os.listdir(input_dir)):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in IMAGE_EXTS:
            continue
        json_path = os.path.join(input_dir, f"{stem}.json")
        if not os.path.isfile(json_path):
            # No sibling JSON → this is an auxiliary file (e.g. an omini
            # control image like 'foo_canny.png'), not a sample.
            continue
        img_path = os.path.join(input_dir, fname)
        samples.append((stem, img_path, json_path))
    if not samples:
        raise FileNotFoundError(f"No image/JSON pairs found in {input_dir}")
    return samples


def find_omini_image(input_dir: str, stem: str, omini_name: str) -> str:
    """Look up the per-sample omini control image: `<stem>_<omini_name>.<ext>`.

    Searches the same input directory for any of the accepted image extensions
    and returns the first match. Raises FileNotFoundError if none exist.
    """
    for ext in IMAGE_EXTS:
        candidate = os.path.join(input_dir, f"{stem}_{omini_name}{ext}")
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"Sample '{stem}': no omini control image for '{omini_name}'. "
        f"Expected one of {[f'{stem}_{omini_name}{e}' for e in IMAGE_EXTS]} "
        f"in {input_dir}."
    )


def load_seg_map_and_prompt(seg_map_path: str, seg_anno_path: str,
                            cond_scale_factor: int):
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
    color_to_text = {tuple(region["color"]): region["text"]
                     for region in seg_anno["segments_info"]}

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
        raise ValueError(f"No region colors in {seg_map_path} matched any "
                         f"'segments_info' entry in {seg_anno_path}.")

    label = torch.from_numpy(np.stack(label, axis=0)).long()  # (n, h, w)

    cond_pixel_values = np.zeros([label.shape[-2], label.shape[-1], 3], dtype=np.uint8)
    cond_pixel_values = visualizer.draw_contours(
        cond_pixel_values, label.cpu().numpy(),
        thickness=1, colors=[(255, 255, 255)] * len(regional_captions),
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


def load_omini_cond(image_path: str, width: int, height: int,
                    omini_cond_scale_factor: int) -> torch.Tensor:
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    cond_resolution = [height // omini_cond_scale_factor,
                       width // omini_cond_scale_factor]
    tf = transforms.Compose([
        transforms.Resize(cond_resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    img = Image.open(image_path).convert("RGB")
    return tf(img)


def load_lora_adapter(pipeline, name: str, path: str, weight_name: str | None = None):
    if os.path.isfile(path) and path.endswith(".safetensors"):
        pipeline.load_lora_weights(
            os.path.dirname(path), adapter_name=name,
            weight_name=os.path.basename(path),
        )
    elif os.path.isdir(path):
        if weight_name is None:
            candidates = [
                "pytorch_lora_weights.safetensors",
                f"{name}.safetensors",
            ]
            for c in candidates:
                if os.path.isfile(os.path.join(path, c)):
                    weight_name = c
                    break
            if weight_name is None:
                st = [f for f in os.listdir(path) if f.endswith(".safetensors")]
                if len(st) == 1:
                    weight_name = st[0]
                else:
                    raise FileNotFoundError(
                        f"Cannot auto-detect LoRA weight file in {path}. "
                        f"Found {st}. Pass an explicit filename."
                    )
        pipeline.load_lora_weights(path, adapter_name=name, weight_name=weight_name)
    else:
        raise FileNotFoundError(f"LoRA path not found: {path}")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_name_or_path", type=str,
                   default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--resume_from_checkpoint", type=str, required=True,
                   help="Directory containing 'cond/' subfolder (+ optional "
                        "adaptive_mask_module.pt, cond2img_scale.pt) OR direct "
                        "path to a cond .safetensors file.")
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
    # Omini conditions.
    p.add_argument("--omini", action="append", nargs=2, default=[],
                   metavar=("NAME", "LORA_PATH"),
                   help="Add one OminiControl stream. Repeat for multiple. "
                        "Per-sample control image is auto-discovered as "
                        "'<stem>_<NAME>.{png,jpg,jpeg}' inside --input_dir.")
    p.add_argument("--omini_cond_weights", type=float, nargs="*", default=None,
                   help="One weight per --omini, in declaration order.")
    p.add_argument("--omini_cond_scale_factor", type=int, default=1)
    p.add_argument("--omini_subject_streams", type=str, nargs="*", default=[],
                   help="Names of --omini streams that are subject-style "
                        "(reference image, not spatially aligned). Their cond "
                        "image position encodings are shifted to "
                        "[0, -W//16] so they sit outside the target's RoPE "
                        "window. Matches OminiControl's subject LoRA "
                        "convention. Spatially-aligned streams (canny, depth, "
                        "fill) should NOT be listed here.")
    # Generation knobs.
    p.add_argument("--conditional_integration_method", type=str,
                   choices=["none", "unified", "decoupled"], default="unified")
    p.add_argument("--attention_mask_method", type=str,
                   choices=["none", "base", "hard", "adaptive"], default="adaptive")
    p.add_argument("--zero_init_cond2img", action="store_true")
    p.add_argument("--hard_attn_block_range", type=int, nargs=2, default=[19, 37],
                   metavar=("START", "END"))
    p.add_argument("--is_filter_cond_token", action="store_true", default=True)
    p.add_argument("--no_filter_cond_token", dest="is_filter_cond_token",
                   action="store_false")
    p.add_argument("--cond2image_attention_weight", type=float, default=1.0)
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
    p.add_argument("--offload", type=str,
                   choices=["none", "model", "sequential"], default="none",
                   help="CPU offloading to trade speed for GPU memory. "
                        "'none': everything stays on the GPU (fastest). "
                        "'model': one pipeline component (text encoders / "
                        "transformer / vae) on the GPU at a time. "
                        "'sequential': stream weights layer-by-layer (slowest, "
                        "lowest memory; needed for seg + several omini streams "
                        "on a ~44GB card).")
    return p.parse_args()


def build_pipeline(args):
    """Load FLUX + seg adapter + every --omini LoRA once. Returns
    (pipeline, omini_adapter_names). Per-sample omini control images are
    resolved later in `generate_for_sample` via `find_omini_image`."""
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

    print("[infer] Loading seg LoRA ('cond')...")
    ckpt_dir = None
    ckpt = args.resume_from_checkpoint
    if os.path.isfile(ckpt) and ckpt.endswith(".safetensors"):
        pipeline.load_lora_weights(
            os.path.dirname(ckpt), adapter_name="cond",
            weight_name=os.path.basename(ckpt),
        )
        ckpt_dir = os.path.dirname(ckpt)
    elif os.path.exists(os.path.join(ckpt, "cond")):
        pipeline.load_lora_weights(
            os.path.join(ckpt, "cond"), adapter_name="cond",
            weight_name="pytorch_lora_weights.safetensors",
        )
        ckpt_dir = ckpt
    else:
        ckpt_dir = ckpt

    if args.zero_init_cond2img and ckpt_dir:
        scale_path = os.path.join(ckpt_dir, "cond2img_scale.pt")
        if os.path.exists(scale_path):
            print(f"[infer] Loading cond2img_scale from {scale_path}...")
            cond2img_state = torch.load(scale_path, map_location="cpu")
            for name, param in flux_transformer.named_parameters():
                if name in cond2img_state:
                    param.data.copy_(cond2img_state[name].to(param.dtype))
        else:
            print(f"[WARNING] zero_init_cond2img=True but no cond2img_scale.pt at {scale_path}")

    if use_adaptive_mask and ckpt_dir:
        amm_path = os.path.join(ckpt_dir, "adaptive_mask_module.pt")
        if os.path.exists(amm_path):
            print(f"[infer] Loading adaptive_mask_module from {amm_path}...")
            flux_transformer.adaptive_mask_module.load_state_dict(
                torch.load(amm_path, map_location="cpu")
            )
        else:
            print(f"[WARNING] attention_mask_method=adaptive but no "
                  f"adaptive_mask_module.pt at {amm_path}")

    omini_adapter_names = []
    for (name, lora_path) in args.omini:
        print(f"[infer] Loading omini LoRA adapter '{name}' from {lora_path}...")
        load_lora_adapter(pipeline, name, lora_path)
        omini_adapter_names.append(name)

    if args.omini_cond_weights is not None:
        if len(args.omini_cond_weights) != len(args.omini):
            raise ValueError(
                f"--omini_cond_weights has {len(args.omini_cond_weights)} entries "
                f"but --omini was given {len(args.omini)} times — must match."
            )

    all_adapter_names = ["cond"] + omini_adapter_names
    pipeline.set_adapters(all_adapter_names)
    pipeline.set_progress_bar_config(disable=False)

    if args.offload == "sequential":
        pipeline.enable_sequential_cpu_offload()
    elif args.offload == "model":
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to("cuda")
    return pipeline, omini_adapter_names


def generate_for_sample(pipeline, args, omini_adapter_names,
                        stem, seg_map_path, seg_anno_path, seeds):
    """Generate `len(seeds)` images for one sample, one per seed.

    Per-sample omini control images are resolved here: for every active
    omini adapter `<name>`, look up `<stem>_<name>.{png,jpg,jpeg}` in
    `args.input_dir`. Raises if any required omini image is missing.
    """
    os.makedirs(args.output_path, exist_ok=True)

    batch = load_seg_map_and_prompt(
        seg_map_path=seg_map_path,
        seg_anno_path=seg_anno_path,
        cond_scale_factor=args.cond_scale_factor,
    )

    omini_conds = None
    omini_position_deltas = None
    if omini_adapter_names:
        omini_conds = []
        omini_position_deltas = []
        # Subject-style streams need their cond positions shifted to
        # [0, -W_in_patches] so the cond image sits outside the target's
        # RoPE window. W_in_patches = batch["image_width"] // 16 (FLUX patch
        # downscale at the latent grid). Spatially-aligned streams get None.
        subject_set = set(args.omini_subject_streams)
        w_in_patches = batch["image_width"] // 16
        for name in omini_adapter_names:
            img_path = find_omini_image(args.input_dir, stem, name)
            print(f"[infer] [{stem}] omini '{name}' -> {img_path}")
            tensor = load_omini_cond(
                img_path,
                width=batch["image_width"],
                height=batch["image_height"],
                omini_cond_scale_factor=args.omini_cond_scale_factor,
            )
            omini_conds.append((tensor + 1) / 2.0)
            if name in subject_set:
                omini_position_deltas.append((0, -w_in_patches))
                print(f"[infer] [{stem}] omini '{name}' -> subject-style, "
                      f"position_delta=(0, {-w_in_patches})")
            else:
                omini_position_deltas.append(None)

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
            omini_conds=omini_conds,
            omini_adapter_names=omini_adapter_names if omini_conds else None,
            omini_cond_scale_factor=args.omini_cond_scale_factor,
            omini_cond_weights=args.omini_cond_weights,
            omini_cond_position_deltas=omini_position_deltas,
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

    pipeline, omini_adapter_names = build_pipeline(args)

    for stem, img_path, json_path in samples:
        try:
            generate_for_sample(
                pipeline, args, omini_adapter_names,
                stem, img_path, json_path, seeds,
            )
        except Exception as e:
            print(f"[infer] [{stem}] FAILED: {e}")


if __name__ == "__main__":
    main()
