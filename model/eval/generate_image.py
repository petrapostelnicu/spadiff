import os
import argparse
import copy
import sys
from pathlib import Path

# Add repo root to sys.path so that 'model.*' imports work
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import torch
import numpy as np
import time
from transformers import AutoProcessor, AutoModel
import accelerate
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, DistributedType, ProjectConfiguration, set_seed
from tqdm.auto import tqdm
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
from functools import partial
from torch.nn import functional as F
from peft.utils import get_peft_model_state_dict
from peft import LoraConfig, set_peft_model_state_dict
from safetensors.torch import load_file

from omegaconf import OmegaConf, DictConfig
from torchmetrics.multimodal import CLIPScore
from torchmetrics.image.fid import FrechetInceptionDistance
from cleanfid import fid as clean_fid
import ImageReward as RM

from model.src.models import FluxTransformer2DModel
from model.src.models.adaptive_mask import AdaptiveMaskModule
from model.src.pipelines import FluxRegionalPipeline
from model.dataset.collate_fn import collate_fn
from model.dataset.no_pad_sampler import NonPadDistributedSampler
from model.utils.utils import instantiate_from_config


def load_img_and_convert_tensor(img_path):
    img = Image.open(img_path).convert('RGB')
    img = np.array(img)
    img = torch.from_numpy(img).permute(2, 0, 1)  # c,h,w
    return img


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
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.trainer.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[kwargs],
    )

    gen_image_dir = Path(args.project.gen_image_dir)

    print(f"[{time.time():.1f}] Creating dataset...")
    val_dataset = instantiate_from_config(args.data.val)

    data_sampler = NonPadDistributedSampler(
        val_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index
    )

    print(f"[{time.time():.1f}] Creating dataloader...")
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
        sampler=data_sampler
    )

    if accelerator.is_main_process:
        os.makedirs(gen_image_dir, exist_ok=True)

        for i in range(args.eval.num_images_per_prompt):
            os.makedirs(os.path.join(gen_image_dir, f'group_{i}'), exist_ok=True)

    if args.seed is not None:
        set_seed(args.seed)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    print(f"[{time.time():.1f}] Loading transformer...")
    flux_transformer = FluxTransformer2DModel.from_pretrained(
        args.model.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
        conditional_integration_method=getattr(args.model, 'conditional_integration_method', 'none'),
        zero_init_cond2img=False,  # never pass True: from_pretrained uses init_empty_weights
    )

    # Add cond2img_scale parameters if requested
    if getattr(args.model, 'zero_init_cond2img', False):
        all_blocks = list(flux_transformer.transformer_blocks) + list(flux_transformer.single_transformer_blocks)
        for block in all_blocks:
            if block.conditional_integration_method == "decoupled":
                block.zero_init_cond2img = True
                block.cond2img_scale = torch.nn.Parameter(torch.zeros(flux_transformer.inner_dim, dtype=weight_dtype))

    # Add AdaptiveMaskModule if using adaptive attention masks
    use_adaptive_mask = getattr(args.model, 'attention_mask_method', 'none') == 'adaptive'
    if use_adaptive_mask:
        num_layers = flux_transformer.config.num_layers + flux_transformer.config.num_single_layers
        flux_transformer.adaptive_mask_module = AdaptiveMaskModule(
            temb_dim=flux_transformer.inner_dim,
            num_layers=num_layers,
            num_heads=flux_transformer.config.num_attention_heads,
            hidden_dim=getattr(args.model, 'adaptive_mask_hidden_dim', 256),
            variant=getattr(args.model, 'adaptive_mask_variant', 'full'),
        ).to(weight_dtype)

    print(f"[{time.time():.1f}] Loading pipeline...")
    pipeline = FluxRegionalPipeline.from_pretrained(
        args.model.pretrained_model_name_or_path,
        transformer=flux_transformer,
        torch_dtype=weight_dtype,
    )

    # Load LoRA weights
    print(f"[{time.time():.1f}] Loading LoRA...")
    if args.resume_from_checkpoint:
        if os.path.isfile(args.resume_from_checkpoint) and args.resume_from_checkpoint.endswith('.safetensors'):
            # Direct path to safetensors file
            pipeline.load_lora_weights(args.resume_from_checkpoint, adapter_name="cond")
            pipeline.set_adapters(['cond'])
            ckpt_dir = os.path.dirname(args.resume_from_checkpoint)
        elif os.path.exists(os.path.join(args.resume_from_checkpoint, 'cond')):
            # Directory with 'cond' subfolder
            pipeline.load_lora_weights(os.path.join(args.resume_from_checkpoint, 'cond'), adapter_name="cond", weight_name="pytorch_lora_weights.safetensors")
            pipeline.set_adapters(['cond'])
            ckpt_dir = args.resume_from_checkpoint
        else:
            ckpt_dir = args.resume_from_checkpoint

        # Load cond2img_scale trained values if present
        if getattr(args.model, 'zero_init_cond2img', False):
            scale_path = os.path.join(ckpt_dir, 'cond2img_scale.pt')
            if os.path.exists(scale_path):
                print(f"[{time.time():.1f}] Loading cond2img_scale from {scale_path}...")
                cond2img_scale_state = torch.load(scale_path, map_location='cpu')
                for name, param in flux_transformer.named_parameters():
                    if name in cond2img_scale_state:
                        param.data.copy_(cond2img_scale_state[name].to(param.dtype))
            else:
                print(f"[WARNING] zero_init_cond2img=True but no cond2img_scale.pt found at {scale_path}")

        # Load adaptive_mask_module trained values if present
        if use_adaptive_mask:
            adaptive_mask_path = os.path.join(ckpt_dir, 'adaptive_mask_module.pt')
            if os.path.exists(adaptive_mask_path):
                print(f"[{time.time():.1f}] Loading adaptive_mask_module from {adaptive_mask_path}...")
                flux_transformer.adaptive_mask_module.load_state_dict(
                    torch.load(adaptive_mask_path, map_location='cpu')
                )
            else:
                print(f"[WARNING] attention_mask_method=adaptive but no adaptive_mask_module.pt found at {adaptive_mask_path}")

    pipeline.set_progress_bar_config(disable=True)

    # Memory optimization options
    print(f"[{time.time():.1f}] Moving to device...")
    generator_device = accelerator.device
    if getattr(args, 'enable_cpu_offload', False):
        # Moves models to GPU only when needed
        pipeline.enable_model_cpu_offload(device=accelerator.device)
        generator_device = "cpu"
    elif getattr(args, 'enable_sequential_cpu_offload', False):
        # Moves individual layers
        pipeline.enable_sequential_cpu_offload(device=accelerator.device)
        generator_device = "cpu"
    else:
        pipeline = pipeline.to(accelerator.device)
        # Debug: verify device placement
        print(f"[DEBUG] accelerator.device: {accelerator.device}")
        print(f"[DEBUG] pipeline._execution_device: {pipeline._execution_device}")
        print(f"[DEBUG] transformer device: {next(pipeline.transformer.parameters()).device}")

    if getattr(args, 'enable_attention_slicing', False):
        pipeline.enable_attention_slicing("auto")

    if getattr(args, 'enable_vae_slicing', False):
        pipeline.enable_vae_slicing()

    print(f"[{time.time():.1f}] Starting generation loop...")
    for batch in tqdm(val_dataloader):
        image_name = batch["image_name"][0]
        save_image_path = os.path.join(gen_image_dir, f'group_{args.eval.num_images_per_prompt - 1}', image_name)
        if os.path.exists(save_image_path):
            continue

        print(
            f"[DEBUG] {image_name} | global_caption: {batch['global_caption']} | regional_captions: {batch['regional_captions']}")

        # Save condition image for visualization
        cond_save_dir = os.path.join(gen_image_dir, 'cond_images')
        os.makedirs(cond_save_dir, exist_ok=True)
        cond_img = (batch["cond_pixel_values"][0] + 1) / 2.0  # denormalize, (3, H, W)
        cond_img = cond_img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        cond_img = (cond_img * 255).astype(np.uint8)
        cond_pil = Image.fromarray(cond_img)
        cond_h, cond_w = cond_img.shape[:2]

        # Draw region IDs at mask centroids if regional labels are provided
        if batch["label"] is not None and len(batch["label"]) > 0:
            draw = ImageDraw.Draw(cond_pil)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except (OSError, IOError):
                font = ImageFont.load_default()
            labels = batch["label"][0]  # (n_regions, H, W)
            label_h, label_w = labels.shape[-2], labels.shape[-1]
            for region_idx in range(labels.shape[0]):
                mask = labels[region_idx].cpu().numpy()
                ys, xs = np.where(mask > 0)
                if len(ys) == 0:
                    continue
                # Centroid in label space, scaled to cond image space
                cy = int(ys.mean() * cond_h / label_h)
                cx = int(xs.mean() * cond_w / label_w)
                text = str(region_idx)
                bbox = draw.textbbox((0, 0), text, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text((cx - tw // 2, cy - th // 2), text, fill=(255, 255, 0), font=font)

        cond_pil.save(os.path.join(cond_save_dir, image_name))

        # Fresh generators per call: group_i uses seed+i, so skipping images doesn't shift seeds
        # Use generator_device
        if args.seed is not None:
            generators = [torch.Generator(device=generator_device).manual_seed(args.seed + i)
                          for i in range(args.eval.num_images_per_prompt)]
        else:
            generators = None

        gen_images = pipeline(
            global_prompt=batch["global_caption"],
            regional_prompts=batch["regional_captions"],
            regional_labels=batch["label"],
            cond=(batch["cond_pixel_values"] + 1) / 2.0,  # denormalize
            attention_mask_method=args.model.attention_mask_method,
            conditional_integration_method=getattr(args.model, 'conditional_integration_method', 'none'),
            is_filter_cond_token=args.model.is_filter_cond_token,
            hard_attn_block_range=args.model.hard_attn_block_range,
            height=batch["pixel_values"].shape[-2],
            width=batch["pixel_values"].shape[-1],
            cond_scale_factor=args.cond_scale_factor,
            num_images_per_prompt=args.eval.num_images_per_prompt,
            guidance_scale=args.eval.guidance_scale,
            num_inference_steps=args.model.num_inference_steps,
            generator=generators,
            max_sequence_length=args.model.max_sequence_length,
            regional_max_sequence_length=args.model.regional_max_sequence_length
        ).images
        for i, image in enumerate(gen_images):
            save_image_path = os.path.join(gen_image_dir, f'group_{i}', image_name)
            image.save(save_image_path)

    accelerator.wait_for_everyone()
    del pipeline
    del flux_transformer


if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    unknown = [s.lstrip('-') for s in unknown]
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    print('###### cli input evaluation setup:  ######\n', cli)
    config = OmegaConf.merge(*configs, cli)

    if config.resolution % (16 * config.cond_scale_factor) != 0:
        raise ValueError(
            f"Image resolution {config.resolution} must be divisible by {16 * config.cond_scale_factor} "
            f"(16 * cond_scale_factor) to ensure proper feature map alignment in the model. "
            f"Please adjust either the resolution or cond_scale_factor."
        )

    main(config)
