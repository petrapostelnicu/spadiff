#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

# Add package paths for development
from pathlib import Path
import sys
import os

import resource
try:
    _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if _soft < _hard:
        resource.setrlimit(resource.RLIMIT_NOFILE, (_hard, _hard))
        print(f"[FD-LIMIT] raised RLIMIT_NOFILE: {_soft} -> {_hard}", flush=True)
except (ValueError, OSError) as _e:
    print(f"[FD-LIMIT] could not raise RLIMIT_NOFILE: {_e}", flush=True)

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from torch import nn

project_root = Path(__file__).parent.parent
repo_root = project_root.parent
sys.path.insert(0, str(repo_root))

import os
import argparse
import copy
import logging
import math
import os
import json
import random
import shutil
from pathlib import Path

import time
import accelerate
import numpy as np
import cv2
import torch
import torch.utils.checkpoint
from torch.nn import functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, DataLoaderConfiguration, set_seed, \
    DistributedDataParallelKwargs, InitProcessGroupKwargs
from datetime import timedelta
from packaging import version
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from tqdm.auto import tqdm
from safetensors.torch import load_file

from omegaconf import OmegaConf, DictConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_density_for_timestep_sampling, \
    compute_loss_weighting_for_sd3, free_memory
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module

from model.src.models import FluxTransformer2DModel
from model.src.models.adaptive_mask import AdaptiveMaskModule
from model.src.pipelines import FluxRegionalPipeline
from model.dataset.collate_fn import collate_fn
from model.dataset.group_sampler import GroupSampler
from model.utils.utils import instantiate_from_config
from model.utils.visualizer import Visualizer

if is_wandb_available():
    import wandb

check_min_version("0.32.2")

logger = get_logger(__name__)

visualizer = Visualizer()


def encode_images(pixels: torch.Tensor, vae: torch.nn.Module, weight_dtype):
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    pixel_latents = (pixel_latents - vae.config.shift_factor) * vae.config.scaling_factor
    return pixel_latents.to(weight_dtype)


def get_lora_target_modules(lora_layers, flux_transformer):
    if lora_layers is not None:
        if lora_layers == "all-linear":
            target_modules = set()
            for name, module in flux_transformer.named_modules():
                if isinstance(module, torch.nn.Linear):
                    target_modules.add(name)
            target_modules = list(target_modules)
        elif lora_layers == "all-linear-in-dit-blocks":
            target_modules = set()
            for name, module in flux_transformer.named_modules():
                if name.startswith("transformer_blocks") or name.startswith("single_transformer_blocks"):
                    if isinstance(module, torch.nn.Linear):
                        target_modules.add(name)
            target_modules = list(target_modules)
        elif lora_layers.startswith("regular_expression:"):
            target_modules = lora_layers[len("regular_expression:"):]
        else:
            target_modules = [layer.strip() for layer in lora_layers.split(",")]
    else:
        target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]
    return target_modules


def log_validation(flux_transformer, vae, text_encoding_pipeline, args, val_dataloader, accelerator, weight_dtype,
                   step):
    logger.info("Running validation... ")

    flux_transformer = accelerator.unwrap_model(flux_transformer)

    pipeline = FluxRegionalPipeline.from_pretrained(
        args.model.pretrained_model_name_or_path,
        transformer=flux_transformer,
        vae=vae,
        text_encoder=text_encoding_pipeline.text_encoder,
        text_encoder_2=text_encoding_pipeline.text_encoder_2,
        tokenizer=text_encoding_pipeline.tokenizer,
        tokenizer_2=text_encoding_pipeline.tokenizer_2,
        torch_dtype=weight_dtype,
    )
    pipeline.set_progress_bar_config(disable=True)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    image_logs = []

    autocast_ctx = torch.autocast(accelerator.device.type, weight_dtype)
    num_validation_images = min(args.trainer.num_validation_images, len(val_dataloader))
    for i, batch in enumerate(val_dataloader):
        # note: val_dataloader bs == 1
        images = []

        # Determine what features to use
        use_regional_control = args.model.attention_mask_method != "none"
        use_conditional = getattr(args.model, 'conditional_integration_method', 'none') != 'none'

        with autocast_ctx:
            image = pipeline(
                global_prompt=batch["global_caption"],
                regional_prompts=batch["regional_captions"] if use_regional_control else None,
                regional_labels=batch["label"] if use_regional_control else None,
                cond=(batch["cond_pixel_values"] + 1) / 2.0 if use_conditional else None,  # denormalize
                attention_mask_method=args.model.attention_mask_method,
                conditional_integration_method=getattr(args.model, 'conditional_integration_method', 'none'),
                is_filter_cond_token=args.model.is_filter_cond_token,
                hard_attn_block_range=args.model.hard_attn_block_range,
                height=batch["pixel_values"].shape[-2],
                width=batch["pixel_values"].shape[-1],
                cond_scale_factor=args.cond_scale_factor,
                num_images_per_prompt=1,
                guidance_scale=args.eval.guidance_scale,
                num_inference_steps=args.model.num_inference_steps,
                generator=generator,
                max_sequence_length=args.model.max_sequence_length,
                regional_max_sequence_length=args.model.regional_max_sequence_length
            ).images[0]
        images.append(image)

        # Get ground truth image - handle both file path and tensor-based datasets
        image_path = batch.get('image_path', [None])[0]
        if image_path is not None and os.path.exists(str(image_path)):
            gt_image = Image.open(image_path).convert('RGB')
            gt_image = gt_image.resize(image.size, resample=Image.BICUBIC)
            gt_image = np.array(gt_image)
        else:
            # Use the pixel_values tensor as ground truth (denormalize from [-1, 1] to [0, 255])
            gt_tensor = batch["pixel_values"][0]  # C, H, W
            gt_tensor = (gt_tensor + 1) / 2.0  # [-1, 1] -> [0, 1]
            gt_tensor = gt_tensor.clamp(0, 1)
            gt_image = (gt_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            gt_pil = Image.fromarray(gt_image)
            gt_pil = gt_pil.resize(image.size, resample=Image.BICUBIC)
            gt_image = np.array(gt_pil)

        # Get condition image for visualization (only if using conditional integration)
        cond_image = None
        if use_conditional:
            cond_tensor = batch["cond_pixel_values"][0]  # C, H, W
            cond_tensor = (cond_tensor + 1) / 2.0  # [-1, 1] -> [0, 1]
            cond_tensor = cond_tensor.clamp(0, 1)
            cond_image = (cond_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            cond_pil = Image.fromarray(cond_image)
            cond_pil = cond_pil.resize(image.size, resample=Image.BICUBIC)
            cond_image = np.array(cond_pil)

        # Create image with regional labels overlay (only if using regional control)
        image_with_label = None
        if use_regional_control and batch["label"] is not None and len(batch["label"]) > 0 and batch["regional_captions"] is not None:
            image_with_label = np.array(image)
            image_with_label = cv2.cvtColor(image_with_label, cv2.COLOR_RGB2BGR)

            label = batch["label"][0]
            label = F.interpolate(label[None].float(), size=image_with_label.shape[:2], mode='nearest-exact')
            label = label[0, ...].long()
            label = label.cpu().numpy()
            image_with_label = visualizer.draw_binary_mask_with_caption(
                image_with_label, label, batch["regional_captions"][0], alpha=0.4
            )
            image_with_label = cv2.cvtColor(image_with_label, cv2.COLOR_BGR2RGB)

        image_logs.append({
            "ground_truth": gt_image,
            "condition": cond_image,
            "image_with_label": image_with_label,
            "images": images,
            "global_caption": batch["global_caption"][0] if batch["global_caption"] is not None else str(i)
        })

        if i == num_validation_images:
            break

    tracker_key = "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for i, log in enumerate(image_logs):
                images = log["images"]
                global_caption = log["global_caption"]
                ground_truth = log["ground_truth"]
                condition = log["condition"]
                image_with_label = log["image_with_label"]

                formatted_images = [ground_truth]
                if condition is not None:
                    formatted_images.append(condition)
                if image_with_label is not None:
                    formatted_images.append(image_with_label)
                for image in images:
                    formatted_images.append(np.asarray(image))
                formatted_images = np.stack(formatted_images)
                tracker.writer.add_images(global_caption, formatted_images, step, dataformats="NHWC")

        elif tracker.name == "wandb":
            formatted_images = []
            for i, log in enumerate(image_logs):
                images = log["images"]
                global_caption = log["global_caption"]
                ground_truth = log["ground_truth"]
                condition = log["condition"]
                image_with_label = log["image_with_label"]

                formatted_images.append(wandb.Image(ground_truth, caption="ground_truth"))
                if condition is not None:
                    formatted_images.append(wandb.Image(condition, caption="condition"))
                if image_with_label is not None:
                    formatted_images.append(wandb.Image(image_with_label, caption="image_with_label"))
                for image in images:
                    image = wandb.Image(image, caption=global_caption)
                    formatted_images.append(image)

            tracker.log({tracker_key: formatted_images})
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

    del pipeline
    free_memory()
    return image_logs


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    return parser


def main(args):
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )
    logging_dir = Path(args.project.output_dir, args.project.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.project.output_dir, logging_dir=logging_dir)
    init_process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))  # nccl timeout
    accelerator = Accelerator(
        gradient_accumulation_steps=args.trainer.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.project.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[init_process_group_kwargs],
    )

    if accelerator.gradient_accumulation_steps > 1:
        import contextlib
        _original_no_sync = accelerator.no_sync
        @contextlib.contextmanager
        def _patched_no_sync(model):
            if accelerator.distributed_type == DistributedType.DEEPSPEED and accelerator.state.deepspeed_plugin.zero_stage >= 2:
                yield
            else:
                with _original_no_sync(model):
                    yield
        accelerator.no_sync = _patched_no_sync

    if torch.backends.mps.is_available():
        logger.info("MPS is enabled. Disabling AMP.")
        accelerator.native_amp = False

    if args.project.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.project.output_dir is not None:
            os.makedirs(args.project.output_dir, exist_ok=True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    print(f"[DEBUG][rank {accelerator.process_index}] Creating train dataset...", flush=True)
    train_dataset = instantiate_from_config(args.data.train)
    print(f"[DEBUG][rank {accelerator.process_index}] Creating val dataset...", flush=True)
    val_dataset = instantiate_from_config(args.data.val)

    print(f"[DEBUG][rank {accelerator.process_index}] Creating train dataloader...", flush=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=GroupSampler(train_dataset, samples_per_gpu=args.trainer.train_batch_size * accelerator.num_processes),
        collate_fn=collate_fn,
        batch_size=args.trainer.train_batch_size,
        num_workers=args.dataloader_num_workers,
        persistent_workers=args.dataloader_num_workers > 0,
        prefetch_factor=4 if args.dataloader_num_workers > 0 else None,
    )

    print(f"[DEBUG][rank {accelerator.process_index}] Creating val dataloader...", flush=True)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
    )

    # Load models.
    vae = AutoencoderKL.from_pretrained(
        args.model.pretrained_model_name_or_path,
        subfolder="vae",
    )
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    # Create a pipeline for text encoding. We will move this pipeline to GPU/CPU as needed.
    text_encoding_pipeline = FluxRegionalPipeline.from_pretrained(
        args.model.pretrained_model_name_or_path, transformer=None, vae=None, torch_dtype=weight_dtype
    )

    flux_transformer = FluxTransformer2DModel.from_pretrained(
        args.model.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
        conditional_integration_method=getattr(args.model, 'conditional_integration_method', 'none'),
        zero_init_cond2img=False,  # never pass True here: from_pretrained uses init_empty_weights,
        # which makes cond2img_scale a meta tensor. Since it's absent from the pretrained
        # checkpoint it would never get real data, causing model.to(device) to fail.
        # The parameter is added manually below after the model is fully loaded.
    )

    # Add cond2img_scale parameters now that the model is fully loaded on real memory.
    # Blocks were created with zero_init_cond2img=False (required by from_pretrained to avoid
    # meta tensor issues), so we set the flag and create the parameter manually here.
    if getattr(args.model, 'zero_init_cond2img', False):
        all_blocks = list(flux_transformer.transformer_blocks) + list(flux_transformer.single_transformer_blocks)
        for block in all_blocks:
            if block.conditional_integration_method == "decoupled":
                block.zero_init_cond2img = True
                block.cond2img_scale = nn.Parameter(torch.zeros(flux_transformer.inner_dim, dtype=weight_dtype))

    # Add AdaptiveMaskModule if enabled (post-load, same pattern as cond2img_scale)
    use_adaptive_mask = getattr(args.model, 'attention_mask_method', 'none') == 'adaptive'
    if use_adaptive_mask:
        adaptive_mask_hidden_dim = getattr(args.model, 'adaptive_mask_hidden_dim', 256)
        adaptive_mask_variant = getattr(args.model, 'adaptive_mask_variant', 'full')
        num_layers = flux_transformer.config.num_layers + flux_transformer.config.num_single_layers
        flux_transformer.adaptive_mask_module = AdaptiveMaskModule(
            temb_dim=flux_transformer.inner_dim,
            num_layers=num_layers,
            num_heads=flux_transformer.config.num_attention_heads,
            hidden_dim=adaptive_mask_hidden_dim,
            variant=adaptive_mask_variant,
        )
        # Load pretrained adaptive mask weights before accelerator.prepare so that DeepSpeed's
        # FP32 optimizer master copy is created from these weights.
        if getattr(args, 'init_adaptive_mask_from', None):
            rc = args.resume_from_checkpoint
            will_resume = False
            if rc and rc != 'null':
                if rc == 'latest':
                    # Resolve "latest": only a real resume if a checkpoint-* folder exists
                    out_dir = args.project.output_dir
                    if os.path.isdir(out_dir):
                        existing = [d for d in os.listdir(out_dir) if d.startswith('checkpoint')]
                        will_resume = len(existing) > 0
                elif os.path.isfile(rc) and rc.endswith('.safetensors'):
                    # Safetensors LoRA resume resets args.resume_from_checkpoint to None;
                    # still load the mask here since the safetensors path only touches LoRA.
                    will_resume = False
                else:
                    will_resume = True
            if not will_resume:
                adaptive_mask_path = args.init_adaptive_mask_from
                logger.info(f"Loading adaptive_mask_module from {adaptive_mask_path}")
                adaptive_mask_state = torch.load(adaptive_mask_path, map_location='cpu')
                flux_transformer.adaptive_mask_module.load_state_dict(adaptive_mask_state)
                logger.info("Loaded adaptive_mask_module successfully")

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.model.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    logger.info("All models loaded successfully")

    vae.requires_grad_(False)
    flux_transformer.requires_grad_(False)
    flux_transformer.eval()

    # Move vae, transformer and text_encoding_pipeline to device and cast to weight_dtype
    print(f"[DEBUG][rank {accelerator.process_index}] Moving VAE to {accelerator.device}...", flush=True)
    vae.to(accelerator.device, dtype=torch.float32)
    print(f"[DEBUG][rank {accelerator.process_index}] Moving transformer to {accelerator.device}...", flush=True)
    flux_transformer.to(accelerator.device, dtype=weight_dtype)
    print(f"[DEBUG][rank {accelerator.process_index}] Moving text_encoding_pipeline to {accelerator.device}...",
          flush=True)
    text_encoding_pipeline = text_encoding_pipeline.to(accelerator.device)

    print(f"[DEBUG][rank {accelerator.process_index}] Getting LoRA target modules...", flush=True)
    cond_target_modules = get_lora_target_modules(args.model.cond_lora_layers, flux_transformer)

    # Check if we should use conditional integration
    use_conditional = getattr(args.model, 'conditional_integration_method', 'none') != 'none'

    if use_conditional:
        cond_lora_config = LoraConfig(
            r=args.model.rank,
            lora_alpha=args.model.rank,
            init_lora_weights="gaussian" if args.model.gaussian_init_lora else True,
            target_modules=cond_target_modules,
            lora_bias=args.model.use_lora_bias,
        )
        print(f"[DEBUG][rank {accelerator.process_index}] Adding LoRA adapter...", flush=True)
        flux_transformer.add_adapter(cond_lora_config, adapter_name='cond')
        flux_transformer.set_adapter(['cond'])

    # cond2img_scale is a plain nn.Parameter on the base model (not part of the LoRA adapter),
    # so it gets frozen by requires_grad_(False) above. Unfreeze it here.
    if getattr(args.model, 'zero_init_cond2img', False):
        for name, param in flux_transformer.named_parameters():
            if 'cond2img_scale' in name:
                param.requires_grad = True

    # adaptive_mask_module is a plain nn.Module on the base model (not part of the LoRA adapter),
    # so it gets frozen by requires_grad_(False) above. Unfreeze it so it co-trains with LoRA.
    if use_adaptive_mask:
        for param in flux_transformer.adaptive_mask_module.parameters():
            param.requires_grad = True

    # Safetensors warm-start (e.g., phase-2 starting from a phase-1 LoRA + cond2img_scale + mask).
    if args.resume_from_checkpoint and isinstance(args.resume_from_checkpoint, str) \
            and os.path.isfile(args.resume_from_checkpoint) \
            and args.resume_from_checkpoint.endswith('.safetensors'):
        logger.info(f"Loading LoRA weights from safetensors file: {args.resume_from_checkpoint}")
        from safetensors.torch import load_file
        raw_state_dict = load_file(args.resume_from_checkpoint)

        model_params = dict(flux_transformer.named_parameters())
        loaded, skipped = 0, 0
        for k, v in raw_state_dict.items():
            param_key = k.replace("transformer.", "", 1) if k.startswith("transformer.") else k
            param_key = param_key.replace(".lora_A.weight", ".lora_A.cond.weight")
            param_key = param_key.replace(".lora_B.weight", ".lora_B.cond.weight")
            if param_key in model_params:
                model_params[param_key].data.copy_(v.to(model_params[param_key].dtype))
                loaded += 1
            elif f"base_model.model.{param_key}" in model_params:
                tgt = model_params[f"base_model.model.{param_key}"]
                tgt.data.copy_(v.to(tgt.dtype))
                loaded += 1
            else:
                skipped += 1
        logger.info(f"Loaded {loaded} LoRA weights directly, skipped {skipped} keys")

        ckpt_dir = os.path.dirname(args.resume_from_checkpoint)
        scale_path = os.path.join(ckpt_dir, '..', 'cond2img_scale.pt')
        if not os.path.exists(scale_path):
            scale_path = os.path.join(ckpt_dir, 'cond2img_scale.pt')
        if os.path.exists(scale_path) and getattr(args.model, 'zero_init_cond2img', False):
            cond2img_state = torch.load(scale_path, map_location='cpu')
            for name, param in flux_transformer.named_parameters():
                if name in cond2img_state:
                    param.data.copy_(cond2img_state[name].to(param.dtype))
            logger.info(f"Loaded cond2img_scale from {scale_path}")

        if use_adaptive_mask:
            mask_path = os.path.join(ckpt_dir, '..', 'adaptive_mask_module.pt')
            if not os.path.exists(mask_path):
                mask_path = os.path.join(ckpt_dir, 'adaptive_mask_module.pt')
            if os.path.exists(mask_path):
                adaptive_mask_state = torch.load(mask_path, map_location='cpu')
                flux_transformer.adaptive_mask_module.load_state_dict(adaptive_mask_state)
                logger.info(f"Loaded adaptive_mask_module from {mask_path}")

        # Mark consumed so the post-prepare resume block doesn't re-run this load.
        args.resume_from_checkpoint = None

    if args.trainer.gradient_checkpointing:
        flux_transformer.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                cond_lora_layers_to_save = None
                for model in models:
                    if isinstance(unwrap_model(model), type(unwrap_model(flux_transformer))):
                        transformer_ = unwrap_model(model)

                        if hasattr(transformer_, 'peft_config') and "cond" in transformer_.peft_config:
                            cond_lora_layers_to_save = get_peft_model_state_dict(transformer_, adapter_name="cond")

                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

                if cond_lora_layers_to_save is not None:
                    FluxRegionalPipeline.save_lora_weights(
                        os.path.join(output_dir, 'cond'),
                        transformer_lora_layers=cond_lora_layers_to_save,
                    )

                # Save cond2img_scale separately (not part of LoRA adapter state)
                cond2img_scale_state = {
                    name: param.data.cpu()
                    for name, param in transformer_.named_parameters()
                    if 'cond2img_scale' in name
                }
                if cond2img_scale_state:
                    torch.save(cond2img_scale_state, os.path.join(output_dir, 'cond2img_scale.pt'))

                # Save adaptive_mask_module separately (not part of LoRA adapter state)
                if use_adaptive_mask and hasattr(transformer_, 'adaptive_mask_module'):
                    torch.save(
                        transformer_.adaptive_mask_module.state_dict(),
                        os.path.join(output_dir, 'adaptive_mask_module.pt'),
                    )

        def load_model_hook(models, input_dir):
            transformer_ = None
            if not accelerator.distributed_type == DistributedType.DEEPSPEED:
                while len(models) > 0:
                    model = models.pop()

                    if isinstance(model, type(unwrap_model(flux_transformer))):
                        transformer_ = model
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")
            else:
                transformer_ = FluxTransformer2DModel.from_pretrained(
                    args.model.pretrained_model_name_or_path,
                    subfolder="transformer",
                    conditional_integration_method=getattr(args.model, 'conditional_integration_method', 'none'),
                    zero_init_cond2img=False,
                )
                if getattr(args.model, 'zero_init_cond2img', False):
                    all_blocks = list(transformer_.transformer_blocks) + list(transformer_.single_transformer_blocks)
                    for block in all_blocks:
                        if block.conditional_integration_method == "decoupled":
                            block.zero_init_cond2img = True
                            block.cond2img_scale = nn.Parameter(torch.zeros(transformer_.inner_dim, dtype=weight_dtype))
                if use_adaptive_mask:
                    num_layers = transformer_.config.num_layers + transformer_.config.num_single_layers
                    transformer_.adaptive_mask_module = AdaptiveMaskModule(
                        temb_dim=transformer_.inner_dim,
                        num_layers=num_layers,
                        num_heads=transformer_.config.num_attention_heads,
                        hidden_dim=getattr(args.model, 'adaptive_mask_hidden_dim', 256),
                        variant=adaptive_mask_variant,
                    )
                if use_conditional:
                    transformer_.add_adapter(cond_lora_config, adapter_name='cond')
                    transformer_.set_adapter(['cond'])

            # load transformer
            lora_path = None
            if os.path.isfile(input_dir) and input_dir.endswith('.safetensors'):
                # Direct safetensors file
                lora_path = input_dir
            elif os.path.exists(os.path.join(input_dir, 'cond')):
                # Directory with 'cond' subfolder
                lora_path = os.path.join(input_dir, 'cond')

            if lora_path is not None:
                lora_state_dict = FluxRegionalPipeline.lora_state_dict(lora_path)
                transformer_lora_state_dict = {
                    f'{k.replace("transformer.", "")}': v
                    for k, v in lora_state_dict.items()
                    if k.startswith("transformer.") and "lora" in k
                }
                incompatible_keys = set_peft_model_state_dict(
                    transformer_, transformer_lora_state_dict, adapter_name="cond"
                )
                if incompatible_keys is not None:
                    # check only for unexpected keys
                    unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                    if unexpected_keys:
                        logger.warning(
                            f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                            f" {unexpected_keys}. "
                        )

            # Restore cond2img_scale if saved alongside this checkpoint
            scale_path = os.path.join(input_dir, 'cond2img_scale.pt')
            if os.path.exists(scale_path) and transformer_ is not None:
                cond2img_scale_state = torch.load(scale_path, map_location='cpu')
                for name, param in transformer_.named_parameters():
                    if name in cond2img_scale_state:
                        param.data.copy_(cond2img_scale_state[name])
                logger.info(f"Loaded cond2img_scale from {scale_path}")

            # Restore adaptive_mask_module if saved alongside this checkpoint
            adaptive_mask_path = os.path.join(input_dir, 'adaptive_mask_module.pt')
            if os.path.exists(adaptive_mask_path) and transformer_ is not None and hasattr(transformer_, 'adaptive_mask_module'):
                adaptive_mask_state = torch.load(adaptive_mask_path, map_location='cpu')
                transformer_.adaptive_mask_module.load_state_dict(adaptive_mask_state)
                logger.info(f"Loaded adaptive_mask_module from {adaptive_mask_path}")

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.trainer.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.trainer.scale_lr:
        args.trainer.learning_rate = (
                args.trainer.learning_rate * args.trainer.gradient_accumulation_steps * args.trainer.train_batch_size * accelerator.num_processes
        )

    optimizer_class = torch.optim.AdamW

    # Optimization parameters
    cond2img_scale_lr_multiplier = getattr(args.optimizer, 'cond2img_scale_lr_multiplier', 1.0)
    adaptive_mask_lr_multiplier = getattr(args.optimizer, 'adaptive_mask_lr_multiplier', 1.0)
    scale_params = []
    mask_params = []
    other_params = []
    for name, param in flux_transformer.named_parameters():
        if not param.requires_grad:
            continue
        if 'cond2img_scale' in name:
            scale_params.append(param)
        elif 'adaptive_mask_module' in name:
            mask_params.append(param)
        else:
            other_params.append(param)

    params_group = [{'params': other_params, 'lr': args.trainer.learning_rate}]
    if scale_params:
        scale_lr = args.trainer.learning_rate * cond2img_scale_lr_multiplier
        params_group.append({'params': scale_params, 'lr': scale_lr})
        logger.info(f"cond2img_scale LR: {scale_lr} ({cond2img_scale_lr_multiplier}x base LR)")
    if mask_params:
        mask_lr = args.trainer.learning_rate * adaptive_mask_lr_multiplier
        params_group.append({'params': mask_params, 'lr': mask_lr})
        logger.info(f"adaptive_mask_module LR: {mask_lr} ({adaptive_mask_lr_multiplier}x base LR)")

    optimizer = optimizer_class(
        params_group,
        lr=args.trainer.learning_rate,
        betas=(args.optimizer.adam_beta1, args.optimizer.adam_beta2),
        weight_decay=args.optimizer.adam_weight_decay,
        eps=args.optimizer.adam_epsilon,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.trainer.gradient_accumulation_steps)
    if args.trainer.max_train_steps is None:
        args.trainer.max_train_steps = args.trainer.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    cond2img_scale_lr_scheduler = getattr(args.scheduler, 'cond2img_scale_lr_scheduler', None)
    if scale_params and cond2img_scale_lr_scheduler in ('cosine', 'linear'):
        # Per-group scheduling: constant for LoRA, decay for cond2img_scale
        total_steps = args.trainer.max_train_steps * accelerator.num_processes
        if cond2img_scale_lr_scheduler == 'cosine':
            scale_lambda = lambda step: 0.5 * (1.0 + math.cos(math.pi * step / total_steps))
        else:  # linear
            scale_lambda = lambda step: max(0.0, 1.0 - step / total_steps)
        # Group order matches params_group construction: [other, scale, mask?]
        lambdas = [lambda step: 1.0, scale_lambda]
        if mask_params:
            lambdas.append(lambda step: 1.0)  # mask group: constant
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)
        logger.info(f"Using per-group scheduler: constant (LoRA) + {cond2img_scale_lr_scheduler} decay (cond2img_scale)")
    else:
        lr_scheduler = get_scheduler(
            args.scheduler.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.scheduler.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.trainer.max_train_steps * accelerator.num_processes,
            num_cycles=args.scheduler.lr_num_cycles,
            power=args.scheduler.lr_power,
        )
    # Prepare everything with our `accelerator`.
    print(f"[DEBUG][rank {accelerator.process_index}] Calling accelerator.prepare() (DDP/NCCL sync)...", flush=True)
    flux_transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        flux_transformer, optimizer, train_dataloader, lr_scheduler
    )
    print(f"[DEBUG][rank {accelerator.process_index}] accelerator.prepare() done.", flush=True)
    unwrap_flux_transformer = unwrap_model(flux_transformer)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.trainer.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.trainer.max_train_steps = args.trainer.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.trainer.num_train_epochs = math.ceil(args.trainer.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = json.dumps(OmegaConf.to_container(args, resolve=True))
        tracker_config = {"tracker_config": tracker_config}
        accelerator.init_trackers(args.project.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.trainer.train_batch_size * accelerator.num_processes * args.trainer.gradient_accumulation_steps

    if accelerator.is_main_process:
        trainable_params = [p for p in unwrap_flux_transformer.parameters() if p.requires_grad]
        total_params_count = sum(p.numel() for p in unwrap_flux_transformer.parameters())
        trainable_params_count = sum(p.numel() for p in trainable_params)

        print("\n====== flux transformers Parameter Statistics ======")
        print(f"Total Parameters: {total_params_count}, ({total_params_count / 1e6:.2f}M)")
        print(f"Trainable Parameters: {trainable_params_count}, ({trainable_params_count / 1e6:.2f}M)")
        print(f"Trainable %: {trainable_params_count / total_params_count * 100:.4f}%")
        print("==================================\n")

    print(f"[DEBUG][rank {accelerator.process_index}] All setup complete, starting training.", flush=True)
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.trainer.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.trainer.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.trainer.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.trainer.max_train_steps}")
    global_step = 0
    first_epoch = 0
    resumed_from_checkpoint = False

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.project.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            logger.info(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
        else:
            logger.info(f"Resuming from checkpoint {path}")
            accelerator.load_state(
                os.path.join(args.project.output_dir, path),
                load_module_strict=False,
            )
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            resume_global_step = global_step * args.trainer.gradient_accumulation_steps
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.trainer.gradient_accumulation_steps)
            resumed_from_checkpoint = True


    progress_bar = tqdm(
        range(0, args.trainer.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    image_logs = None

    # Log validation at step 0 to verify pretrained behavior before any training
    if global_step == 0 and accelerator.is_main_process and args.trainer.num_validation_images > 0:
        image_logs = log_validation(
            flux_transformer, vae, text_encoding_pipeline, args,
            val_dataloader=val_dataloader,
            accelerator=accelerator,
            weight_dtype=weight_dtype,
            step=0,
        )

    for epoch in range(first_epoch, args.trainer.num_train_epochs):

        # Set epoch for distributed samplers (if applicable)
        if hasattr(train_dataloader, 'set_epoch'):
            train_dataloader.set_epoch(epoch)
        elif hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)

        if args.resume_from_checkpoint and epoch == first_epoch:
            # Skip steps until we reach the resumed step
            work_dataloader = accelerate.skip_first_batches(train_dataloader, num_batches=resume_step)
        else:
            work_dataloader = train_dataloader

        # [DIAGNOSTIC] timing breakdown for first few microbatches.
        _diag_enable = True
        _diag_max_steps = 10

        def _diag_active(step):
            return _diag_enable and accelerator.is_main_process and step < _diag_max_steps

        def _diag_t():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            import time as _time
            return _time.time()

        _diag_t0_ref = [None]
        _diag_prev_end_ref = [None]

        for step, batch in enumerate(work_dataloader):
            if _diag_active(step):
                _t_fetch = _diag_t()
                _gap = (_t_fetch - _diag_prev_end_ref[0]) if _diag_prev_end_ref[0] is not None else 0.0
                _diag_t0_ref[0] = _t_fetch
                print(f"[DIAG] step={step} batch_fetched gap_since_prev={_gap:.2f}s", flush=True)
            with accelerator.accumulate(flux_transformer):
                # vae encode
                pixel_latents = encode_images(batch["pixel_values"], vae.to(accelerator.device), weight_dtype)
                bsz = pixel_latents.shape[0]

                cond_pixel_latents = None
                cond_ids = None
                cond_seq_lens = [0 for _ in range(bsz)]
                pad_seq_lens = [0 for _ in range(bsz)]

                if use_conditional:
                    cond_pixel_latents = encode_images(batch["cond_pixel_values"], vae.to(accelerator.device),
                                                       weight_dtype)
                    cond_pixel_latents = FluxRegionalPipeline._pack_latents(
                        cond_pixel_latents,
                        batch_size=cond_pixel_latents.shape[0],
                        num_channels_latents=cond_pixel_latents.shape[1],
                        height=cond_pixel_latents.shape[2],
                        width=cond_pixel_latents.shape[3],
                    )

                    cond_ids = FluxRegionalPipeline._prepare_latent_image_ids(
                        batch["cond_pixel_values"].shape[0],
                        batch["cond_pixel_values"].shape[-2] // (vae_scale_factor * 2),
                        batch["cond_pixel_values"].shape[-1] // (vae_scale_factor * 2),
                        accelerator.device,
                        weight_dtype,
                    )
                    assert batch["pixel_values"].shape[-2] // batch["cond_pixel_values"].shape[-2] == \
                           batch["pixel_values"].shape[-1] // batch["cond_pixel_values"].shape[-1]
                    assert batch["pixel_values"].shape[-1] % batch["cond_pixel_values"].shape[-1] == 0
                    cond_scale_factor = batch["pixel_values"].shape[-1] // batch["cond_pixel_values"].shape[-1]
                    scale_bias = (cond_scale_factor - 1.0) / 2
                    cond_ids[..., 1:] = cond_ids[..., 1:] * cond_scale_factor + scale_bias

                    # discard cond tokens that are entirely composed of zero values
                    if args.model.is_filter_cond_token:
                        cond_pixel_latents, cond_ids, cond_seq_lens, pad_seq_lens = FluxRegionalPipeline.filter_cond_token(
                            batch["cond_pixel_values"],
                            cond_pixel_latents,
                            cond_ids,
                            vae_scale_factor=vae_scale_factor * 2
                        )
                    else:
                        # When not filtering, all cond tokens are valid (no padding)
                        cond_seq_lens = [cond_pixel_latents.shape[1] for _ in range(bsz)]
                        pad_seq_lens = [0 for _ in range(bsz)]

                if args.trainer.offload:
                    # offload vae to CPU.
                    vae.cpu()

                noise = torch.randn_like(pixel_latents, device=accelerator.device, dtype=weight_dtype)
                # Sample a random timestep for each image
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.model.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.model.logit_mean,
                    logit_std=args.model.logit_std,
                    mode_scale=args.model.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=pixel_latents.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
                noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise

                # pack the latents.
                packed_noisy_model_input = FluxRegionalPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=bsz,
                    num_channels_latents=noisy_model_input.shape[1],
                    height=noisy_model_input.shape[2],
                    width=noisy_model_input.shape[3],
                )

                # latent image ids for RoPE.
                latent_image_ids = FluxRegionalPipeline._prepare_latent_image_ids(
                    bsz,
                    noisy_model_input.shape[2] // 2,
                    noisy_model_input.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )

                # handle guidance
                if unwrap_flux_transformer.config.guidance_embeds:
                    guidance_vec = torch.full(
                        (bsz,),
                        args.trainer.guidance_scale,
                        device=noisy_model_input.device,
                        dtype=weight_dtype,
                    )
                else:
                    guidance_vec = None

                # text encoding.
                text_encoding_pipeline = text_encoding_pipeline.to(accelerator.device)
                global_caption = batch["global_caption"]
                regional_captions = batch["regional_captions"]

                use_regional_control = args.model.attention_mask_method != "none"

                with torch.no_grad():
                    if use_regional_control:
                        # Use regional encoding with attention masks
                        (
                            prompt_embeds,
                            pooled_prompt_embeds,
                            txt_seq_lens,
                            text_ids,
                        ) = text_encoding_pipeline.encode_all_prompt(
                            global_prompt=global_caption,
                            regional_prompts=regional_captions,
                            global_max_sequence_length=args.model.max_sequence_length,
                            regional_max_sequence_length=args.model.regional_max_sequence_length,
                        )
                    else:
                        # Use standard encoding without regional prompts
                        (
                            prompt_embeds,
                            pooled_prompt_embeds,
                            text_ids,
                        ) = text_encoding_pipeline.encode_prompt(
                            prompt=global_caption,
                            prompt_2=None,
                            max_sequence_length=args.model.max_sequence_length,
                        )
                        txt_seq_lens = None

                if _diag_active(step):
                    _t = _diag_t()
                    _n_regions = [len(rc) for rc in batch["regional_captions"]]
                    print(f"[DIAG] step={step} text_encoded t=+{_t-_diag_t0_ref[0]:.2f}s "
                          f"txt_seq_lens={txt_seq_lens} cond_seq_lens={cond_seq_lens} n_regions={_n_regions}", flush=True)

                # prepare attention mask (only for regional control)
                joint_attention_kwargs = {}
                if use_regional_control:
                    # Only include condition tokens in attention mask for unified integration.
                    # In decoupled mode, condition tokens are processed in a separate attention
                    # path (cond_joint_attention) that doesn't use the regional attention mask.
                    if args.model.conditional_integration_method == "unified":
                        mask_cond_seq_lens = cond_seq_lens
                        mask_pad_seq_lens = pad_seq_lens
                    else:
                        mask_cond_seq_lens = [0] * len(cond_seq_lens)
                        mask_pad_seq_lens = [0] * len(pad_seq_lens)

                    attention_mask, hard_attention_mask = FluxRegionalPipeline.prepare_attention_mask(
                        attention_mask_method=args.model.attention_mask_method,
                        regional_labels=batch['label'],
                        txt_seq_lens=txt_seq_lens,
                        cond_seq_lens=mask_cond_seq_lens,
                        pad_seq_lens=mask_pad_seq_lens,
                        height=noisy_model_input.shape[2] // 2,
                        width=noisy_model_input.shape[3] // 2,
                        num_attention_heads=unwrap_flux_transformer.config.num_attention_heads,
                        dtype=weight_dtype,
                        device=accelerator.device,
                    )
                    joint_attention_kwargs["attention_mask"] = attention_mask
                    joint_attention_kwargs["hard_attention_mask"] = hard_attention_mask
                else:
                    joint_attention_kwargs["attention_mask"] = None
                    joint_attention_kwargs["hard_attention_mask"] = None

                if _diag_active(step):
                    _t = _diag_t()
                    _N = (hard_attention_mask.shape[-1]
                          if (use_regional_control and hard_attention_mask is not None) else -1)
                    print(f"[DIAG] step={step} mask_prepared t=+{_t-_diag_t0_ref[0]:.2f}s N={_N}", flush=True)

                if args.trainer.offload:
                    text_encoding_pipeline = text_encoding_pipeline.to("cpu")
                    torch.cuda.empty_cache()

                # Predict.
                model_pred = flux_transformer(
                    hidden_states=packed_noisy_model_input,
                    cond_hidden_states=cond_pixel_latents,
                    hard_attn_block_range=args.model.hard_attn_block_range,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    timestep=timesteps / 1000,
                    img_ids=latent_image_ids,
                    txt_ids=text_ids,
                    cond_ids=cond_ids,
                    guidance=guidance_vec,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False
                )[0]

                model_pred = FluxRegionalPipeline._unpack_latents(
                    model_pred,
                    height=noisy_model_input.shape[2] * vae_scale_factor,
                    width=noisy_model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.model.weighting_scheme, sigmas=sigmas)

                # flow-matching loss
                target = noise - pixel_latents
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()
                if _diag_active(step):
                    _t = _diag_t()
                    print(f"[DIAG] step={step} forward_done t=+{_t-_diag_t0_ref[0]:.2f}s", flush=True)
                accelerator.backward(loss)
                if _diag_active(step):
                    _t = _diag_t()
                    print(f"[DIAG] step={step} backward_done t=+{_t-_diag_t0_ref[0]:.2f}s", flush=True)

                if accelerator.sync_gradients:
                    params_to_clip = flux_transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.optimizer.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                if _diag_active(step):
                    _t = _diag_t()
                    print(f"[DIAG] step={step} optimizer_done t=+{_t-_diag_t0_ref[0]:.2f}s", flush=True)
                    _diag_prev_end_ref[0] = _t

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                    if global_step % args.trainer.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.trainer.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.project.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.trainer.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.trainer.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.project.output_dir, removing_checkpoint)
                                    try:
                                        shutil.rmtree(removing_checkpoint, ignore_errors=True)
                                    except Exception as e:
                                        logger.warning(f"Failed to remove checkpoint {removing_checkpoint}: {e}")

                        save_path = os.path.join(args.project.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path, exclude_frozen_parameters=True)
                        logger.info(f"Saved state to {save_path}")

                if accelerator.is_main_process:
                    if args.trainer.num_validation_images > 0 and global_step % args.trainer.validation_steps == 0:
                        image_logs = log_validation(
                            flux_transformer=flux_transformer,
                            vae=vae,
                            text_encoding_pipeline=text_encoding_pipeline,
                            args=args,
                            val_dataloader=val_dataloader,
                            accelerator=accelerator,
                            weight_dtype=weight_dtype,
                            step=global_step,
                        )

            loss_logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**loss_logs)
            accelerator.log(loss_logs, step=global_step)

            if use_adaptive_mask:
                adaptive_module = unwrap_flux_transformer.adaptive_mask_module
                num_layers = unwrap_flux_transformer.config.num_layers + unwrap_flux_transformer.config.num_single_layers
                variant = adaptive_mask_variant
                has_timestep = variant in ("full", "no_per_head", "timestep_only")
                has_layer = variant in ("full", "no_per_head", "layer_only")

                def _make_temb(t_val):
                    t_tensor = torch.tensor([t_val], device=accelerator.device, dtype=weight_dtype) * 1000
                    if unwrap_flux_transformer.config.guidance_embeds:
                        return unwrap_flux_transformer.time_text_embed(
                            t_tensor,
                            torch.tensor([args.trainer.guidance_scale], device=accelerator.device, dtype=weight_dtype),
                            pooled_prompt_embeds[:1],
                        )
                    return unwrap_flux_transformer.time_text_embed(t_tensor, pooled_prompt_embeds[:1])

                with torch.no_grad():
                    if variant == "scalar":
                        # Single learnable value
                        val = adaptive_module(_make_temb(0.5), 0).item()
                        accelerator.log({"adaptive_mask/strength": val}, step=global_step)

                    else:
                        ref_timesteps = [0.1, 0.5, 0.9] if has_timestep else [0.5]
                        for ref_t in ref_timesteps:
                            t_name = f"{ref_t:.1f}".replace(".", "_")
                            ref_temb = _make_temb(ref_t)
                            layer_range = range(num_layers) if has_layer else range(1)
                            all_strengths = []
                            for layer_idx in layer_range:
                                s = adaptive_module(ref_temb, layer_idx)
                                all_strengths.append(s[0])
                            strength_grid = torch.stack(all_strengths)  # (num_layers_or_1, num_heads_or_1)
                            logs = {f"adaptive_mask/mean_t{t_name}": strength_grid.mean().item()}
                            # Only log min/max when there is actual variation across the grid
                            if strength_grid.numel() > 1:
                                logs[f"adaptive_mask/min_t{t_name}"] = strength_grid.min().item()
                                logs[f"adaptive_mask/max_t{t_name}"] = strength_grid.max().item()
                            accelerator.log(logs, step=global_step)

                        # Log visualizations based on variant
                        if has_layer:
                            # Average over timesteps for a representative view
                            avg_timesteps = [0.1, 0.3, 0.5, 0.7, 0.9] if has_timestep else [0.5]
                            grids = []
                            for t_val in avg_timesteps:
                                ref_temb = _make_temb(t_val)
                                layer_strengths = []
                                for layer_idx in range(num_layers):
                                    s = adaptive_module(ref_temb, layer_idx)
                                    layer_strengths.append(s[0])
                                grids.append(torch.stack(layer_strengths))
                            strength_grid_np = torch.stack(grids).mean(0).cpu().float().numpy()

                            if strength_grid_np.shape[-1] > 1:
                                # Heatmap averaged over timesteps
                                grid_min, grid_max = strength_grid_np.min(), strength_grid_np.max()
                                if grid_max > grid_min:
                                    heatmap = (strength_grid_np - grid_min) / (grid_max - grid_min)
                                else:
                                    heatmap = np.zeros_like(strength_grid_np)
                                # Heatmap at t=0.5 only
                                t05_grid = grids[avg_timesteps.index(0.5)].cpu().float().numpy()
                                t05_min, t05_max = t05_grid.min(), t05_grid.max()
                                if t05_max > t05_min:
                                    heatmap_t05 = (t05_grid - t05_min) / (t05_max - t05_min)
                                else:
                                    heatmap_t05 = np.zeros_like(t05_grid)
                                for tracker in accelerator.trackers:
                                    if tracker.name == "tensorboard":
                                        tracker.writer.add_image(
                                            "adaptive_mask/strength_heatmap_avg",
                                            heatmap[np.newaxis, :, :],
                                            global_step=global_step,
                                        )
                                        tracker.writer.add_image(
                                            "adaptive_mask/strength_heatmap_t0_5",
                                            heatmap_t05[np.newaxis, :, :],
                                            global_step=global_step,
                                        )

                            # Log per-layer strengths as scalars
                            layer_logs = {
                                "adaptive_mask/strength_double_blocks": strength_grid_np[:19].mean(),
                                "adaptive_mask/strength_single_blocks": strength_grid_np[19:].mean(),
                            }
                            # Log individual layer strengths (mean across heads if per-head)
                            for li in range(num_layers):
                                layer_logs[f"adaptive_mask/layer_{li:02d}"] = float(strength_grid_np[li].mean())
                            accelerator.log(layer_logs, step=global_step)

            if getattr(args.model, 'zero_init_cond2img', False):
                scales = [
                    p.abs().mean()
                    for n, p in unwrap_flux_transformer.named_parameters()
                    if 'cond2img_scale' in n
                ]
                if scales:
                    accelerator.log({"params/cond2img_scale_mean": torch.stack(scales).mean().item()}, step=global_step)

            if global_step >= args.trainer.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    unknown = [s.lstrip('-') for s in unknown]
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    print('###### cli input training setup:  ######\n', cli)
    config = OmegaConf.merge(*configs, cli)

    if config.resolution % (16 * config.cond_scale_factor) != 0:
        raise ValueError(
            f"Image resolution {config.resolution} must be divisible by {16 * config.cond_scale_factor} "
            f"(16 * cond_scale_factor) to ensure proper feature map alignment in the model. "
            f"Please adjust either the resolution or cond_scale_factor."
        )

    main(config)
