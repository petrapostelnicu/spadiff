# Copyright 2024 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
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
# limitations under the License.


from typing import Any, Dict, List, Optional, Tuple, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FluxTransformer2DLoadersMixin, FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import (
    Attention,
    AttentionProcessor,
    FusedFluxAttnProcessor2_0,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.embeddings import CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings, \
    FluxPosEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from .adaptive_mask import AdaptiveMaskModule
from .attention import cond_joint_attention, FluxRegionalAttnProcessor2_0, AdaptiveMaskSpec
from multi_conditioning.utils.lora_controller import select_lora, disable_lora

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@maybe_allow_in_graph
class FluxSingleTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, conditional_integration_method, zero_init_cond2img=False, mlp_ratio=4.0):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.conditional_integration_method = conditional_integration_method
        # Zero-init per-channel scale for condition-image residual (decoupled mode only).
        # Initialized to zero so training starts from pretrained FLUX behavior.

        self.zero_init_cond2img = zero_init_cond2img and conditional_integration_method == "decoupled"
        # cond2img_scale is NOT created here. It is added post-load in train.py.

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)
        processor = FluxRegionalAttnProcessor2_0()
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True,
        )

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            temb: torch.FloatTensor,
            temb_cond: torch.FloatTensor,
            image_rotary_emb=None,
            attention_mask=None,
            cond_hidden_states=None,
            cond_rotary_emb=None,
            cond2image_attention_weight: float = 1.0,
            omini_cond_hidden_states_list=None,
            omini_cond_rotary_embs=None,
            omini_adapter_names=None,
    ):
        is_use_cond = cond_hidden_states is not None
        is_use_omini = omini_cond_hidden_states_list is not None and len(omini_cond_hidden_states_list) > 0

        residual = hidden_states

        # Disable LoRA for image/text tokens (base model behavior)
        with disable_lora((self.norm.linear, self.proj_mlp)):
            norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
            mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        if is_use_cond:
            residual_cond = cond_hidden_states
            with select_lora((self.norm.linear, self.proj_mlp), 'cond'):
                norm_cond_hidden_states, cond_gate = self.norm(cond_hidden_states, emb=temb_cond)
                mlp_cond_hidden_states = self.act_mlp(self.proj_mlp(norm_cond_hidden_states))

        # Norm each omini condition with its own LoRA adapter
        omini_norms = []  # list of (norm_hs, gate, mlp_hs, residual) per condition
        if is_use_omini:
            for cond_i, omini_hs in enumerate(omini_cond_hidden_states_list):
                with select_lora((self.norm.linear, self.proj_mlp), omini_adapter_names[cond_i]):
                    norm_omini, omini_gate = self.norm(omini_hs, emb=temb_cond)
                    mlp_omini = self.act_mlp(self.proj_mlp(norm_omini))
                omini_norms.append((norm_omini, omini_gate, mlp_omini, omini_hs))

        # omini always uses unified path
        attn_cond = norm_cond_hidden_states if (is_use_cond and self.conditional_integration_method == "unified") else None
        result = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            cond_hidden_states=attn_cond,
            cond_rotary_emb=cond_rotary_emb if attn_cond is not None else None,
            attention_mask=attention_mask,
            omini_cond_hidden_states_list=[n[0] for n in omini_norms] if is_use_omini else None,
            omini_cond_rotary_embs=omini_cond_rotary_embs,
            omini_adapter_names=omini_adapter_names,
        )
        if not isinstance(result, tuple):
            result = (result,)
        attn_output = result[0]

        # Unpack attention results by tracking index
        idx = 1
        cond_attn_output = None
        if is_use_cond:
            if self.conditional_integration_method == "unified":
                cond_attn_output = result[idx]; idx += 1
            else:
                # Decoupled: separate joint attention for seg cond (img+cond Q/K/V concatenated)
                cond_result = cond_joint_attention(
                    attn=self.attn,
                    hidden_states=norm_hidden_states,
                    cond_hidden_states=norm_cond_hidden_states,
                    image_rotary_emb=image_rotary_emb,
                    cond_rotary_emb=cond_rotary_emb,
                )
                attn_output2, cond_attn_output = cond_result[:2]
                # Scale condition contribution by cond2image_attention_weight.
                if self.zero_init_cond2img and hasattr(self, 'cond2img_scale'):
                    attn_output = attn_output + (self.cond2img_scale * attn_output2) * cond2image_attention_weight
                else:
                    attn_output = attn_output + attn_output2 * cond2image_attention_weight

        # Unpack omini attention outputs (one per condition)
        omini_attn_outputs = []
        if is_use_omini:
            for _ in omini_norms:
                omini_attn_outputs.append(result[idx]); idx += 1

        # Disable LoRA for image/text tokens output projection
        with disable_lora((self.proj_out,)):
            hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
            gate = gate.unsqueeze(1)
            hidden_states = gate * self.proj_out(hidden_states)
            hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        if is_use_cond:
            with select_lora((self.proj_out,), 'cond'):
                cond_hidden_states = torch.cat([cond_attn_output, mlp_cond_hidden_states], dim=2)
                cond_gate = cond_gate.unsqueeze(1)
                cond_hidden_states = cond_gate * self.proj_out(cond_hidden_states)
                cond_hidden_states = residual_cond + cond_hidden_states

        # Apply residual for each omini condition with its own LoRA adapter
        omini_cond_hidden_states_list_out = []
        if is_use_omini:
            for cond_i, (norm_omini, omini_gate, mlp_omini, residual_omini) in enumerate(omini_norms):
                with select_lora((self.proj_out,), omini_adapter_names[cond_i]):
                    omini_out = torch.cat([omini_attn_outputs[cond_i], mlp_omini], dim=2)
                    omini_gate = omini_gate.unsqueeze(1)
                    omini_out = omini_gate * self.proj_out(omini_out)
                    omini_out = residual_omini + omini_out
                omini_cond_hidden_states_list_out.append(omini_out)

        return hidden_states, cond_hidden_states, omini_cond_hidden_states_list_out


@maybe_allow_in_graph
class FluxTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, conditional_integration_method, zero_init_cond2img=False, qk_norm="rms_norm", eps=1e-6):
        super().__init__()

        self.conditional_integration_method = conditional_integration_method
        # Zero-init per-channel scale for condition→image residual (decoupled mode only).
        # Initialized to zero so training starts from pretrained FLUX behavior.
        self.zero_init_cond2img = zero_init_cond2img and conditional_integration_method == "decoupled"
        # cond2img_scale is NOT created here — see FluxSingleTransformerBlock for explanation.

        self.norm1 = AdaLayerNormZero(dim)

        self.norm1_context = AdaLayerNormZero(dim)

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )
        processor = FluxRegionalAttnProcessor2_0()
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=eps,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor,
            temb: torch.FloatTensor,
            temb_cond: torch.FloatTensor,
            image_rotary_emb=None,
            attention_mask=None,
            cond_hidden_states=None,
            cond_rotary_emb=None,
            cond2image_attention_weight: float = 1.0,
            omini_cond_hidden_states_list=None,
            omini_cond_rotary_embs=None,
            omini_adapter_names=None,
    ):
        is_use_cond = cond_hidden_states is not None
        is_use_omini = omini_cond_hidden_states_list is not None and len(omini_cond_hidden_states_list) > 0

        # Disable LoRA for image tokens (base model behavior)
        with disable_lora((self.norm1.linear,)):
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        if is_use_cond:
            with select_lora((self.norm1.linear,), 'cond'):
                # load cond lora for condition token
                norm_cond_hidden_states, cond_gate_msa, cond_shift_mlp, cond_scale_mlp, cond_gate_mlp = self.norm1(
                    cond_hidden_states, emb=temb_cond)

        omini_norms = []
        if is_use_omini:
            for cond_i, omini_hs in enumerate(omini_cond_hidden_states_list):
                with select_lora((self.norm1.linear,), omini_adapter_names[cond_i]):
                    o_norm, o_gate_msa, o_shift_mlp, o_scale_mlp, o_gate_mlp = self.norm1(omini_hs, emb=temb_cond)
                omini_norms.append((o_norm, o_gate_msa, o_shift_mlp, o_scale_mlp, o_gate_mlp, omini_hs))

        attn_cond = norm_cond_hidden_states if (is_use_cond and self.conditional_integration_method == "unified") else None
        result = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            cond_hidden_states=attn_cond,
            image_rotary_emb=image_rotary_emb,
            cond_rotary_emb=cond_rotary_emb if attn_cond is not None else None,
            attention_mask=attention_mask,
            omini_cond_hidden_states_list=[n[0] for n in omini_norms] if is_use_omini else None,
            omini_cond_rotary_embs=omini_cond_rotary_embs,
            omini_adapter_names=omini_adapter_names,
        )
        attn_output, context_attn_output = result[:2]

        # Unpack attention results by tracking index
        idx = 2
        cond_attn_output = None
        if is_use_cond:
            if self.conditional_integration_method == "unified":
                cond_attn_output = result[idx]; idx += 1
            else:
                # Decoupled: separate joint attention for seg cond (img+cond Q/K/V concatenated)
                txt_seq_len = encoder_hidden_states.shape[1]
                img_rotary_emb = (image_rotary_emb[0][txt_seq_len:], image_rotary_emb[1][txt_seq_len:])
                cond_result = cond_joint_attention(
                    attn=self.attn,
                    hidden_states=norm_hidden_states,
                    cond_hidden_states=norm_cond_hidden_states,
                    image_rotary_emb=img_rotary_emb,
                    cond_rotary_emb=cond_rotary_emb,
                )
                attn_output2, cond_attn_output = cond_result[:2]
                # Scale condition contribution by cond2image_attention_weight.
                if self.zero_init_cond2img and hasattr(self, 'cond2img_scale'):
                    attn_output = attn_output + (self.cond2img_scale * attn_output2) * cond2image_attention_weight
                else:
                    attn_output = attn_output + attn_output2 * cond2image_attention_weight

        # Unpack omini attention outputs (one per condition)
        omini_attn_outputs = []
        if is_use_omini:
            for _ in omini_norms:
                omini_attn_outputs.append(result[idx]); idx += 1

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        # Disable LoRA for image tokens feedforward
        with disable_lora((self.ff.net[0].proj, self.ff.net[2],)):
            ff_output = self.ff(norm_hidden_states)
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        if is_use_cond:
            cond_attn_output = cond_gate_msa.unsqueeze(1) * cond_attn_output
            cond_hidden_states = cond_hidden_states + cond_attn_output
            norm_cond_hidden_states = self.norm2(cond_hidden_states)
            norm_cond_hidden_states = norm_cond_hidden_states * (1 + cond_scale_mlp[:, None]) + cond_shift_mlp[:, None]

            with select_lora((self.ff.net[0].proj, self.ff.net[2],), 'cond'):
                cond_ff_output = self.ff(norm_cond_hidden_states)
                cond_ff_output = cond_gate_mlp.unsqueeze(1) * cond_ff_output
            cond_hidden_states = cond_hidden_states + cond_ff_output

        # Process each omini condition
        omini_cond_hidden_states_list_out = []
        if is_use_omini:
            for cond_i, (o_norm, o_gate_msa, o_shift_mlp, o_scale_mlp, o_gate_mlp, residual_omini) in enumerate(omini_norms):
                omini_attn = o_gate_msa.unsqueeze(1) * omini_attn_outputs[cond_i]
                omini_hs = residual_omini + omini_attn
                norm_omini = self.norm2(omini_hs)
                norm_omini = norm_omini * (1 + o_scale_mlp[:, None]) + o_shift_mlp[:, None]

                with select_lora((self.ff.net[0].proj, self.ff.net[2],), omini_adapter_names[cond_i]):
                    omini_ff = self.ff(norm_omini)
                    omini_ff = o_gate_mlp.unsqueeze(1) * omini_ff
                omini_hs = omini_hs + omini_ff
                omini_cond_hidden_states_list_out.append(omini_hs)

        return encoder_hidden_states, hidden_states, cond_hidden_states, omini_cond_hidden_states_list_out


class FluxTransformer2DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, FluxTransformer2DLoadersMixin
):
    """
    The Transformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Parameters:
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of layers of MMDiT blocks to use.
        num_single_layers (`int`, *optional*, defaults to 18): The number of layers of single DiT blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        joint_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        pooled_projection_dim (`int`): Number of dimensions to use when projecting the `pooled_projections`.
        guidance_embeds (`bool`, defaults to False): Whether to use guidance embeddings.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]

    @register_to_config
    def __init__(
            self,
            patch_size: int = 1,
            in_channels: int = 64,
            out_channels: Optional[int] = None,
            num_layers: int = 19,
            num_single_layers: int = 38,
            attention_head_dim: int = 128,
            num_attention_heads: int = 24,
            joint_attention_dim: int = 4096,
            pooled_projection_dim: int = 768,
            guidance_embeds: bool = False,
            axes_dims_rope: Tuple[int] = (16, 56, 56),
            conditional_integration_method: str = "unified",
            zero_init_cond2img: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        )

        self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.inner_dim)

        self.x_embedder = nn.Linear(self.config.in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    conditional_integration_method=conditional_integration_method,
                    zero_init_cond2img=zero_init_cond2img,
                )
                for i in range(self.config.num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    conditional_integration_method=conditional_integration_method,
                    zero_init_cond2img=zero_init_cond2img,
                )
                for i in range(self.config.num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedFluxAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedFluxAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
            self,
            hidden_states: torch.Tensor,
            cond_hidden_states: torch.Tensor,
            hard_attn_block_range=[19, 37],
            encoder_hidden_states: torch.Tensor = None,
            pooled_projections: torch.Tensor = None,
            timestep: torch.LongTensor = None,
            img_ids: torch.Tensor = None,
            txt_ids: torch.Tensor = None,
            cond_ids: torch.Tensor = None,
            guidance: torch.Tensor = None,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            controlnet_block_samples=None,
            controlnet_single_block_samples=None,
            controlnet_blocks_repeat: bool = False,
            return_dict: bool = True,
            omini_cond_hidden_states_list: List[torch.Tensor] = None,
            omini_cond_ids_list: List[torch.Tensor] = None,
            omini_adapter_names: List[str] = None,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:

        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )
        # Disable LoRA for image tokens embedding
        with disable_lora((self.x_embedder,)):
            hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )

        # Condition timestep embedding (timestep=0 for all conditions)
        is_use_omini = omini_cond_hidden_states_list is not None and len(omini_cond_hidden_states_list) > 0
        need_temb_cond = cond_hidden_states is not None or is_use_omini
        if need_temb_cond:
            temb_cond = (
                self.time_text_embed(torch.zeros_like(timestep), pooled_projections)
                if guidance is None
                else self.time_text_embed(
                    torch.zeros_like(timestep), guidance, pooled_projections
                )
            )
        else:
            temb_cond = None

        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # prepare joint_attention_kwargs for regional control
        joint_attention_kwargs = joint_attention_kwargs or {}

        is_use_hard_mask = joint_attention_kwargs.get('hard_attention_mask', None) is not None
        is_use_adaptive_mask = hasattr(self, 'adaptive_mask_module') and self.adaptive_mask_module is not None
        cond2image_attention_weight = joint_attention_kwargs.get('cond2image_attention_weight', 1.0)
        cross_control_mask = joint_attention_kwargs.get('cross_control_mask', None)

        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        # Seg condition RoPE
        cond_rotary_emb = None
        if cond_hidden_states is not None:
            if cond_ids.ndim == 3:
                cos_list, sin_list = [], []
                for i in range(cond_ids.shape[0]):
                    cos, sin = self.pos_embed(cond_ids[i])
                    cos_list.append(cos)
                    sin_list.append(sin)
                cond_rotary_emb = (torch.stack(cos_list, dim=0), torch.stack(sin_list, dim=0))
            else:
                cond_rotary_emb = self.pos_embed(cond_ids)

        # OminiControl condition RoPE
        omini_cond_rotary_embs = None
        if is_use_omini:
            omini_cond_rotary_embs = []
            for omini_ids in omini_cond_ids_list:
                if omini_ids.ndim == 3:
                    cos_list, sin_list = [], []
                    for i in range(omini_ids.shape[0]):
                        cos, sin = self.pos_embed(omini_ids[i])
                        cos_list.append(cos)
                        sin_list.append(sin)
                    omini_cond_rotary_embs.append((torch.stack(cos_list, dim=0), torch.stack(sin_list, dim=0)))
                else:
                    omini_cond_rotary_embs.append(self.pos_embed(omini_ids))

        # Embed conditions through x_embedder with respective LoRA
        if cond_hidden_states is not None:
            with select_lora((self.x_embedder,), 'cond'):
                cond_hidden_states = self.x_embedder(cond_hidden_states)

        # Embed each omini condition with its own LoRA adapter
        if is_use_omini:
            omini_cond_hidden_states_list = list(omini_cond_hidden_states_list)
            for cond_i in range(len(omini_cond_hidden_states_list)):
                with select_lora((self.x_embedder,), omini_adapter_names[cond_i]):
                    omini_cond_hidden_states_list[cond_i] = self.x_embedder(omini_cond_hidden_states_list[cond_i])

        adaptive_blocked = None
        adaptive_soft_bias = None
        adaptive_cross_ctrl = None
        if is_use_adaptive_mask and is_use_hard_mask:
            hm = joint_attention_kwargs['hard_attention_mask']
            adaptive_blocked = hm.isinf()[:, 0].contiguous()
            adaptive_soft_bias = hm[:, 0].contiguous()
            adaptive_soft_bias = torch.where(
                adaptive_soft_bias.isinf(),
                torch.zeros((), dtype=adaptive_soft_bias.dtype, device=adaptive_soft_bias.device),
                adaptive_soft_bias,
            )
            # Cross-control: stream-vs-stream cells that must stay hard -inf even
            # under AMM softening
            if cross_control_mask is not None:
                adaptive_cross_ctrl = cross_control_mask.isinf()[:, 0].contiguous()

        # ---- Double transformer blocks ----
        for index_block, block in enumerate(self.transformer_blocks):

            if is_use_adaptive_mask and is_use_hard_mask:
                strengths = self.adaptive_mask_module(temb, index_block)
                attention_mask = AdaptiveMaskSpec(
                    blocked=adaptive_blocked,
                    strengths=strengths,
                    soft_bias=adaptive_soft_bias,
                    cross_control=adaptive_cross_ctrl,
                )
            elif is_use_hard_mask and \
                    index_block <= hard_attn_block_range[1] and \
                    index_block >= hard_attn_block_range[0]:
                attention_mask = joint_attention_kwargs['hard_attention_mask']
            else:
                attention_mask = joint_attention_kwargs.get('attention_mask', None)

            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states, cond_hidden_states, omini_cond_hidden_states_list = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    temb_cond,
                    image_rotary_emb,
                    attention_mask,
                    cond_hidden_states,
                    cond_rotary_emb,
                    cond2image_attention_weight,
                    omini_cond_hidden_states_list,
                    omini_cond_rotary_embs,
                    omini_adapter_names,
                    **ckpt_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states, cond_hidden_states, omini_cond_hidden_states_list = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    temb_cond=temb_cond,
                    image_rotary_emb=image_rotary_emb,
                    attention_mask=attention_mask,
                    cond_hidden_states=cond_hidden_states,
                    cond_rotary_emb=cond_rotary_emb,
                    cond2image_attention_weight=cond2image_attention_weight,
                    omini_cond_hidden_states_list=omini_cond_hidden_states_list,
                    omini_cond_rotary_embs=omini_cond_rotary_embs,
                    omini_adapter_names=omini_adapter_names,
                )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                if controlnet_blocks_repeat:
                    hidden_states = (
                            hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # ---- Single transformer blocks ----
        for index_block, block in enumerate(self.single_transformer_blocks):
            layer_idx = index_block + self.config.num_layers

            if is_use_adaptive_mask and is_use_hard_mask:
                strengths = self.adaptive_mask_module(temb, layer_idx)
                attention_mask = AdaptiveMaskSpec(
                    blocked=adaptive_blocked,
                    strengths=strengths,
                    soft_bias=adaptive_soft_bias,
                    cross_control=adaptive_cross_ctrl,
                )
            elif is_use_hard_mask and \
                    layer_idx <= hard_attn_block_range[1] and \
                    layer_idx >= hard_attn_block_range[0]:
                attention_mask = joint_attention_kwargs['hard_attention_mask']
            else:
                attention_mask = joint_attention_kwargs.get('attention_mask', None)

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, cond_hidden_states, omini_cond_hidden_states_list = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    temb,
                    temb_cond,
                    image_rotary_emb,
                    attention_mask,
                    cond_hidden_states,
                    cond_rotary_emb,
                    cond2image_attention_weight,
                    omini_cond_hidden_states_list,
                    omini_cond_rotary_embs,
                    omini_adapter_names,
                    **ckpt_kwargs,
                )

            else:
                hidden_states, cond_hidden_states, omini_cond_hidden_states_list = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    temb_cond=temb_cond,
                    image_rotary_emb=image_rotary_emb,
                    attention_mask=attention_mask,
                    cond_hidden_states=cond_hidden_states,
                    cond_rotary_emb=cond_rotary_emb,
                    cond2image_attention_weight=cond2image_attention_weight,
                    omini_cond_hidden_states_list=omini_cond_hidden_states_list,
                    omini_cond_rotary_embs=omini_cond_rotary_embs,
                    omini_adapter_names=omini_adapter_names,
                )

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1]:, ...] = (
                        hidden_states[:, encoder_hidden_states.shape[1]:, ...]
                        + controlnet_single_block_samples[index_block // interval_control]
                )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)