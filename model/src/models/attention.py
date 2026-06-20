import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union, Any, Dict
import torch.nn.functional as F
import torch

from model.utils.lora_controller import select_lora, disable_lora

try:
    from torch.nn.attention.flex_attention import flex_attention as _flex_attention_raw
    import torch._dynamo as _dynamo
    _dynamo.config.cache_size_limit = max(_dynamo.config.cache_size_limit, 256)
    # FLEX_COMPILE=0 disables torch.compile and falls back to FlexAttention's
    # eager path.
    if os.environ.get("FLEX_COMPILE", "1") == "0":
        _flex_attention = _flex_attention_raw
    else:
        _flex_attention = torch.compile(_flex_attention_raw, dynamic=False)
    _FLEX_AVAILABLE = True
except ImportError:
    _flex_attention = None
    _FLEX_AVAILABLE = False


def _flex_kernel_options():
    """Pick FlexAttention kernel block sizes based on GPU shared-memory budget.

    Returns None on big-SMEM GPUs to let the autotuner pick optimal sizes.
    """
    if not torch.cuda.is_available():
        return None
    try:
        smem_per_block = torch.cuda.get_device_properties(0).shared_memory_per_block
    except AttributeError:
        # Older torch builds may not expose this. Be conservative.
        return {
            "BLOCK_M": 64, "BLOCK_N": 64,
            "BLOCK_M1": 32, "BLOCK_N1": 32,
            "BLOCK_M2": 32, "BLOCK_N2": 32,
        }
    # 130 KB threshold: above => let autotuner pick (A100/H100). Below => cap.
    if smem_per_block < 130 * 1024:
        return {
            "BLOCK_M": 64, "BLOCK_N": 64,
            "BLOCK_M1": 32, "BLOCK_N1": 32,
            "BLOCK_M2": 32, "BLOCK_N2": 32,
        }
    return None


_FLEX_KERNEL_OPTIONS = _flex_kernel_options()


@dataclass
class AdaptiveMaskSpec:
    """Lazy form of the adaptive attention mask, consumed by FlexAttention.

    Attributes:
        blocked: (B, N, N) bool. Hard-blocking pattern is identical across
                 heads, so head dim is deduplicated
                 and broadcast inside score_mod.
        strengths: (B, H) float. Bias at blocked cells is -strengths[b, h].
        soft_bias: optional float tensor of shape (B, N, N) carrying the
                   log-space additive bias to apply at NON-blocked cells.
                   When None, non-blocked cells get a 0 bias. When provided,
                   non-blocked cells get the per-cell value from this tensor.
    """
    blocked: torch.Tensor
    strengths: torch.Tensor
    soft_bias: Optional[torch.Tensor] = None


_USE_FLEX = os.environ.get("USE_FLEX", "1") == "1" and _FLEX_AVAILABLE


def _flex_aligned(query, key, value, spec: AdaptiveMaskSpec):
    """Adaptive masking via FlexAttention (USE_FLEX=1) or SDPA-math (USE_FLEX=0)."""
    blocked = spec.blocked
    strengths = spec.strengths
    soft_bias = spec.soft_bias

    if not _USE_FLEX:
        # SDPA math path: build the bias once, dispatch to SDPA.
        if soft_bias is None:
            non_blocked_value = torch.zeros((), dtype=query.dtype, device=query.device)
        else:
            non_blocked_value = soft_bias[:, None, :, :].to(query.dtype)
        bias = torch.where(
            blocked[:, None, :, :],
            -strengths[:, :, None, None].to(query.dtype),
            non_blocked_value,
        )
        # Force math backend
        prev_flash = torch.backends.cuda.flash_sdp_enabled()
        prev_mem = torch.backends.cuda.mem_efficient_sdp_enabled()
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            return F.scaled_dot_product_attention(
                query, key, value, attn_mask=bias,
                dropout_p=0.0, is_causal=False,
            )
        finally:
            torch.backends.cuda.enable_flash_sdp(prev_flash)
            torch.backends.cuda.enable_mem_efficient_sdp(prev_mem)

    # FlexAttention path.
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    if soft_bias is None:
        def score_mod(score, b, h, q_idx, kv_idx):
            is_blocked = blocked[b, q_idx, kv_idx]
            return torch.where(
                is_blocked,
                score - strengths[b, h].to(score.dtype),
                score,
            )
    else:
        def score_mod(score, b, h, q_idx, kv_idx):
            is_blocked = blocked[b, q_idx, kv_idx]
            return torch.where(
                is_blocked,
                score - strengths[b, h].to(score.dtype),
                score + soft_bias[b, q_idx, kv_idx].to(score.dtype),
            )

    if _FLEX_KERNEL_OPTIONS is not None:
        return _flex_attention(
            query, key, value, score_mod=score_mod,
            kernel_options=_FLEX_KERNEL_OPTIONS,
        )
    return _flex_attention(query, key, value, score_mod=score_mod)


def _sdpa_aligned(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
    """SDPA dispatcher.

    For AdaptiveMaskSpec inputs, route to FlexAttention via _flex_aligned.
    For tensor masks with requires_grad, force the SDPA math backend.
    Otherwise pass through to standard SDPA (flash backend if mask is None).
    """
    if isinstance(attn_mask, AdaptiveMaskSpec):
        if not _FLEX_AVAILABLE:
            raise RuntimeError(
                "AdaptiveMaskSpec requires FlexAttention (torch >= 2.5). "
                f"Got torch={torch.__version__}"
            )
        return _flex_aligned(query, key, value, attn_mask)

    if attn_mask is not None and torch.is_tensor(attn_mask) and attn_mask.requires_grad:
        prev_flash = torch.backends.cuda.flash_sdp_enabled()
        prev_mem = torch.backends.cuda.mem_efficient_sdp_enabled()
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            return F.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask,
                dropout_p=dropout_p, is_causal=is_causal,
            )
        finally:
            torch.backends.cuda.enable_flash_sdp(prev_flash)
            torch.backends.cuda.enable_mem_efficient_sdp(prev_mem)

    return F.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask,
        dropout_p=dropout_p, is_causal=is_causal,
    )


def apply_rotary_emb(
        x: torch.Tensor,
        freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos, sin = freqs_cis
    if cos.ndim == 2:
        # [S, D] -> [B, H, S, D]
        cos = cos[None, None]
        sin = sin[None, None]
    elif cos.ndim == 3:
        # [B, S, D] -> [B, H, S, D]
        cos = cos.unsqueeze(dim=1)
        sin = sin.unsqueeze(dim=1)

    cos, sin = cos.to(x.device), sin.to(x.device)

    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, H, S, D//2]
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)

    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

    return out


class FluxRegionalAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
            self,
            attn,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor = None,
            cond_hidden_states: torch.FloatTensor = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
            cond_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:

        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # Disable LoRA for image/text tokens (base model behavior)
        with disable_lora((attn.to_q, attn.to_k, attn.to_v)):
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        if cond_hidden_states is not None:
            with select_lora((attn.to_q, attn.to_k, attn.to_v), 'cond'):
                # load default lora for condition token
                cond_query = attn.to_q(cond_hidden_states)
                cond_key = attn.to_k(cond_hidden_states)
                cond_value = attn.to_v(cond_hidden_states)

            cond_query = cond_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            cond_key = cond_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            cond_value = cond_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_q is not None:
                cond_query = attn.norm_q(cond_query)
            if attn.norm_k is not None:
                cond_key = attn.norm_k(cond_key)
            if cond_rotary_emb is not None:
                cond_query = apply_rotary_emb(cond_query, cond_rotary_emb)
                cond_key = apply_rotary_emb(cond_key, cond_rotary_emb)
            query = torch.cat([query, cond_query], dim=2)
            key = torch.cat([key, cond_key], dim=2)
            value = torch.cat([value, cond_value], dim=2)

        hidden_states = _sdpa_aligned(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            if cond_hidden_states is not None:
                encoder_hidden_states, hidden_states, cond_hidden_states = (
                    hidden_states[:, : encoder_hidden_states.shape[1]],
                    hidden_states[:, encoder_hidden_states.shape[1]: -cond_hidden_states.shape[1]],
                    hidden_states[:, -cond_hidden_states.shape[1]:],
                )
                with select_lora((attn.to_out[0],), 'cond'):
                    # linear proj
                    cond_hidden_states = attn.to_out[0](cond_hidden_states)
                    # dropout
                    cond_hidden_states = attn.to_out[1](cond_hidden_states)
            else:
                encoder_hidden_states, hidden_states = (
                    hidden_states[:, : encoder_hidden_states.shape[1]],
                    hidden_states[:, encoder_hidden_states.shape[1]:],
                )

            # Disable LoRA for image tokens output projection
            with disable_lora((attn.to_out[0],)):
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return (
                (hidden_states, encoder_hidden_states, cond_hidden_states)
                if cond_hidden_states is not None else (hidden_states, encoder_hidden_states)
            )
        else:
            if cond_hidden_states is not None:
                hidden_states, cond_hidden_states = (
                    hidden_states[:, : -cond_hidden_states.shape[1]],
                    hidden_states[:, -cond_hidden_states.shape[1]:],
                )
                return (hidden_states, cond_hidden_states)
            else:
                return (hidden_states,)


def cond_joint_attention(
        attn,
        hidden_states: torch.FloatTensor,
        cond_hidden_states: torch.FloatTensor,
        image_rotary_emb: Optional[torch.Tensor] = None,
        cond_rotary_emb: Optional[torch.Tensor] = None,
) -> torch.FloatTensor:
    """
    Perform joint attention between hidden_states (image) and cond_hidden_states (condition).
    Both produce Q/K/V which are concatenated before a single SDPA call, then split back.
    Used in decoupled mode where cond tokens have a separate attention path from text tokens.

    - Query/Key/Value for image tokens use base projections (LoRA disabled)
    - Query/Key/Value for condition tokens use projections with 'cond' LoRA
    - Output projection for image uses base (LoRA disabled), for cond uses 'cond' LoRA
    """
    batch_size = hidden_states.shape[0]

    # Query/Key/Value from hidden_states (image tokens) - disable LoRA
    with disable_lora((attn.to_q, attn.to_k, attn.to_v)):
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    if image_rotary_emb is not None:
        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)

    # Key/Value from cond_hidden_states with 'cond' LoRA
    if cond_hidden_states is not None:
        with select_lora((attn.to_q, attn.to_k, attn.to_v), 'cond'):
            # load default lora for condition token
            cond_query = attn.to_q(cond_hidden_states)
            cond_key = attn.to_k(cond_hidden_states)
            cond_value = attn.to_v(cond_hidden_states)

        cond_query = cond_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        cond_key = cond_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        cond_value = cond_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            cond_query = attn.norm_q(cond_query)
        if attn.norm_k is not None:
            cond_key = attn.norm_k(cond_key)
        if cond_rotary_emb is not None:
            cond_query = apply_rotary_emb(cond_query, cond_rotary_emb)
            cond_key = apply_rotary_emb(cond_key, cond_rotary_emb)
        query = torch.cat([query, cond_query], dim=2)
        key = torch.cat([key, cond_key], dim=2)
        value = torch.cat([value, cond_value], dim=2)

    # Joint attention
    hidden_states = _sdpa_aligned(
        query, key, value, dropout_p=0.0, is_causal=False
    )

    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)

    if cond_hidden_states is not None:
        hidden_states, cond_hidden_states = (
                            hidden_states[:, : -cond_hidden_states.shape[1]],
                            hidden_states[:, -cond_hidden_states.shape[1]:],
                        )
        if attn.to_out is not None:
            # Condition tokens: use LoRA
            with select_lora((attn.to_out[0],), 'cond'):
                cond_hidden_states = attn.to_out[0](cond_hidden_states)
                cond_hidden_states = attn.to_out[1](cond_hidden_states)

            # Image tokens: disable LoRA (base model behavior)
            with disable_lora((attn.to_out[0],)):
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)
        return (hidden_states, cond_hidden_states)
    else:
        if attn.to_out is not None:
            # No condition tokens: disable LoRA for image tokens
            with disable_lora((attn.to_out[0],)):
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)
        return (hidden_states,)