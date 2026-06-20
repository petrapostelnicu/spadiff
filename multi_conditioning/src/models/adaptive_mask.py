import torch
import torch.nn as nn


class AdaptiveMaskModule(nn.Module):
    """Learns attention mask strengths that replace hard -inf blocking.

    Supports several ablation variants controlled by the `variant` parameter:
        - "full": timestep + layer conditioning, per-head output (default)
        - "no_per_head": timestep + layer conditioning, scalar output
        - "timestep_only": timestep conditioning only, scalar output
        - "layer_only": layer conditioning only, scalar output
        - "scalar": single learnable parameter, no conditioning
    """

    def __init__(self, temb_dim=3072, num_layers=57, num_heads=24, hidden_dim=256, variant="full"):
        super().__init__()
        self.variant = variant
        output_dim = num_heads if variant == "full" else 1

        if variant == "scalar":
            self.raw_strength = nn.Parameter(torch.full((1,), 7.0))
            self.softplus = nn.Softplus()
            return

        use_layer = variant in ("full", "no_per_head", "layer_only")
        use_temb = variant in ("full", "no_per_head", "timestep_only")

        if use_layer:
            self.layer_embed = nn.Embedding(num_layers, hidden_dim)

        input_dim = 0
        if use_temb:
            input_dim += temb_dim
        if use_layer:
            input_dim += hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus(),
        )
        # Zero-init final linear weights and set bias so that softplus outputs ~7
        # (strong blocking), so training starts near hard-mask behavior.
        with torch.no_grad():
            self.mlp[2].weight.zero_()
            self.mlp[2].bias.fill_(7.0)

    def forward(self, temb, layer_idx):
        """
        Args:
            temb: (batch, temb_dim) - timestep embedding
            layer_idx: int - transformer block index (0..num_layers-1)

        Returns:
            strengths: (batch, num_heads) or (batch, 1) - positive mask strengths
        """
        if self.variant == "scalar":
            return self.softplus(self.raw_strength).expand(temb.shape[0], -1)

        parts = []
        if self.variant in ("full", "no_per_head", "timestep_only"):
            parts.append(temb)
        if self.variant in ("full", "no_per_head", "layer_only"):
            layer_emb = self.layer_embed(
                torch.tensor(layer_idx, device=temb.device)
            )
            layer_emb = layer_emb.expand(temb.shape[0], -1)
            parts.append(layer_emb)

        x = torch.cat(parts, dim=-1)
        return self.mlp(x)
