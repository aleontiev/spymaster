"""
Multi-Scale JEPA: Combines embeddings from multiple JEPA models at different timescales.

This wrapper loads multiple pre-trained JEPA models (e.g., 15:5, 90:30, daily) and
combines their embeddings for downstream tasks like policy learning or probing.

Fusion strategies:
1. Concatenation: Simple [emb_1, emb_2, ...] -> larger embedding
2. Attention: Learned weighted combination based on query
3. Gated: Learned gates per scale
"""
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.lejepa import LeJEPA


class MultiScaleJEPA(nn.Module):
    """
    Multi-scale JEPA that combines embeddings from multiple timescales.

    Example usage:
        # Load two pre-trained JEPAs
        model = MultiScaleJEPA.from_checkpoints({
            "short": "checkpoints/jepa-15-5.pt",   # 15 min context, 5 min target
            "medium": "checkpoints/jepa-90-30.pt", # 90 min context, 30 min target
        })

        # Encode at multiple scales
        combined_emb = model.encode({
            "short": context_15min,
            "medium": context_90min,
        })
        # combined_emb shape: [B, num_scales * embedding_dim] for concat
        #                  or [B, embedding_dim] for attention/gated
    """

    def __init__(
        self,
        models: Dict[str, LeJEPA],
        fusion: str = "concat",
        embedding_dim: Optional[int] = None,
    ):
        """
        Args:
            models: Dict mapping scale name to LeJEPA model
            fusion: Fusion strategy - "concat", "attention", or "gated"
            embedding_dim: Output embedding dim (only for attention/gated fusion)
        """
        super().__init__()

        self.scale_names = list(models.keys())
        self.models = nn.ModuleDict(models)
        self.fusion_type = fusion
        self.num_scales = len(models)

        # Get embedding dim from first model
        first_model = next(iter(models.values()))
        self.per_scale_dim = first_model.embedding_dim
        self.total_concat_dim = self.num_scales * self.per_scale_dim

        # Freeze all JEPA models (we only train fusion layers)
        for model in self.models.values():
            for param in model.parameters():
                param.requires_grad = False

        # Setup fusion layers
        if fusion == "concat":
            self.output_dim = self.total_concat_dim
        elif fusion == "attention":
            self.output_dim = embedding_dim or self.per_scale_dim
            # Query projection for attention
            self.query_proj = nn.Linear(self.per_scale_dim, self.per_scale_dim)
            # Key projections per scale
            self.key_projs = nn.ModuleDict({
                name: nn.Linear(self.per_scale_dim, self.per_scale_dim)
                for name in self.scale_names
            })
            # Output projection
            self.out_proj = nn.Linear(self.per_scale_dim, self.output_dim)
        elif fusion == "gated":
            self.output_dim = embedding_dim or self.per_scale_dim
            # Gate network: takes all embeddings, outputs per-scale weights
            self.gate_net = nn.Sequential(
                nn.Linear(self.total_concat_dim, self.num_scales * 2),
                nn.ReLU(),
                nn.Linear(self.num_scales * 2, self.num_scales),
                nn.Softmax(dim=-1),
            )
            # Output projection
            self.out_proj = nn.Linear(self.per_scale_dim, self.output_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion}")

    @classmethod
    def from_checkpoints(
        cls,
        checkpoint_paths: Dict[str, Union[str, Path]],
        fusion: str = "concat",
        embedding_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> "MultiScaleJEPA":
        """
        Load multiple JEPA models from checkpoints.

        Args:
            checkpoint_paths: Dict mapping scale name to checkpoint path
            fusion: Fusion strategy
            embedding_dim: Output embedding dim (for attention/gated)
            device: Device to load models to

        Returns:
            MultiScaleJEPA instance with loaded models
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        models = {}
        for name, path in checkpoint_paths.items():
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {path}")

            checkpoint = torch.load(path, map_location=device, weights_only=False)
            config = checkpoint.get("config", {})

            # Create model with config from checkpoint
            model = LeJEPA(
                input_dim=config.get("input_dim", config.get("feature_dim", 95)),
                d_model=config.get("d_model", 128),
                nhead=config.get("nhead", config.get("num_heads", 8)),
                num_layers=config.get("num_layers", 4),
                embedding_dim=config.get("embedding_dim", 64),
                max_context_len=config.get("max_context_len", 90),
                dropout=config.get("dropout", 0.1),
                lambda_reg=config.get("lambda_reg", 0.5),
                reg_type=config.get("reg_type", "sigreg"),
            ).to(device)

            # Load weights (handle torch.compile prefix)
            state_dict_key = "model_state_dict" if "model_state_dict" in checkpoint else "state_dict"
            state_dict = checkpoint[state_dict_key]
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace("._orig_mod", "")
                new_state_dict[new_key] = value
            model.load_state_dict(new_state_dict)
            model.eval()

            models[name] = model
            print(f"Loaded {name} JEPA from {path}")
            print(f"  - input_dim: {config.get('input_dim', 70)}, embedding_dim: {config.get('embedding_dim', 64)}")
            print(f"  - max_context_len: {config.get('max_context_len', 90)}")

        return cls(models=models, fusion=fusion, embedding_dim=embedding_dim).to(device)

    def encode_scale(
        self,
        scale_name: str,
        x_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode at a single scale.

        Args:
            scale_name: Name of the scale (e.g., "short", "medium")
            x_context: Context tensor [B, T, F]

        Returns:
            Embedding [B, embedding_dim]
        """
        model = self.models[scale_name]
        with torch.no_grad():
            return model.encode(x_context)

    def encode(
        self,
        contexts: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Encode at all scales and fuse.

        Args:
            contexts: Dict mapping scale name to context tensor [B, T_scale, F]

        Returns:
            Fused embedding [B, output_dim]
        """
        # Encode at each scale
        embeddings = {}
        for name in self.scale_names:
            if name not in contexts:
                raise ValueError(f"Missing context for scale: {name}")
            embeddings[name] = self.encode_scale(name, contexts[name])

        # Fuse embeddings
        return self._fuse(embeddings)

    def _fuse(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse embeddings from multiple scales."""
        # Stack embeddings in consistent order
        emb_list = [embeddings[name] for name in self.scale_names]
        stacked = torch.stack(emb_list, dim=1)  # [B, num_scales, embedding_dim]

        if self.fusion_type == "concat":
            # Simple concatenation
            return stacked.view(stacked.shape[0], -1)  # [B, num_scales * embedding_dim]

        elif self.fusion_type == "attention":
            # Attention-based fusion
            # Use mean of embeddings as query
            query = self.query_proj(stacked.mean(dim=1))  # [B, embedding_dim]

            # Compute attention scores
            scores = []
            for i, name in enumerate(self.scale_names):
                key = self.key_projs[name](stacked[:, i])  # [B, embedding_dim]
                score = (query * key).sum(dim=-1, keepdim=True)  # [B, 1]
                scores.append(score)
            scores = torch.cat(scores, dim=-1)  # [B, num_scales]
            weights = F.softmax(scores / (self.per_scale_dim ** 0.5), dim=-1)  # [B, num_scales]

            # Weighted combination
            weights = weights.unsqueeze(-1)  # [B, num_scales, 1]
            combined = (stacked * weights).sum(dim=1)  # [B, embedding_dim]
            return self.out_proj(combined)

        elif self.fusion_type == "gated":
            # Gated fusion
            flat = stacked.view(stacked.shape[0], -1)  # [B, num_scales * embedding_dim]
            gates = self.gate_net(flat)  # [B, num_scales]
            gates = gates.unsqueeze(-1)  # [B, num_scales, 1]

            # Weighted combination
            combined = (stacked * gates).sum(dim=1)  # [B, embedding_dim]
            return self.out_proj(combined)

        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

    def forward(
        self,
        contexts: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass (alias for encode)."""
        return self.encode(contexts)

    def get_scale_configs(self) -> Dict[str, Dict]:
        """Get configuration for each scale."""
        configs = {}
        for name, model in self.models.items():
            configs[name] = {
                "input_dim": model.input_dim,
                "embedding_dim": model.embedding_dim,
                "max_context_len": model.max_context_len,
                "d_model": model.d_model,
            }
        return configs


class MultiScaleProbe(nn.Module):
    """
    Linear probe on top of MultiScaleJEPA for direction prediction.

    Example:
        multi_jepa = MultiScaleJEPA.from_checkpoints({...})
        probe = MultiScaleProbe(multi_jepa, num_classes=2)

        # Training
        emb = multi_jepa.encode(contexts)
        logits = probe(emb)
        loss = F.cross_entropy(logits, labels)
    """

    def __init__(
        self,
        multi_jepa: MultiScaleJEPA,
        num_classes: int = 2,
        hidden_dim: Optional[int] = None,
    ):
        """
        Args:
            multi_jepa: MultiScaleJEPA model
            num_classes: Number of output classes
            hidden_dim: Hidden layer dimension (None for linear probe)
        """
        super().__init__()
        self.multi_jepa = multi_jepa
        input_dim = multi_jepa.output_dim

        if hidden_dim:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.classifier = nn.Linear(input_dim, num_classes)

    def forward(
        self,
        contexts: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass: encode at all scales, then classify.

        Returns:
            Logits [B, num_classes]
        """
        emb = self.multi_jepa.encode(contexts)
        return self.classifier(emb)

    def encode(
        self,
        contexts: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Get combined embedding without classification."""
        return self.multi_jepa.encode(contexts)
