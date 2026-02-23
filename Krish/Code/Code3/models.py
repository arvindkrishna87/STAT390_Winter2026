"""
Model architectures for MIL training
- Robust KimiaNet weight loading for your two checkpoint formats:
  (A) KimiaNet.pth: likely features-only keys (e.g., "conv0.weight", "denseblock1....")
  (B) KimiaNet.pth: wrapped keys like "module.model.0.conv0.weight"
- Avoids silent hybrid initialization by (i) using weights=None and (ii) verifying a high match rate.
- Fixes zero-data branch bug: nn.Sequential has no .out_features
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Any, Optional, Tuple

from config import MODEL_CONFIG


# ----------------------------
# KimiaNet loading utilities
# ----------------------------
def _unwrap_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    """Handle common checkpoint wrappers."""
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
        if "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
            return ckpt_obj["model_state_dict"]
        return ckpt_obj
    raise ValueError(f"Checkpoint is not a dict/state_dict. Got type={type(ckpt_obj)}")


def _strip_prefix(s: str, prefix: str) -> str:
    return s[len(prefix):] if s.startswith(prefix) else s


def _make_features_state_dict_from_kimianet(sd_raw: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert multiple KimiaNet checkpoint key styles into torchvision DenseNet121 'features.*' keys.

    Supports:
      1) Already-full DenseNet keys:         "features.conv0.weight"
      2) Features-only keys:                "conv0.weight"
      3) Your wrapped checkpoint keys:      "module.model.0.conv0.weight"
         (also handles "model.0." without "module.")
    """
    sd = dict(sd_raw)

    # 1) Remove DataParallel wrapper if present
    if any(k.startswith("module.") for k in sd.keys()):
        sd = { _strip_prefix(k, "module."): v for k, v in sd.items() }

    out: Dict[str, torch.Tensor] = {}

    for k, v in sd.items():
        # Case: full-model style already contains "features."
        if k.startswith("features."):
            out[k] = v
            continue

        # Case: wrapped sequential encoder: "model.0.<encoder_key>"
        # Your checkpoint example: "module.model.0.conv0.weight" -> after stripping module -> "model.0.conv0.weight"
        if k.startswith("model.0."):
            kk = _strip_prefix(k, "model.0.")
            out[f"features.{kk}"] = v
            continue

        # Case: some wrappers might be "model.features.<...>"
        if k.startswith("model.features."):
            kk = _strip_prefix(k, "model.")
            out[kk] = v
            continue

        # Case: features-only style (encoder keys without prefix)
        # e.g., "conv0.weight", "denseblock1.denselayer1.norm1.weight", ...
        out[f"features.{k}"] = v

    return out


def load_kimianet_densenet121(
    kimianet_path: str,
    device: str = "cpu",
    min_feature_match_frac: float = 0.95,
    verbose: bool = True,
) -> models.DenseNet:
    """
    Create a torchvision DenseNet121 and load KimiaNet weights into encoder (features.*)
    for your checkpoint key formats.

    Uses weights=None to avoid torchvision pretrained deprecation warnings.
    """
    # Avoid ImageNet mixing; KimiaNet should fully define the encoder
    base_model = models.densenet121(weights=None)

    ckpt = torch.load(kimianet_path, map_location=torch.device(device))
    sd_raw = _unwrap_state_dict(ckpt)
    sd_feat = _make_features_state_dict_from_kimianet(sd_raw)

    model_sd = base_model.state_dict()
    model_feature_keys = [k for k in model_sd.keys() if k.startswith("features.")]
    model_feature_keyset = set(model_feature_keys)

    # Keep only keys that (a) exist in model and (b) match shape.
    filtered = {
        k: v for k, v in sd_feat.items()
        if (k in model_sd) and (model_sd[k].shape == v.shape)
    }

    # Load filtered (prevents shape-mismatch junk from polluting unexpected keys)
    res = base_model.load_state_dict(filtered, strict=False)

    matched_feature_keys = len(model_feature_keyset.intersection(filtered.keys()))
    match_frac = matched_feature_keys / max(1, len(model_feature_keyset))

    if verbose:
        print("=== KimiaNet load diagnostics ===")
        print(f"Checkpoint path: {kimianet_path}")
        print(f"Checkpoint total keys (raw): {len(sd_raw)}")
        print(f"Checkpoint keys mapped to features.*: {len(sd_feat)}")
        print(f"Filtered keys (exist+shape match): {len(filtered)}")
        print(f"Model feature keys: {len(model_feature_keyset)}")
        print(f"Matched feature keys: {matched_feature_keys}/{len(model_feature_keyset)} ({match_frac:.1%})")
        print(f"Missing keys reported by PyTorch: {len(res.missing_keys)}")
        print(f"Unexpected keys reported by PyTorch: {len(res.unexpected_keys)}")
        if len(res.missing_keys) > 0:
            print("  Missing examples:", res.missing_keys[:10])
        if len(res.unexpected_keys) > 0:
            print("  Unexpected examples:", res.unexpected_keys[:10])

        # Show a couple example key mappings for sanity
        ex_raw = next(iter(sd_raw.keys()))
        ex_mapped = next(iter(sd_feat.keys()))
        print(f"Example raw key:    {ex_raw}")
        print(f"Example mapped key: {ex_mapped}")

    if match_frac < min_feature_match_frac:
        raise RuntimeError(
            "KimiaNet weights did not load cleanly into torchvision DenseNet121 encoder.\n"
            f"Only matched {match_frac:.1%} of encoder keys; expected >= {min_feature_match_frac:.0%}.\n"
            "Given your earlier output, this usually means a remaining naming/shape mismatch.\n"
            "Next: print a few raw keys + a few base_model.features.state_dict().keys() and compare."
        )

    return base_model


# ----------------------------
# Model definition
# ----------------------------
class AttentionPool(nn.Module):
    """
    Attention pooling mechanism for MIL
    Pools patch-level features into single bag-level representation
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        weights = self.attention(x)            # (B, M, 1)
        weights = torch.softmax(weights, dim=1)
        weighted_x = (weights * x).sum(dim=1)  # (B, D)
        if return_weights:
            return weighted_x, weights.squeeze(-1)
        return weighted_x


class HierarchicalAttnMIL(nn.Module):
    """
    Hierarchical Attention MIL model for multi-stain pathology images
    """
    def __init__(
        self,
        base_model: Optional[nn.Module] = None,
        num_classes: int = 2,
        embed_dim: int = 512,
        dropout: float = 0.3,
        kimianet_path: str = "KimiaNet.pth",  # <-- default to your wrapped file
        kimianet_min_feature_match_frac: float = 0.95,
    ):
        super().__init__()

        self.num_classes = num_classes

        if base_model is None:
            base_model = load_kimianet_densenet121(
                kimianet_path=kimianet_path,
                device="cpu",
                min_feature_match_frac=kimianet_min_feature_match_frac,
                verbose=True,
            )

        self.features = base_model.features

        # Freeze the pretrained feature extractor
        for param in self.features.parameters():
            param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d((2, 2))

        self.patch_projector = nn.Sequential(
            nn.Linear(base_model.classifier.in_features * 4, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.patch_attention = AttentionPool(embed_dim, MODEL_CONFIG["attention_hidden_dim"], dropout=dropout)
        self.stain_attention = AttentionPool(embed_dim, MODEL_CONFIG["attention_hidden_dim"], dropout=dropout)
        self.case_attention = AttentionPool(embed_dim, MODEL_CONFIG["attention_hidden_dim"], dropout=dropout)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def process_single_stain(
        self,
        slice_list: List[torch.Tensor],
        stain_name: str,
        return_attn_weights: bool = False,
    ):
        slice_embeddings = []
        slice_attention_weights = []

        for slice_tensor in slice_list:
            P, C, H, W = slice_tensor.shape

            with torch.no_grad():
                patch_features = self.features(slice_tensor)   # (P, F, h, w)
                pooled = self.pool(patch_features).view(P, -1) # (P, 4*F)

            patch_embeddings = self.patch_projector(pooled)    # (P, D)

            if return_attn_weights:
                slice_emb, patch_weights = self.patch_attention(patch_embeddings.unsqueeze(0), return_weights=True)
                slice_attention_weights.append(patch_weights.squeeze(0).detach())
            else:
                slice_emb = self.patch_attention(patch_embeddings.unsqueeze(0))

            slice_embeddings.append(slice_emb.squeeze(0))

            del patch_features, pooled, patch_embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if slice_embeddings:
            stain_slice_embeddings = torch.stack(slice_embeddings)  # (num_slices, D)

            if return_attn_weights:
                stain_emb, stain_weights = self.stain_attention(stain_slice_embeddings.unsqueeze(0), return_weights=True)
                stain_attention_info = {
                    "slice_weights": stain_weights.squeeze(0).detach(),
                    "patch_weights": slice_attention_weights,
                }
            else:
                stain_emb = self.stain_attention(stain_slice_embeddings.unsqueeze(0))
                stain_attention_info = None

            del stain_slice_embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return stain_emb.squeeze(0), stain_attention_info

        return None, None

    def forward(self, stain_slices_dict: Dict[str, List[torch.Tensor]], return_attn_weights: bool = False):
        stain_embeddings = []
        stain_names = []
        stain_attention_weights = {}

        for stain_name, slice_list in stain_slices_dict.items():
            if not slice_list:
                continue

            stain_emb, stain_attn_info = self.process_single_stain(slice_list, stain_name, return_attn_weights)

            if stain_emb is not None:
                stain_embeddings.append(stain_emb)
                stain_names.append(stain_name)

                if return_attn_weights and stain_attn_info:
                    stain_attention_weights[stain_name] = stain_attn_info

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # If no stains have data, return zero logits (FIXED)
        if not stain_embeddings:
            logits = torch.zeros(self.num_classes, device=next(self.parameters()).device)
            if return_attn_weights:
                return logits, {}
            return logits

        case_stain_embeddings = torch.stack(stain_embeddings)  # (num_stains, D)

        if return_attn_weights:
            case_emb, case_weights = self.case_attention(case_stain_embeddings.unsqueeze(0), return_weights=True)
            all_weights = {
                "case_weights": case_weights.squeeze(0),
                "stain_weights": stain_attention_weights,
                "stain_order": stain_names,
            }
        else:
            case_emb = self.case_attention(case_stain_embeddings.unsqueeze(0))

        logits = self.classifier(case_emb.squeeze(0))

        del case_stain_embeddings, stain_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if return_attn_weights:
            return logits, all_weights

        return logits


def create_model(
    num_classes: int = None,
    embed_dim: int = None,
    dropout: float = None,
    kimianet_path: str = "KimiaNet.pth",
    kimianet_min_feature_match_frac: float = 0.95,
) -> HierarchicalAttnMIL:
    """
    Factory function to create the MIL model.
    - Does NOT use pretrained=True (no ImageNet mixing)
    - Loads KimiaNet weights with key remapping + shape filtering + verification
    """
    if num_classes is None:
        num_classes = MODEL_CONFIG["num_classes"]
    if embed_dim is None:
        embed_dim = MODEL_CONFIG["embed_dim"]
    if dropout is None:
        from config import TRAINING_CONFIG
        dropout = TRAINING_CONFIG.get("dropout", 0.3)

    model = HierarchicalAttnMIL(
        base_model=None,
        num_classes=num_classes,
        embed_dim=embed_dim,
        dropout=dropout,
        kimianet_path=kimianet_path,
        kimianet_min_feature_match_frac=kimianet_min_feature_match_frac,
    )

    return model