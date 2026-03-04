import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple

from config import MODEL_CONFIG

# -------------------------------------------------------
# ROMA - Mandatory changes - replaced attention pool with gated attention pool 
class GatedAttentionPool(nn.Module):
    """
    Gated Attention pooling for MIL.
    - No dropout in the attention MLP
    - Gating mechanism: tanh branch * sigmoid branch
    - Optionally returns attention weights and/or entropy
    Input:  x (B, M, D)
    Output: weighted_x (B, D), optionally weights (B, M), optionally entropy (B,)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.V = nn.Linear(input_dim, hidden_dim)
        self.U = nn.Linear(input_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, return_weights: bool = False, return_entropy: bool = False):
        # x: (B, M, D)
        Vx = torch.tanh(self.V(x))
        Ux = torch.sigmoid(self.U(x))
        weights = torch.softmax(self.w(Vx * Ux), dim=1)  # (B, M, 1)
        weighted_x = (weights * x).sum(dim=1)             # (B, D)

        entropy = None
        if return_entropy:
            p = weights.squeeze(-1)                           # (B, M)
            entropy = -(p * torch.log(p + 1e-8)).sum(dim=1)  # (B,)

        if return_weights and return_entropy:
            return weighted_x, weights.squeeze(-1), entropy
        elif return_weights:
            return weighted_x, weights.squeeze(-1)
        elif return_entropy:
            return weighted_x, entropy
        return weighted_x
# end of ROMA addition
# -------------------------------------------------------

class HierarchicalAttnMIL(nn.Module):
    """
    Hierarchical Attention MIL model for multi-stain pathology images
    ASSUMES precomputed pooled features per patch: (P, 4096).
    Uses GatedAttentionPool with no dropout in the attention MLP.
    """
    def __init__(
        self,
        num_classes: int = 2,
        embed_dim: int = 512,
        dropout: float = 0.3,
        pooled_dim: int = 4096,  # = 4*F for DenseNet121 with 2x2 pooling (F=1024)
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pooled_dim = pooled_dim

        # Project pooled patch features -> patch embedding
        self.patch_projector = nn.Sequential(
            nn.Linear(self.pooled_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Attention modules
        # -------------------------------------------------------
        # ROMA - Mandatory changes - using gated attention pool and delete dropout
        self.patch_attention = GatedAttentionPool(embed_dim, MODEL_CONFIG["attention_hidden_dim"]) 
        self.stain_attention = GatedAttentionPool(embed_dim, MODEL_CONFIG["attention_hidden_dim"]) 
        self.case_attention  = GatedAttentionPool(embed_dim, MODEL_CONFIG["attention_hidden_dim"]) 
        # -------------------------------------------------------

        # Final classifier
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
        """
        slice_list: list of tensors, each tensor is one slice:
          slice_tensor shape (P, pooled_dim) e.g., (P, 4096)
        """
        slice_embeddings = []
        slice_attention_weights = []
        patch_entropies = [] # ROMA mandatory changes addition

        device = next(self.parameters()).device

        for slice_tensor in slice_list:
            # Expect precomputed pooled features: (P, pooled_dim)
            if slice_tensor.dim() != 2:
                raise ValueError(
                    f"[{stain_name}] Expected pooled features with shape (P, {self.pooled_dim}), "
                    f"but got {tuple(slice_tensor.shape)}"
                )
            if slice_tensor.size(1) != self.pooled_dim:
                raise ValueError(
                    f"[{stain_name}] Expected pooled_dim={self.pooled_dim}, "
                    f"but got {slice_tensor.size(1)}"
                )

            pooled = slice_tensor.to(device, non_blocking=True)  # (P, pooled_dim)
            patch_embeddings = self.patch_projector(pooled)      # (P, D)

            # -------------------------------------------------------
            # ROMA - Mandatory changes edit
            if return_attn_weights:
                slice_emb, patch_weights, patch_entropy = self.patch_attention(
                    patch_embeddings.unsqueeze(0), return_weights=True, return_entropy=True
                )
                slice_attention_weights.append(patch_weights.squeeze(0).detach())  # (P,)
                patch_entropies.append(patch_entropy.squeeze(0))                   # scalar
            else:
                slice_emb = self.patch_attention(patch_embeddings.unsqueeze(0))
            # End of edited code
            # -------------------------------------------------------

            slice_embeddings.append(slice_emb.squeeze(0))  # (D,)

        if not slice_embeddings:
            return None, None

        stain_slice_embeddings = torch.stack(slice_embeddings)  # (num_slices, D)

        # -------------------------------------------------------
        # ROMA - Mandatory changes edit
        if return_attn_weights:
            stain_emb, stain_weights, stain_entropy = self.stain_attention(
                stain_slice_embeddings.unsqueeze(0), return_weights=True, return_entropy=True
            )
            stain_attention_info = {
                "slice_weights":   stain_weights.squeeze(0).detach(),  # (num_slices,)
                "patch_weights":   slice_attention_weights,            # list[(P_i,)]
                "slice_entropy":   stain_entropy.squeeze(0),           # scalar
                "patch_entropies": patch_entropies,                    # list[scalar]
            }
        # End of edited code
        # -------------------------------------------------------
        else:
            stain_emb = self.stain_attention(stain_slice_embeddings.unsqueeze(0))
            stain_attention_info = None

        return stain_emb.squeeze(0), stain_attention_info

    def forward(self, stain_slices_dict: Dict[str, List[torch.Tensor]], return_attn_weights: bool = False):
        stain_embeddings = []
        stain_names = []
        stain_attention_weights = {}

        for stain_name, slice_list in stain_slices_dict.items():
            if not slice_list:
                continue

            stain_emb, stain_attn_info = self.process_single_stain(
                slice_list, stain_name, return_attn_weights
            )

            if stain_emb is not None:
                stain_embeddings.append(stain_emb)
                stain_names.append(stain_name)
                if return_attn_weights and stain_attn_info is not None:
                    stain_attention_weights[stain_name] = stain_attn_info

        if not stain_embeddings:
            logits = torch.zeros(self.num_classes, device=next(self.parameters()).device)
            if return_attn_weights:
                return logits, {}
            return logits

        case_stain_embeddings = torch.stack(stain_embeddings)  # (num_stains, D)
        # -------------------------------------------------------
        # ROMA - Mandatory changes edit
        if return_attn_weights:
            case_emb, case_weights, case_entropy = self.case_attention(
                case_stain_embeddings.unsqueeze(0), return_weights=True, return_entropy=True
            )
            all_weights = {
                "case_weights": case_weights.squeeze(0),   # (num_stains,)
                "case_entropy": case_entropy.squeeze(0),   # scalar
                "stain_weights": stain_attention_weights,
                "stain_order": stain_names,
            }
        # End of edited code
        # -------------------------------------------------------
        else:
            case_emb = self.case_attention(case_stain_embeddings.unsqueeze(0))

        logits = self.classifier(case_emb.squeeze(0))  # (num_classes,)

        if return_attn_weights:
            return logits, all_weights
        return logits

    # -------------------------------------------------------
    # PAIGE - optional task 3 added code
    def predict_patches(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Applies the trained patch_projector and classifier directly to raw patch embeddings, to be called after training is complete.
        Takes (N, 4096) embeddings --> patch_projector --> (N, 512) --> classifier --> logits --> softmax -->probs
        """
        projected = self.patch_projector(patch_embeddings)  # (N, 4096) --> (N, 512)
        logits = self.classifier(projected) # (N, 512) --> (N, 2)
        return torch.softmax(logits, dim=-1) # convert raw logits to class probabilities
    # End of added code
    # -------------------------------------------------------

def create_model(
    num_classes: int = None,
    embed_dim: int = None,
    dropout: float = None,
    pooled_dim: int = 4096,
) -> HierarchicalAttnMIL:
    if num_classes is None:
        num_classes = MODEL_CONFIG["num_classes"]
    if embed_dim is None:
        embed_dim = MODEL_CONFIG["embed_dim"]
    if dropout is None:
        from config import TRAINING_CONFIG
        dropout = TRAINING_CONFIG.get("dropout", 0.3)

    return HierarchicalAttnMIL(
        num_classes=num_classes,
        embed_dim=embed_dim,
        dropout=dropout,
        pooled_dim=pooled_dim,
    )