import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple

from config import MODEL_CONFIG

class AttentionPool(nn.Module):
    """
    Attention pooling mechanism for MIL.
    Input:  x (B, M, D)
    Output: weighted_x (B, D), optionally weights (B, M)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            #nn.Dropout(dropout),#Removed dropout from attention to preserve interpretability of weights
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        weights = self.attention(x)            # (B, M, 1)
        weights = torch.softmax(weights, dim=1)
        weighted_x = (weights * x).sum(dim=1)  # (B, D)
        if return_weights:
            return weighted_x, weights.squeeze(-1)  # (B, D), (B, M)
        return weighted_x


class HierarchicalAttnMIL(nn.Module):
    """
    Hierarchical Attention MIL model for multi-stain pathology images
    ASSUMES precomputed pooled features per patch: (P, 4096).
    """
    def __init__(
        self,
        num_classes: int = 2,
        embed_dim: int = 512,
        project_dropout: float = 0.3, #new change to add separate dropout for patch projector
        classifier_dropout: float = 0.3, #new change to dropout for the classifier (tune for OPTUNA)
        pooled_dim: int = 4096,  # = 4*F for DenseNet121 with 2x2 pooling (F=1024)
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pooled_dim = pooled_dim

        # Project pooled patch features -> patch embedding
        self.patch_projector = nn.Sequential(
            nn.Linear(self.pooled_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(project_dropout),
        )

        # Attention modules
        self.patch_attention = AttentionPool(embed_dim, MODEL_CONFIG["attention_hidden_dim"])
        self.stain_attention = AttentionPool(embed_dim, MODEL_CONFIG["attention_hidden_dim"])
        self.case_attention  = AttentionPool(embed_dim, MODEL_CONFIG["attention_hidden_dim"])

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
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

            if return_attn_weights:
                slice_emb, patch_weights = self.patch_attention(
                    patch_embeddings.unsqueeze(0), return_weights=True
                )
                slice_attention_weights.append(patch_weights.squeeze(0).detach())  # (P,)
            else:
                slice_emb = self.patch_attention(patch_embeddings.unsqueeze(0))

            slice_embeddings.append(slice_emb.squeeze(0))  # (D,)

        if not slice_embeddings:
            return None, None

        stain_slice_embeddings = torch.stack(slice_embeddings)  # (num_slices, D)

        if return_attn_weights:
            stain_emb, stain_weights = self.stain_attention(
                stain_slice_embeddings.unsqueeze(0), return_weights=True
            )
            stain_attention_info = {
                "slice_weights": stain_weights.squeeze(0).detach(),  # (num_slices,)
                "patch_weights": slice_attention_weights,            # list[(P_i,)]
            }
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

        if return_attn_weights:
            case_emb, case_weights = self.case_attention(
                case_stain_embeddings.unsqueeze(0), return_weights=True
            )
            all_weights = {
                "case_weights": case_weights.squeeze(0),  # (num_stains,)
                "stain_weights": stain_attention_weights,
                "stain_order": stain_names,
            }
        else:
            case_emb = self.case_attention(case_stain_embeddings.unsqueeze(0))

        logits = self.classifier(case_emb.squeeze(0))  # (num_classes,)

        if return_attn_weights:
            return logits, all_weights
        return logits


def create_model(
    num_classes: int = None,
    embed_dim: int = None,
    classifier_dropout: float = None,
    project_dropout: float = None,
    pooled_dim: int = 4096,
) -> HierarchicalAttnMIL:
    if num_classes is None:
        num_classes = MODEL_CONFIG["num_classes"]
    if embed_dim is None:
        embed_dim = MODEL_CONFIG["embed_dim"]
    if classifier_dropout is None:
        from config import TRAINING_CONFIG
        classifier_dropout = TRAINING_CONFIG.get("dropout", 0.3)
    if project_dropout is None:
        from config import TRAINING_CONFIG
        project_dropout = TRAINING_CONFIG.get("project_dropout", 0.3)

    return HierarchicalAttnMIL(
        num_classes=num_classes,
        embed_dim=embed_dim,
        classifier_dropout=classifier_dropout,
        project_dropout=project_dropout,
        pooled_dim=pooled_dim,
    )