"""
Model architectures for MIL training
"""
import os
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Any, Optional, Tuple

from config import MODEL_CONFIG


class AttentionPool(nn.Module):
    """
    Gated attention pooling for Multiple Instance Learning (MIL)

    This module takes a bag of embeddings (e.g., patches, slices, or stains)
    and computes a weighted sum where the weights are learned via
    gated attention.

    Compared to plain attention, the gate allows the model to learn
    feature interactions (i.e., "this feature matters only when another
    feature is present"), which is very useful for heterogeneous
    pathology data.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.0, temperature = 1.0):
        super().__init__()
        self.temperature = temperature

        self.V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )

        self.U = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )

        self.w = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor, return_weights: bool = False):
        """
        Args:
            x: (B, M, D)
               B = batch size
               M = number of instances in the bag (patches / slices / stains)
               D = embedding dimension

            return_weights:
               If True, also return attention weights for visualization

        Returns:
            weighted_x: (B, D) → bag-level representation
            weights: (B, M)   → attention weights (optional)
        """        

        A = self.V(x) * (1 + self.U(x))          # (B, M, H)
        weights = self.w(A)                # (B, M, 1)
        weights = torch.softmax(weights / self.temperature, dim=1)

        weighted_x = (weights * x).sum(dim=1)

        if return_weights:
            return weighted_x, weights.squeeze(-1)

        return weighted_x


class HierarchicalAttnMIL(nn.Module):
    """
    Hierarchical Attention MIL model for multi-stain pathology images
    
    Three levels of attention:
    1. Patch-level: within each stain-slice
    2. Stain-level: across slices within each stain  
    3. Case-level: across different stains
    """
    def __init__(self, base_model=None, num_classes: int = 2, embed_dim: int = 512, dropout: float = 0.05):
        super().__init__()
        
        if base_model is None:
            base_model = models.densenet121(pretrained=True)
        
        # Shared feature extractor (pretrained CNN) - FROZEN
        self.features = base_model.features
        
        # Freeze the pretrained feature extractor
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Adaptive pooling to get richer features than just 1x1
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        
        # Patch projector: maps CNN features to patch embeddings with dropout
        self.patch_projector = nn.Sequential(
            nn.Linear(base_model.classifier.in_features * 4, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Three levels of attention with dropout
        self.patch_attention = AttentionPool(embed_dim, MODEL_CONFIG['attention_hidden_dim'], dropout=dropout, temperature=2.0)
        self.stain_attention = AttentionPool(embed_dim, MODEL_CONFIG['attention_hidden_dim'], dropout=dropout, temperature=2.5)
        self.case_attention = AttentionPool(embed_dim, MODEL_CONFIG['attention_hidden_dim'], dropout=dropout, temperature=1.0)
        
        # Final classifier with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
    
    def process_single_stain(self, slice_list: List[torch.Tensor], stain_name: str, 
                            return_attn_weights: bool = False):
        """
        Process a single stain sequentially to save memory
        """
        slice_embeddings = []
        slice_attention_weights = []
        
        # Process each slice within this stain
        for slice_tensor in slice_list:
            # slice_tensor shape: (P, C, H, W) where P = number of patches
            P, C, H, W = slice_tensor.shape
            
            # Extract features for all patches in this slice (frozen features)
            with torch.no_grad():
                patch_features = self.features(slice_tensor)  # (P, F, h, w)
                pooled = self.pool(patch_features).view(P, -1)  # (P, 4*F)
            
            # Only train the projector and attention modules
            patch_embeddings = self.patch_projector(pooled)  # (P, D) - now includes ReLU and dropout
            
            # Apply patch-level attention to get slice embedding
            if return_attn_weights:
                slice_emb, patch_weights = self.patch_attention(
                    patch_embeddings.unsqueeze(0), return_weights=True
                )
                slice_attention_weights.append(patch_weights.squeeze(0).detach())
            else:
                slice_emb = self.patch_attention(patch_embeddings.unsqueeze(0))
            
            slice_embeddings.append(slice_emb.squeeze(0))  # (D,)
            
            # Clear intermediate tensors
            del patch_features, pooled, patch_embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Stack slice embeddings for this stain
        if slice_embeddings:
            stain_slice_embeddings = torch.stack(slice_embeddings)  # (num_slices, D)
            
            # Apply stain-level attention across slices
            if return_attn_weights:
                stain_emb, stain_weights = self.stain_attention(
                    stain_slice_embeddings.unsqueeze(0), return_weights=True
                )
                stain_attention_info = {
                    'slice_weights': stain_weights.squeeze(0).detach(),
                    'patch_weights': slice_attention_weights
                }
            else:
                stain_emb = self.stain_attention(stain_slice_embeddings.unsqueeze(0))
                stain_attention_info = None
            
            # Clean up
            del stain_slice_embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return stain_emb.squeeze(0), stain_attention_info
        
        return None, None

    def forward(self, stain_slices_dict: Dict[str, List[torch.Tensor]], 
                return_attn_weights: bool = False):
        """
        Sequential stain processing to reduce memory usage
        """
        stain_embeddings = []
        stain_names = []
        stain_attention_weights = {}
        
        # Process each stain sequentially
        for stain_name, slice_list in stain_slices_dict.items():
            if not slice_list:  # Skip if no slices for this stain
                continue
            
            # Process this stain completely
            stain_emb, stain_attn_info = self.process_single_stain(
                slice_list, stain_name, return_attn_weights
            )
            
            if stain_emb is not None:
                stain_embeddings.append(stain_emb)
                stain_names.append(stain_name)
                
                if return_attn_weights and stain_attn_info:
                    stain_attention_weights[stain_name] = stain_attn_info
            
            # Force cleanup after each stain
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # If no stains have data, return zero logits
        if not stain_embeddings:
            logits = torch.zeros(self.classifier.out_features).to(next(self.parameters()).device)
            if return_attn_weights:
                return logits, {}
            return logits
        
        # Stack stain embeddings for case-level attention
        case_stain_embeddings = torch.stack(stain_embeddings)  # (num_stains, D)
        
        # Apply case-level attention across stains
        if return_attn_weights:
            case_emb, case_weights = self.case_attention(
                case_stain_embeddings.unsqueeze(0), return_weights=True
            )
            # Package all attention weights for visualization
            all_weights = {
                'case_weights': case_weights.squeeze(0),
                'stain_weights': stain_attention_weights,
                'stain_order': stain_names
            }
        else:
            case_emb = self.case_attention(case_stain_embeddings.unsqueeze(0))
        
        # Final classification with dropout
        logits = self.classifier(case_emb.squeeze(0))  # (num_classes,) - now includes dropout
        
        # Final cleanup
        del case_stain_embeddings, stain_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if return_attn_weights:
            return logits, all_weights
        
        return logits

def load_kimianet_densenet(weights_path: str):
    """
    Load DenseNet-121 and initialize with KimiaNet checkpoint weights.
    """
    base_model = models.densenet121(pretrained=False)
    ckpt = torch.load(weights_path, map_location="cpu")

    model_dict = base_model.state_dict()
    filtered = {
        k: v for k, v in ckpt.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }

    model_dict.update(filtered)
    base_model.load_state_dict(model_dict)
    return base_model


def create_model(
    num_classes: int = None,
    embed_dim: int = None,
    dropout: float = None,
    pretrained: bool = True
) -> HierarchicalAttnMIL:
    """
    Factory function to create the MIL model
    """
    if num_classes is None:
        num_classes = MODEL_CONFIG['num_classes']
    if embed_dim is None:
        embed_dim = MODEL_CONFIG['embed_dim']
    if dropout is None:
        from config import TRAINING_CONFIG
        dropout = TRAINING_CONFIG.get('dropout', 0.3)

    if pretrained:
        # Prefer config key if present; fallback to local file beside this models.py
        weights_path = MODEL_CONFIG.get('kimianet_weights_path')
        if not weights_path:
            weights_path = os.path.join(
                os.path.dirname(__file__),
                "KimiaNetPyTorchWeights.pth"
            )
        base_model = load_kimianet_densenet(weights_path)
    else:
        base_model = models.densenet121(pretrained=False)

    model = HierarchicalAttnMIL(
        base_model=base_model,
        num_classes=num_classes,
        embed_dim=embed_dim,
        dropout=dropout
    )
    return model