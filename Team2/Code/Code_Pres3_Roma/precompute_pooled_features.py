#!/usr/bin/env python3
"""
precompute_pooled_features.py

Precompute per-patch pooled DenseNet/KimiaNet features for all patch PNGs.

For each patch image:
  embeddings_dir/<basename>.pt containing a 1D tensor of shape (4*F,)
  where F = DenseNet121 classifier.in_features (typically 1024),
  and we use AdaptiveAvgPool2d((2,2)) -> flatten -> length 4*F = 4096.

This preserves patch-level attention because training still sees a set of patch vectors per slice.

IMPORTANT:
- Precomputing makes your image transforms effectively deterministic.
  (No RandomResizedCrop / random flips at training time unless you precompute multiple variants.)
"""

import os
import argparse
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as T
import torchvision.models as models


# ----------------------------
# Dataset over PNG paths
# ----------------------------
class PatchPathDataset(Dataset):
    """
    Returns (fname, tensor) for each image, or (fname, None) if unreadable.
    fname is the basename (as passed in filenames list).
    """
    def __init__(self, patches_dir: str, filenames: List[str], transform: T.Compose):
        self.patches_dir = patches_dir
        self.filenames = filenames
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[str, Optional[torch.Tensor]]:
        fname = self.filenames[idx]
        path = os.path.join(self.patches_dir, fname)

        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            return fname, None

        x = self.transform(img)
        return fname, x


def collate_skip_bad(batch):
    """
    Filter out unreadable images from the batch.

    Returns:
      fnames: List[str]              (ONLY the good ones)
      xs:     Tensor(B, C, H, W)     (ONLY the good ones) OR None if none good
      bad:    int                   (# bad items in the original batch)
    """
    bad = sum(1 for _, x in batch if x is None)
    good = [(f, x) for (f, x) in batch if x is not None]
    if len(good) == 0:
        return [], None, bad
    fnames, xs = zip(*good)
    return list(fnames), torch.stack(xs, dim=0), bad


# ----------------------------
# KimiaNet loader utilities
# ----------------------------
def _unwrap_state_dict(ckpt_obj):
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


def _make_features_state_dict_from_kimianet(sd_raw):
    """
    Convert multiple KimiaNet checkpoint key styles into torchvision DenseNet121 'features.*' keys.

    Supports:
      1) already full keys:          "features.conv0.weight"
      2) features-only keys:         "conv0.weight"
      3) wrapped sequential encoder: "module.model.0.conv0.weight" (or "model.0.*")
      4) "model.features.*"
    """
    sd = dict(sd_raw)

    # Remove DataParallel wrapper if present
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {_strip_prefix(k, "module."): v for k, v in sd.items()}

    out = {}
    for k, v in sd.items():
        if k.startswith("features."):
            out[k] = v
        elif k.startswith("model.0."):
            kk = _strip_prefix(k, "model.0.")
            out[f"features.{kk}"] = v
        elif k.startswith("model.features."):
            kk = _strip_prefix(k, "model.")
            out[kk] = v  # already starts with "features."
        else:
            out[f"features.{k}"] = v
    return out


def load_kimianet_densenet121(
    kimianet_path: str,
    device: str = "cpu",
    verbose: bool = True,
) -> models.DenseNet:
    """
    Instantiate torchvision DenseNet121 (weights=None) and load KimiaNet weights into encoder features.*.
    Shape-mismatching keys are filtered out.
    """
    base_model = models.densenet121(weights=None)

    ckpt = torch.load(kimianet_path, map_location=torch.device(device))
    sd_raw = _unwrap_state_dict(ckpt)
    sd_feat = _make_features_state_dict_from_kimianet(sd_raw)

    model_sd = base_model.state_dict()
    feat_keys = [k for k in model_sd.keys() if k.startswith("features.")]

    filtered = {k: v for k, v in sd_feat.items() if (k in model_sd and model_sd[k].shape == v.shape)}
    res = base_model.load_state_dict(filtered, strict=False)

    if verbose:
        matched = len(set(feat_keys).intersection(filtered.keys()))
        print("=== KimiaNet load diagnostics (precompute) ===")
        print(f"Checkpoint: {kimianet_path}")
        print(f"Filtered keys loaded: {len(filtered)}")
        print(f"Matched features.* keys: {matched}/{len(feat_keys)} ({matched/max(1,len(feat_keys)):.1%})")
        print(f"Missing keys: {len(res.missing_keys)} (examples: {res.missing_keys[:5]})")
        print(f"Unexpected keys: {len(res.unexpected_keys)}")

    return base_model


# ----------------------------
# Helpers
# ----------------------------
def list_pngs_in_dir(patches_dir: str) -> List[str]:
    fnames = [f for f in os.listdir(patches_dir) if f.lower().endswith(".png")]
    fnames.sort()
    return fnames


def out_path_for_fname(embeddings_dir: str, fname: str) -> str:
    out_name = os.path.splitext(fname)[0] + ".pt"
    return os.path.join(embeddings_dir, out_name)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches_dir", default="/projects/e32998/patches_benign_split", help="Folder with PNG patches")
    ap.add_argument("--embeddings_dir", default="/projects/e32998/patches_pooled4096", help="Where to save .pt features")
    ap.add_argument("--kimianet_ckpt", default="KimiaNet.pth", help="KimiaNet checkpoint path")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp16", help="Storage dtype for saved features")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .pt files")
    ap.add_argument("--log_every", type=int, default=50, help="Print progress every N steps")
    args = ap.parse_args()

    os.makedirs(args.embeddings_dir, exist_ok=True)

    print(f"[INFO] patches_dir:     {args.patches_dir}")
    print(f"[INFO] embeddings_dir:  {args.embeddings_dir}")
    print(f"[INFO] kimianet_ckpt:   {args.kimianet_ckpt}")
    print(f"[INFO] device:         {args.device}")
    print(f"[INFO] batch_size:      {args.batch_size}")
    print(f"[INFO] num_workers:     {args.num_workers}")
    print(f"[INFO] overwrite:       {args.overwrite}")
    print(f"[INFO] dtype:           {args.dtype}")

    # Deterministic transform (no random augmentation)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    filenames = list_pngs_in_dir(args.patches_dir)
    total_png = len(filenames)
    print(f"[INFO] Found {total_png} PNG files.")

    # Load KimiaNet DenseNet encoder
    base_model = load_kimianet_densenet121(args.kimianet_ckpt, device="cpu", verbose=True)
    features = base_model.features
    features.eval()
    for p in features.parameters():
        p.requires_grad = False

    pool = nn.AdaptiveAvgPool2d((2, 2))
    pool.eval()

    features = features.to(args.device)
    pool = pool.to(args.device)

    ds = PatchPathDataset(args.patches_dir, filenames, transform)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_skip_bad,
        persistent_workers=(args.num_workers > 0),
    )

    save_dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    # Counters (count files, not batches)
    seen_total = 0
    saved_total = 0
    skipped_existing = 0
    skipped_bad = 0

    # Use inference_mode for speed + lower overhead than no_grad
    with torch.inference_mode():
        for step, (fnames, x, bad) in enumerate(dl, start=1):
            # Count bad + seen (good + bad) for this batch
            skipped_bad += bad
            seen_total += (len(fnames) + bad)

            # x is None only if all images in this batch were unreadable
            if x is None:
                if args.log_every > 0 and step % args.log_every == 0:
                    print(
                        f"[PROGRESS] step={step} seen={seen_total}/{total_png} "
                        f"saved={saved_total} skipped_existing={skipped_existing} skipped_bad={skipped_bad}"
                    )
                continue

            # Decide which items to compute, using indices for perfect alignment
            to_compute: List[Tuple[str, str]] = []
            keep_idx: List[int] = []

            for i, fname in enumerate(fnames):
                out_path = out_path_for_fname(args.embeddings_dir, fname)
                if (not args.overwrite) and os.path.exists(out_path):
                    skipped_existing += 1
                else:
                    to_compute.append((fname, out_path))
                    keep_idx.append(i)

            if len(to_compute) == 0:
                if args.log_every > 0 and step % args.log_every == 0:
                    print(
                        f"[PROGRESS] step={step} seen={seen_total}/{total_png} "
                        f"saved={saved_total} skipped_existing={skipped_existing} skipped_bad={skipped_bad}"
                    )
                continue

            # Subselect x using index list to preserve order exactly
            x_keep = x[keep_idx].to(args.device, non_blocking=True)

            feats = features(x_keep)          # (B, F, h, w)
            pooled = pool(feats).flatten(1)   # (B, 4*F)
            pooled = pooled.to(dtype=save_dtype).cpu()

            for j, (_, out_path) in enumerate(to_compute):
                torch.save(pooled[j], out_path)
                saved_total += 1

            if args.log_every > 0 and (step % args.log_every == 0 or seen_total >= total_png):
                print(
                    f"[PROGRESS] step={step} seen={seen_total}/{total_png} "
                    f"saved={saved_total} skipped_existing={skipped_existing} skipped_bad={skipped_bad}"
                )

    print("[DONE]")
    print(f"[SUMMARY] total_png={total_png}")
    print(f"[SUMMARY] saved={saved_total} skipped_existing={skipped_existing} skipped_bad={skipped_bad}")
    print(f"[SUMMARY] embeddings_dir={args.embeddings_dir}")
    print("Note: saved tensors are length 4096 (for DenseNet121 with 2x2 pooling).")


if __name__ == "__main__":
    main()