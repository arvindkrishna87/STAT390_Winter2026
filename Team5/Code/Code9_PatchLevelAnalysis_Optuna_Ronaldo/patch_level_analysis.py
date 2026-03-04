"""
patch_level_analysis.py
-----------------------
For every test case that is a benign false-positive (GT=Benign, case-level
prediction=HighGrade), apply patch_projector → classifier directly to each
4096-dim embedding and save the patch images that were predicted HighGrade.

These are the patches most likely driving the incorrect case classification.

For ALL test cases, report the percentage of patches predicted in each class.
"""

import os
import shutil
from tkinter import Image
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

try:
    from tqdm import tqdm
    TQDM = True
except ImportError:
    TQDM = False


# ---------------------------------------------------------------------------
# Locate the classifier Linear inside the model
# ---------------------------------------------------------------------------

def _get_classifier_linear(model):
    import torch.nn as nn
    for attr in ("classifier", "fc", "head", "output_layer"):
        layer = getattr(model, attr, None)
        if layer is None:
            continue
        if isinstance(layer, nn.Linear):
            return layer
        if isinstance(layer, nn.Sequential):
            for sub in layer.children():
                if isinstance(sub, nn.Linear):
                    return sub
    # fallback: last Linear in the whole model
    last = None
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            last = mod
    if last is not None:
        return last
    raise RuntimeError("Cannot find classifier nn.Linear in the model.")


# ---------------------------------------------------------------------------
# Load embeddings for one case, tracking original patch paths
# ---------------------------------------------------------------------------

def _load_embeddings_with_paths(case_stain_dict, embeddings_dir):
    """
    Parameters
    ----------
    case_stain_dict : {stain: [[patch_path, ...], ...]}
        The value for one case_id in test_case_dict.
    embeddings_dir  : directory containing <stem>.pt embedding files

    Returns
    -------
    embeddings  : Tensor (N, 4096)
    patch_paths : list[str] of length N  — original image paths, one per patch
    """
    all_embs  = []
    all_paths = []

    for stain, slice_list in case_stain_dict.items():
        for patch_paths in slice_list:          # patch_paths is a list of file paths
            for patch_path in patch_paths:
                stem     = os.path.splitext(os.path.basename(patch_path))[0]
                emb_path = os.path.join(embeddings_dir, stem + ".pt")

                if not os.path.exists(emb_path):
                    continue
                try:
                    v = torch.load(emb_path, map_location="cpu")
                    if v.dim() != 1:
                        v = v.view(-1)
                    all_embs.append(v.float())
                    all_paths.append(patch_path)
                except Exception:
                    continue

    if not all_embs:
        return None, []

    return torch.stack(all_embs, dim=0), all_paths   # (N, 4096), [path×N]


# ---------------------------------------------------------------------------
# Patch-level forward pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def _predict_patches(model, embeddings, classifier_linear, device):
    """
    embeddings : Tensor (N, 4096)
    Returns probs (N, C) and preds (N,) on CPU.
    """
    x         = embeddings.to(device)
    projected = model.patch_projector(x)        # (N, 512)
    logits    = classifier_linear(projected)    # (N, num_classes)
    probs     = F.softmax(logits, dim=1).cpu()
    preds     = probs.argmax(dim=1).cpu()
    return probs, preds


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_patch_level_analysis(
    model,
    test_case_dict,
    test_label_map,
    case_level_preds,       # {case_id: predicted_label_int}  from predictions.csv
    embeddings_dir,
    output_dir,
    device,
    class_names = ("Benign", "HighGrade"),
):
    """
    Parameters
    ----------
    model             : trained MIL model
    test_case_dict    : {case_id: {stain: [[patch_path, ...], ...]}}
    test_label_map    : {case_id: int}
    case_level_preds  : {case_id: int}  — from trainer.evaluate() / predictions.csv
    embeddings_dir    : directory with .pt embedding files
    output_dir        : root run directory; outputs go in patch_level_analysis/
    device            : torch.device
    class_names       : ("Benign", "HighGrade")
    """
    class_names   = list(class_names)
    benign_name   = class_names[0]
    hg_name       = class_names[1]
    hg_idx        = 1

    out_dir       = os.path.join(output_dir, "patch_level_analysis")
    img_root      = os.path.join(out_dir, "misclassified_patches")
    os.makedirs(out_dir,  exist_ok=True)
    os.makedirs(img_root, exist_ok=True)

    model.eval()
    classifier_linear = _get_classifier_linear(model)

    print("\n" + "=" * 60)
    print("PATCH-LEVEL PREDICTION ANALYSIS")
    print("=" * 60)
    print(f"  Classes        : {class_names}")
    print(f"  # test cases   : {len(test_case_dict)}")
    print(f"  Embeddings dir : {embeddings_dir}")
    print(f"  Output dir     : {out_dir}")
    print()

    records  = []
    records_patch = []
    all_misclassified = []   # global list across all cases for top-10 grid
    iterator = (tqdm(test_case_dict.items(), desc="  Cases", unit="case")
                if TQDM else test_case_dict.items())

    for case_id, stain_dict in iterator:
        gt_label   = test_label_map[case_id]
        case_pred  = case_level_preds.get(case_id)
        gt_name    = class_names[gt_label]   if gt_label  < len(class_names) else str(gt_label)
        pred_name  = class_names[case_pred]  if case_pred is not None and case_pred < len(class_names) else str(case_pred)
        is_fp      = (gt_label == 0) and (case_pred == hg_idx)   # benign GT, HG pred
        is_fn      = (gt_label == hg_idx) and (case_pred == 0)  # HG GT, benign pred

        # Load embeddings + original file paths
        embeddings, patch_paths = _load_embeddings_with_paths(stain_dict, embeddings_dir)

        if embeddings is None:
            print(f"  WARNING: no embeddings for {case_id} — skipping")
            continue

        probs, preds = _predict_patches(model, embeddings, classifier_linear, device)

        for i in range(len(preds)):
            records_patch.append({
                "case_id": case_id,
                "patch_path": patch_paths[i],
                "gt_label": gt_label,
                "case_pred": case_pred,
                "patch_pred": preds[i].item(),
                "prob_benign": probs[i, 0].item(),
                "prob_highgrade": probs[i, 1].item(),
            })

        n_total = len(preds)
        counts  = {cn: int((preds == i).sum()) for i, cn in enumerate(class_names)}
        pcts    = {cn: 100.0 * counts[cn] / n_total for cn in class_names}

        records.append({
            "case_id"          : case_id,
            "gt_label"         : gt_label,
            "gt_class"         : gt_name,
            "case_pred_label"  : case_pred,
            "case_pred_class"  : pred_name,
            "case_correct"     : (case_pred == gt_label) if case_pred is not None else None,
            "false_positive"   : is_fp,
            "false_negative"   : is_fn,
            "n_patches"        : n_total,
            **{f"n_pred_{cn}"   : counts[cn] for cn in class_names},
            **{f"pct_pred_{cn}" : pcts[cn]   for cn in class_names},
        })

        # ---- Collect globally misclassified patches (patch pred != case GT) -
        # wrong_mask = (preds != gt_label)
        wrong_mask = (preds != gt_label) if is_fp else torch.zeros(len(preds), dtype=torch.bool)
        wrong_idxs = wrong_mask.nonzero(as_tuple=True)[0]

        for idx in wrong_idxs:
            idx       = idx.item()
            pred_cls  = preds[idx].item()
            confidence= probs[idx, pred_cls].item()   # confidence in the wrong class
            all_misclassified.append({
                "case_id"    : case_id,
                "gt_label"   : gt_label,
                "gt_class"   : gt_name,
                "patch_pred" : pred_cls,
                "pred_class" : class_names[pred_cls] if pred_cls < len(class_names) else str(pred_cls),
                "confidence" : confidence,
                "patch_path" : patch_paths[idx],
            })

        # ---- Save HG-predicted patch images for false-positive cases ------
        if is_fp:
            _save_misclassified_patches(
                preds, probs, patch_paths,
                target_pred_idx = hg_idx,
                target_pred_name= hg_name,
                case_id         = case_id,
                img_root        = img_root,
            )
            _plot_top10_fp_case_grid(
                preds,
                probs,
                patch_paths,
                case_id,
                gt_name,
                hg_name,
                out_dir,
                top_k=10
            )
            msg = (f"  FALSE POSITIVE  {case_id}  —  "
                   f"{counts[hg_name]}/{n_total} patches pred {hg_name}  "
                   f"({pcts[hg_name]:.1f}%)")
            tqdm.write(msg) if TQDM else print(msg)

        # ---- Save benign-predicted patch images for false-negative cases ---
        if is_fn:
            _save_misclassified_patches(
                preds, probs, patch_paths,
                target_pred_idx = 0,
                target_pred_name= benign_name,
                case_id         = case_id,
                img_root        = img_root,
            )
            msg = (f"  FALSE NEGATIVE  {case_id}  —  "
                   f"{counts[benign_name]}/{n_total} patches pred {benign_name}  "
                   f"({pcts[benign_name]:.1f}%)")
            tqdm.write(msg) if TQDM else print(msg)

    # -----------------------------------------------------------------------
    # DataFrame, summary, outputs
    # -----------------------------------------------------------------------
    df = pd.DataFrame(records)
    patch_df = pd.DataFrame(records_patch)

    patch_csv_path = os.path.join(out_dir, "case_patch_level_predictions.csv")
    patch_df.to_csv(patch_csv_path, index=False)

    print(f"Saved patch-level CSV → {patch_csv_path}")
    
    if df.empty:
        print("  WARNING: no cases analysed.")
        return df

    _print_summary(df, class_names)
    _save_csv(df, out_dir)
    _plot_stacked_bars(df, class_names, out_dir)
    _plot_hg_histogram(df, class_names, out_dir)
    _plot_top10_misclassified_grid(all_misclassified, class_names, out_dir)

    print(f"\n  Patch images for misclassified cases → {img_root}/")
    print("  Patch-level analysis complete.")
    return df



# ---------------------------------------------------------------------------
# Shared image-saving helper
# ---------------------------------------------------------------------------

def _save_misclassified_patches(preds, probs, patch_paths,
                                 target_pred_idx, target_pred_name,
                                 case_id, img_root):
    """
    Copy patch images where preds == target_pred_idx into
    img_root/<case_id>/<error_type>/, sorted by descending confidence.
    """
    indices  = (preds == target_pred_idx).nonzero(as_tuple=True)[0]
    conf     = probs[indices, target_pred_idx]
    order    = conf.argsort(descending=True)

    case_img_dir = os.path.join(img_root, str(case_id))
    os.makedirs(case_img_dir, exist_ok=True)

    n_saved = 0
    for rank, idx_in_group in enumerate(order):
        patch_idx  = indices[idx_in_group].item()
        patch_path = patch_paths[patch_idx]
        confidence = conf[idx_in_group].item()

        if not os.path.exists(patch_path):
            continue

        dst_name = (f"rank{rank+1:03d}_conf{confidence:.3f}_"
                    f"{os.path.basename(patch_path)}")
        shutil.copy(patch_path, os.path.join(case_img_dir, dst_name))
        n_saved += 1

    return n_saved


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def _print_summary(df, class_names):
    hg_name     = class_names[1]
    benign_name = class_names[0]
    col         = f"pct_pred_{hg_name}"

    print("\n" + "-" * 60)
    print("  PATCH-LEVEL SUMMARY")
    print("-" * 60)
    for lbl, name in enumerate(class_names):
        sub = df[df["gt_label"] == lbl]
        if sub.empty:
            continue
        print(f"  GT={name} ({len(sub)} cases) | "
              f"% {hg_name} patches:  "
              f"mean={sub[col].mean():.1f}%  "
              f"median={sub[col].median():.1f}%  "
              f"max={sub[col].max():.1f}%")

    for error_type, flag_col, gt_name, pred_name, sort_col in [
        ("False Positives", "false_positive", benign_name, hg_name,     f"pct_pred_{hg_name}"),
        ("False Negatives", "false_negative", hg_name,     benign_name, f"pct_pred_{benign_name}"),
    ]:
        subset = df[df[flag_col] == True]
        print(f"\n  {error_type}  (GT={gt_name}, pred={pred_name}):")
        if subset.empty:
            print(f"    None found.")
            continue
        print(f"  {'Case ID':<22} {'% Benign':>10} {'% HG':>8} {'# patches':>10}")
        print("  " + "-" * 54)
        for _, r in subset.sort_values(sort_col, ascending=False).iterrows():
            print(f"  {str(r['case_id']):<22}"
                  f"  {r[f'pct_pred_{benign_name}']:>8.1f}%"
                  f"  {r[f'pct_pred_{hg_name}']:>6.1f}%"
                  f"  {int(r['n_patches']):>9}")


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def _save_csv(df, out_dir):
    path = os.path.join(out_dir, "patch_level_predictions.csv")
    df.to_csv(path, index=False)
    print(f"\n  Saved → {path}")

    for flag_col, fname in [
        ("false_positive", "false_positive_cases.csv"),
        ("false_negative", "false_negative_cases.csv"),
    ]:
        subset = df[df[flag_col] == True]
        if not subset.empty:
            p = os.path.join(out_dir, fname)
            subset.to_csv(p, index=False)
            print(f"  Saved → {p}")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _plot_stacked_bars(df, class_names, out_dir):
    benign_name = class_names[0]
    hg_name     = class_names[1]
    col_hg      = f"pct_pred_{hg_name}"
    col_bn      = f"pct_pred_{benign_name}"

    plot_df = df.sort_values(col_hg, ascending=True).reset_index(drop=True)
    n, y    = len(plot_df), np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.32)))
    ax.barh(y, plot_df[col_bn], color="#4C9BE8", label=benign_name)
    ax.barh(y, plot_df[col_hg], left=plot_df[col_bn], color="#E84C4C", label=hg_name)

    labels = []
    for _, row in plot_df.iterrows():
        if row["false_positive"]:
            marker = "  FP"
        elif row["false_negative"]:
            marker = "  FN"
        else:
            marker = ""
        abbrev = "HG" if row["gt_class"] == "HighGrade" else "B"
        labels.append(f"{row['case_id']}  [{abbrev}]{marker}")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("% Patches")
    ax.set_title("Patch-Level Class Distribution per Test Case\n(FP = false positive, FN = false negative)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "patch_distribution_per_case.png"), dpi=300)
    plt.close()


def _plot_hg_histogram(df, class_names, out_dir):
    hg_name = class_names[1]
    col     = f"pct_pred_{hg_name}"
    bins    = np.linspace(0, 100, 21)
    colors  = ["#4C9BE8", "#E84C4C"]

    fig, axes = plt.subplots(1, len(class_names),
                              figsize=(6 * len(class_names), 4))
    axes = list(axes) if len(class_names) > 1 else [axes]

    for ax, lbl, name in zip(axes, range(len(class_names)), class_names):
        sub = df[df["gt_label"] == lbl][col]
        if sub.empty:
            ax.set_title(f"GT = {name} (0 cases)")
            continue
        ax.hist(sub, bins=bins, color=colors[lbl], edgecolor="white", linewidth=0.5)
        ax.set_title(f"GT = {name}  ({len(sub)} cases)")
        ax.set_xlabel(f"% patches predicted {hg_name}")
        ax.set_ylabel("# Cases")

    plt.suptitle(f"Distribution of {hg_name}-predicted patch fraction by GT class",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "patch_hg_pct_histogram.png"), dpi=300)
    plt.close()

# ---------------------------------------------------------------------------
# Top-10 most confidently misclassified patches — summary grid
# ---------------------------------------------------------------------------

def _plot_top10_misclassified_grid(all_misclassified, class_names, out_dir=None):
    """
    Build a grid showing the 10 most confidently misclassified patches
    across all cases.

    Images are loaded from:
        misclassified_patches/<filename>

    all_misclassified must contain dicts with:
        case_id
        patch_path   (original full path OR filename)
        confidence
        gt_class
        pred_class
    """

    if not all_misclassified:
        print("No misclassified patches found — skipping.")
        return

    # Sort globally by confidence (descending)
    ranked = sorted(all_misclassified,
                    key=lambda x: x["confidence"],
                    reverse=True)

    top10 = ranked[:10]

    n      = len(top10)
    ncols  = min(5, n)
    nrows  = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 3.2, nrows * 3.8)
    )

    # Normalize axes to 2D list
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [list(axes)]
    elif ncols == 1:
        axes = [[ax] for ax in axes]
    else:
        axes = [list(row) for row in axes]

    for rank, entry in enumerate(top10):

        row = rank // ncols
        col = rank % ncols
        ax  = axes[row][col]

        patch_path = entry["patch_path"]
        
        if os.path.exists(patch_path):
            try:
                img = Image.open(patch_path).convert("RGB")
                ax.imshow(img)
            except Exception:
                ax.set_facecolor("#cccccc")
                ax.text(0.5, 0.5, "load error",
                        ha="center", va="center",
                        transform=ax.transAxes, fontsize=9)
        else:
            ax.set_facecolor("#cccccc")
            ax.text(0.5, 0.5, "not found",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)

        ax.set_xticks([])
        ax.set_yticks([])

        # Remove borders
        for spine in ax.spines.values():
            spine.set_visible(False)

        
        if entry['gt_class'] != entry['pred_class']:
            if entry['gt_class'] == "Benign":
                type_error = ""
            elif entry['gt_class'] == "HighGrade":
                type_error = ""
        
        caption = (
            f"#{rank+1}  conf={entry['confidence']:.3f}\n"
            f"GT: {'HG' if entry['gt_class'] == 'HighGrade' else 'B'} → Pred: {'HG' if entry['pred_class'] == 'HighGrade' else 'B'}\n"
            f"Case: {entry['case_id']}"
        )

        ax.text(0.5, -0.05,
            # 0.5, -0.18,
            caption,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12
        )

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r][c].set_visible(False)

    fig.suptitle(
        "Top 10 Most Confidently Misclassified Patches\n"
        "(Patch prediction ≠ Case ground truth)",
        fontsize=14,
        y=1.0
    )

    plt.subplots_adjust(hspace=0.08, wspace=0.05)
    # plt.tight_layout()
    save_path = os.path.join(out_dir, "top10_misclassified_global.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)  

    
# ---------------------------------------------------------------------------
# Top-10 HG patches per FALSE POSITIVE case (one grid per case)
# ---------------------------------------------------------------------------
def _plot_top10_fp_case_grid(
    preds,
    probs,
    patch_paths,
    case_id,
    gt_name,
    pred_name,
    out_dir=None,
    top_k=10
):
    """
    For ONE false-positive case:

    - Select patches predicted HighGrade
    - Rank by HG confidence
    - Plot TOP-K
    - Load images from misclassified_patches/
    """

    # Indices where patch predicted HighGrade (class 1)
    hg_indices = (preds == 1).nonzero(as_tuple=True)[0]

    if len(hg_indices) == 0:
        print(f"No HG patches for case {case_id}")
        return

    confidences = probs[hg_indices, 1]
    order = torch.argsort(confidences, descending=True)

    top_indices = hg_indices[order[:top_k]]
    top_conf    = confidences[order[:top_k]]

    n = len(top_indices)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 3.2, nrows * 3.8)
    )

    # Normalize axes
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [list(axes)]
    elif ncols == 1:
        axes = [[ax] for ax in axes]
    else:
        axes = [list(row) for row in axes]

    for rank in range(n):

        row = rank // ncols
        col = rank % ncols
        ax  = axes[row][col]

        patch_idx = top_indices[rank].item()

        patch_path = patch_paths[patch_idx]
        
        conf = top_conf[rank].item()

        if os.path.exists(patch_path):
            try:
                img = Image.open(patch_path).convert("RGB")
                ax.imshow(img)
            except Exception:
                ax.set_facecolor("#cccccc")
                ax.text(0.5, 0.5, "load error",
                        ha="center", va="center",
                        transform=ax.transAxes, fontsize=9)
        else:
            ax.set_facecolor("#cccccc")
            ax.text(0.5, 0.5, "not found",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)

        ax.set_xticks([])
        ax.set_yticks([])

        # Remove borders
        for spine in ax.spines.values():
            spine.set_visible(False)

        caption = (
            f"#{rank+1}  conf={conf:.3f}"
        )

        ax.text(
            0.5, -0.05, #-0.18,
            caption,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12
        )

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r][c].set_visible(False)

    fig.suptitle(
        f"Top {n} Most Confident FP Patches\n"
        f"FALSE POSITIVE Case: {case_id}",
        fontsize=14,
        y=1.0
    )

    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    # plt.tight_layout()
    save_path = os.path.join(out_dir, f"top{top_k}_fp_case_{case_id}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)