"""
Analyze how often we exceed:
  - max_slices_per_stain (e.g., 5)
  - per_slice_cap (e.g., 500 patches)
using the *raw* case_dict built from all patches.
"""

from typing import Dict, List, Tuple

from config import DATA_PATHS
from data_utils import (
    load_labels,
    get_all_patch_files,
    group_patches_by_slice,
    build_slice_to_class_map,
    build_case_dict,
)


def build_full_case_dict():
    """
    Reconstruct case_dict and label_map for ALL slices/patches
    (no train/val/test split).
    """
    # 1. Load labels
    labels = load_labels()  # uses DATA_PATHS['labels_csv'] by default

    # 2. Get all patch filenames in the patches_dir
    all_files = get_all_patch_files()  # uses DATA_PATHS['patches_dir']

    # 3. Group patches into slices: {(case_id, slice_id): [patch_paths]}
    patches = group_patches_by_slice(
        all_files,
        root_dir=DATA_PATHS["patches_dir"],
    )

    # 4. Map (case_id, slice_id) -> class label (0/1)
    slice_to_class = build_slice_to_class_map(patches, labels)

    # 5. Build case_dict: {case_id: {stain: [[patches_of_slice1], ...]}}
    #    Use ALL slices (no split) by passing list(patches.keys())
    slice_list = list(patches.keys())
    case_dict, label_map = build_case_dict(slice_list, patches, slice_to_class)

    return case_dict, label_map


def label_to_str(label):
    """Convert numeric label to human-readable string."""
    if label == 0:
        return "benign"
    if label == 1:
        return "high-grade"
    return "unknown"


def analyze_caps(
    case_dict: Dict,
    label_map: Dict,
    slice_cap: int = 500,
    max_slices_per_stain: int = 5,
):
    """
    Count:
      - how many stain entries have > max_slices_per_stain slices
      - how many slice entries have > slice_cap patches

    Prints ALL such entries, with benign/high-grade info per case.
    """
    stains_over_cap = []
    slices_over_cap = []

    for case_id, stain_map in case_dict.items():
        for stain, slice_lists in stain_map.items():
            # ---- stains with too many slices ----
            num_slices = len(slice_lists)
            if num_slices > max_slices_per_stain:
                stains_over_cap.append((case_id, stain, num_slices))

            # ---- individual slices with too many patches ----
            for slice_idx, patch_paths in enumerate(slice_lists):
                num_patches = len(patch_paths)
                if num_patches > slice_cap:
                    slices_over_cap.append(
                        (case_id, stain, slice_idx, num_patches)
                    )

    print("\n========== CAP ANALYSIS ==========\n")

    # ---- summary counts ----
    print(f"Stain instances with > {max_slices_per_stain} slices: {len(stains_over_cap)}")
    print(f"Slice instances with > {slice_cap} patches: {len(slices_over_cap)}")

    # ---- detailed listing: ALL stains over slice cap ----
    print("\n--- Stains with > max_slices_per_stain ---")
    if not stains_over_cap:
        print("None.")
    else:
        for cid, stain, n_slices in stains_over_cap:
            label = label_map.get(cid, None)
            label_str = label_to_str(label)
            print(f"Case {cid} ({label_str}) | Stain {stain} | #slices = {n_slices}")

    # ---- detailed listing: ALL slices over patch cap ----
    print(f"\n--- Slices with > {slice_cap} patches ---")
    if not slices_over_cap:
        print("None.")
    else:
        for cid, stain, s_idx, n_patches in slices_over_cap:
            label = label_map.get(cid, None)
            label_str = label_to_str(label)
            print(
                f"Case {cid} ({label_str}) | Stain {stain} | "
                f"Slice index {s_idx} | #patches = {n_patches}"
            )

    print("\n==================================\n")


if __name__ == "__main__":
    case_dict, label_map = build_full_case_dict()
    # Use your current training caps (500 patches / slice, 5 slices / stain)
    analyze_caps(case_dict, label_map, slice_cap=500, max_slices_per_stain=5)
