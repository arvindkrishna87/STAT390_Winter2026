#!/usr/bin/env python3
"""
make_grouped_folds.py

Creates multiple train/val/test splits where EACH benign family appears in TEST
in at least one fold, and families are never split across train/val/test.

Outputs:
  data_splits/data_splits_fold00.npz
  data_splits/data_splits_fold01.npz
  ...

Assumes:
- load_labels / get_all_patch_files / group_patches_by_slice / build_slice_to_class_map exist.
- save_data_splits exists (utils.py)
"""

import os
import json
import argparse
import numpy as np

from config import DATA_PATHS, TRAINING_CONFIG, SPLIT_CONFIG
from data_utils import (
    load_labels, get_all_patch_files, group_patches_by_slice, build_slice_to_class_map
)
from utils import save_data_splits


def load_families(path: str):
    with open(path, "r") as f:
        obj = json.load(f)
    fams = [sorted(list(map(int, fam))) for fam in obj.get("benign_families", [])]
    # remove empties + ensure unique
    fams = [fam for fam in fams if len(fam) > 0]
    return fams


def build_case_to_label(slice_to_class: dict):
    # slice_to_class keys: (case_id, slice_id) -> y
    case_to_label = {}
    for (case_id, _slice_id), y in slice_to_class.items():
        case_to_label.setdefault(int(case_id), int(y))
    return case_to_label


def build_units(case_ids, families):
    """
    Turn case IDs into "units":
    - each family is a unit (list of case_ids)
    - all other cases are singleton units
    """
    case_ids = set(case_ids)
    used = set()

    units = []
    family_units = []

    for fam in families:
        fam_set = set(fam)
        present = sorted(list(fam_set.intersection(case_ids)))
        if len(present) == 0:
            continue
        # If partially present, that means your dataset is missing some of the family members.
        # That’s risky for leakage assumptions — fail loudly.
        if len(present) != len(set(fam)):
            raise RuntimeError(
                f"Family {fam} partially present in dataset: present={present}. "
                "Fix benign_families.json to match dataset."
            )
        family_units.append(present)
        used |= fam_set

    # singleton units for remaining
    for c in sorted(case_ids - used):
        units.append([c])

    # final list of units = family units + singleton units
    return family_units + units, family_units


def unit_label(unit, case_to_label):
    """
    Unit label:
    - benign family units should be all benign (0). If not, fail.
    - singleton = its label.
    """
    labels = [case_to_label[c] for c in unit]
    if len(set(labels)) != 1:
        raise RuntimeError(f"Unit has mixed labels (should not happen): unit={unit}, labels={labels}")
    return labels[0]


def sample_units_stratified(rng, units, case_to_label, n_cases_target, force_include_units=None):
    """
    Pick units until we hit target number of CASES (not units).
    Attempts to maintain class balance by alternating class picks based on remaining availability.

    force_include_units: list of units that must be included first (e.g., a benign family in TEST).
    """
    if force_include_units is None:
        force_include_units = []

    picked = []
    picked_cases = set()

    # Add forced units
    for u in force_include_units:
        for c in u:
            if c in picked_cases:
                raise RuntimeError(f"Duplicate forced case {c} across forced units.")
        picked.append(u)
        picked_cases |= set(u)

    # Remaining units pool
    remaining_units = [u for u in units if set(u).isdisjoint(picked_cases)]

    # Split remaining by label
    benign_units = [u for u in remaining_units if unit_label(u, case_to_label) == 0]
    high_units   = [u for u in remaining_units if unit_label(u, case_to_label) == 1]

    rng.shuffle(benign_units)
    rng.shuffle(high_units)

    # Greedy fill
    def total_cases(p):
        return sum(len(u) for u in p)

    while total_cases(picked) < n_cases_target and (benign_units or high_units):
        # choose which class to draw from based on remaining capacity
        # (simple heuristic: keep ratios near global)
        if not benign_units:
            u = high_units.pop()
        elif not high_units:
            u = benign_units.pop()
        else:
            # alternate randomly but weighted by remaining counts
            if rng.random() < (len(high_units) / (len(high_units) + len(benign_units))):
                u = high_units.pop()
            else:
                u = benign_units.pop()

        # don’t exceed target too much (but allow a small overshoot because units are atomic)
        if total_cases(picked) + len(u) <= n_cases_target or total_cases(picked) < n_cases_target:
            picked.append(u)
            picked_cases |= set(u)

    return picked, picked_cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", type=str, default=DATA_PATHS["labels_csv"])
    ap.add_argument("--patches_dir", type=str, default=DATA_PATHS["patches_dir"])
    ap.add_argument("--families_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="data_splits")
    ap.add_argument("--seed", type=int, default=TRAINING_CONFIG["random_state"])
    ap.add_argument("--train_ratio", type=float, default=float(SPLIT_CONFIG["train_ratio"]))
    ap.add_argument("--val_ratio", type=float, default=float(SPLIT_CONFIG["val_ratio"]))
    ap.add_argument("--test_ratio", type=float, default=float(SPLIT_CONFIG["test_ratio"]))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    families = load_families(args.families_json)

    labels = load_labels(args.labels_csv)
    all_files = get_all_patch_files(args.patches_dir)
    patches = group_patches_by_slice(all_files, args.patches_dir)
    slice_to_class = build_slice_to_class_map(patches, labels)

    case_to_label = build_case_to_label(slice_to_class)
    case_ids = sorted(case_to_label.keys())
    N_total = len(case_ids)

    # Build atomic units
    all_units, family_units = build_units(case_ids, families)

    # Sanity: families should be benign
    for fam in family_units:
        y = unit_label(fam, case_to_label)
        if y != 0:
            raise RuntimeError(f"Family unit is not benign (expected 0): {fam} has label {y}")

    # Decide fold count = number of families
    K = len(family_units)
    if K == 0:
        raise RuntimeError("No benign families found. families_json may be empty or not matching dataset.")

    # Split sizes by CASE count
    n_test = max(1, int(round(args.test_ratio * N_total)))
    n_val  = max(1, int(round(args.val_ratio * N_total)))
    n_train = N_total - n_test - n_val
    if n_train <= 0:
        raise RuntimeError("Invalid split ratios; train ended up <= 0.")

    os.makedirs(args.out_dir, exist_ok=True)

    for k in range(K):
        forced_test_unit = family_units[k]

        # TEST: force include family unit k, then fill to n_test
        test_units, test_cases = sample_units_stratified(
            rng, all_units, case_to_label, n_test, force_include_units=[forced_test_unit]
        )

        # Remaining after test
        remaining_units = [u for u in all_units if set(u).isdisjoint(test_cases)]

        # VAL: sample from remaining
        val_units, val_cases = sample_units_stratified(
            rng, remaining_units, case_to_label, n_val
        )

        # TRAIN: everything else
        train_cases = set(case_ids) - test_cases - val_cases

        # Family constraint check: no family partially in test
        for fam in family_units:
            fam_set = set(fam)
            inter = fam_set.intersection(test_cases)
            if inter and inter != fam_set:
                raise RuntimeError(f"Leakage: family split across test and non-test: fam={fam}, in_test={sorted(inter)}")

        # Save
        fold_dir = os.path.join(args.out_dir, f"fold{k:02d}")
        os.makedirs(fold_dir, exist_ok=True)
        save_data_splits(
            sorted(train_cases),
            sorted(val_cases),
            sorted(test_cases),
            save_dir=fold_dir
        )
        print(f"[OK] Saved fold{k:02d} splits to {fold_dir}/data_splits.npz")
        print(f"     TEST forced family: {forced_test_unit}")


if __name__ == "__main__":
    main()