#!/usr/bin/env python3
"""
make_splits.py

Creates a NEW train/val/test split (by CASE), prints benign vs high-grade CASE IDs
in each split, and saves a data_splits.npz compatible with:

  python main.py --load_splits /path/to/data_splits.npz

ENFORCED:
- Split by CASE (no leakage).
- Exclude specified cases from TEST.
- Keep split sizes close to target ratios (default 60/20/20) and NON-EMPTY.
- Match benign/high-grade proportions across Train/Val/Test as tightly as possible.

Why not "exact" always?
Exact equality of proportions can be mathematically impossible unless split sizes
are compatible with the reduced fraction of (H_total / N_total). In those cases,
forcing exact equality can collapse val/test to 0. This script instead enforces
a near-exact rational approximation with a small denominator to keep all splits
non-empty and proportions effectively equal.

Labels:
  0 = benign (Class 1.0)
  1 = high-grade (Class 3.0 or 4.0)
"""

import os
import argparse
from fractions import Fraction

import numpy as np

from config import DATA_PATHS, TRAINING_CONFIG, SPLIT_CONFIG
from data_utils import (
    load_labels,
    get_all_patch_files,
    group_patches_by_slice,
    build_slice_to_class_map,
    build_case_dict,
    report_no_leak,
)
from utils import save_data_splits

import json


EXCLUDE_FROM_TEST_DEFAULT = {46}

def load_benign_families(path: str) -> list[list[int]]:
    with open(path, "r") as f:
        obj = json.load(f)
    groups = obj.get("benign_families", None)
    if groups is None:
        raise ValueError("Expected JSON with key 'benign_families'")
    groups = [sorted([int(x) for x in g]) for g in groups]
    return groups

def _split_counts(case_ids, case_to_label):
    n = len(case_ids)
    h = sum(1 for c in case_ids if case_to_label[c] == 1)
    b = n - h
    return b, h, n


def _print_split(name, case_ids, case_to_label):
    benign = sorted([c for c in case_ids if case_to_label[c] == 0])
    high = sorted([c for c in case_ids if case_to_label[c] == 1])
    total = len(case_ids)
    ratio = (len(high) / total) if total else 0.0

    print("\n" + "=" * 80)
    print(f"{name.upper()} SPLIT")
    print("=" * 80)
    print(f"Total cases:      {total}")
    print(f"Benign (0):       {len(benign)}")
    print(f"High-grade (1):   {len(high)}")
    print(f"High-grade ratio: {ratio:.6f}")

    print("\nBenign CASE IDs:")
    print(benign)
    print("\nHigh-grade CASE IDs:")
    print(high)


def _target_split_sizes(N_total, train_ratio, val_ratio, test_ratio):
    """
    Rounded targets with non-empty val/test when possible.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.")

    N_test = int(round(test_ratio * N_total))
    N_val = int(round(val_ratio * N_total))
    N_train = N_total - N_test - N_val

    # Keep non-empty val/test if possible (N_total >= 3)
    if N_total >= 3:
        if N_test == 0:
            N_test = 1
        if N_val == 0:
            N_val = 1
        N_train = N_total - N_test - N_val
        if N_train <= 0:
            # emergency fallback: force train at least 1
            N_train = 1
            # redistribute
            rem = N_total - N_train
            # split remaining roughly evenly between val/test
            N_val = max(1, rem // 2)
            N_test = rem - N_val

    return N_train, N_val, N_test


def _allocate_class_counts_with_common_ratio(
    N_train, N_val, N_test,
    H_total, N_total,
    max_den=None,
):
    """
    Allocate H_train/H_val/H_test so that H_i / N_i are as equal as possible.

    Strategy:
    - Use global ratio r = H_total / N_total
    - Approximate r with a rational p/q with small denominator (limit_denominator)
      so that p/q is feasible across typical split sizes.
    - Use p/q to compute desired highs: round(N_i * p/q), then adjust to sum to H_total.
    """
    if max_den is None:
        # Denominator should be small enough to avoid collapsing splits,
        # but large enough to preserve ratio fidelity.
        max_den = max(5, min(N_train, N_val, N_test, 50))

    r = Fraction(H_total, N_total)
    r_approx = r.limit_denominator(max_den)  # p/q small
    p, q = r_approx.numerator, r_approx.denominator

    # Initial rounded allocations
    H_train = int(round(N_train * p / q))
    H_val = int(round(N_val * p / q))
    H_test = int(round(N_test * p / q))

    # Clip to feasible bounds
    H_train = max(0, min(H_train, N_train))
    H_val = max(0, min(H_val, N_val))
    H_test = max(0, min(H_test, N_test))

    # Adjust totals to match H_total exactly
    Hs = [H_train, H_val, H_test]
    Ns = [N_train, N_val, N_test]

    def ratio_err(i):
        # how far split i's ratio is from r_approx
        return abs(Fraction(Hs[i], Ns[i]) - r_approx)

    current = sum(Hs)
    diff = H_total - current

    # If we need to add highs: increment splits where it least increases ratio error
    while diff > 0:
        # candidate splits where H < N (can add)
        candidates = [i for i in range(3) if Hs[i] < Ns[i]]
        if not candidates:
            break
        # choose split that keeps ratio closest
        best_i = min(
            candidates,
            key=lambda i: abs(Fraction(Hs[i] + 1, Ns[i]) - r_approx)
        )
        Hs[best_i] += 1
        diff -= 1

    # If we need to remove highs: decrement splits where it least increases ratio error
    while diff < 0:
        candidates = [i for i in range(3) if Hs[i] > 0]
        if not candidates:
            break
        best_i = min(
            candidates,
            key=lambda i: abs(Fraction(Hs[i] - 1, Ns[i]) - r_approx)
        )
        Hs[best_i] -= 1
        diff += 1

    if sum(Hs) != H_total:
        raise RuntimeError(
            f"Could not allocate highs to match H_total exactly. "
            f"Allocated={sum(Hs)}, required={H_total}. "
            f"Try increasing max_den or revisiting constraints."
        )

    return (Hs[0], Hs[1], Hs[2], r, r_approx)


def split_by_case_with_constraints(
    slice_to_class: dict,
    exclude_from_test: set,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    max_ratio_den: int | None = None,
    benign_families: list[list[int]] | None = None,
    test_benign_family_idx: int | None = None,
):
    rng = np.random.default_rng(seed)

    # Build case -> label (binary)
    case_to_label = {}
    for (case_id, _slice_id), y in slice_to_class.items():
        y = int(y)
        case_to_label.setdefault(case_id, y)

    case_ids = sorted(case_to_label.keys())
    N_total = len(case_ids)
    if N_total == 0:
        raise RuntimeError("No cases found to split.")

    H_total = sum(1 for c in case_ids if case_to_label[c] == 1)
    B_total = N_total - H_total
    if H_total == 0 or B_total == 0:
        raise RuntimeError(
            "Cannot enforce matched proportions if only one class exists.\n"
            f"Counts: benign={B_total}, high={H_total}"
        )

    # Targets (non-empty)
    N_train, N_val, N_test = _target_split_sizes(N_total, train_ratio, val_ratio, test_ratio)

    # Allocate per-split class counts to match global ratio closely and exactly match totals
    H_train, H_val, H_test, r_exact, r_approx = _allocate_class_counts_with_common_ratio(
        N_train, N_val, N_test, H_total, N_total, max_den=max_ratio_den
    )
    B_train, B_val, B_test = N_train - H_train, N_val - H_val, N_test - H_test

    print("\nOverall case counts:")
    print(f"  Total:  {N_total}  |  Benign: {B_total}  |  High: {H_total}")
    print(f"  Exact global high ratio:   {float(r_exact):.6f}  ({r_exact.numerator}/{r_exact.denominator})")
    print(f"  Enforced approx ratio:     {float(r_approx):.6f}  ({r_approx.numerator}/{r_approx.denominator})")
    print("\nTarget split sizes (non-empty) and required class counts:")
    print(f"  Train: N={N_train}  => benign={B_train}, high={H_train}")
    print(f"  Val:   N={N_val}    => benign={B_val}, high={H_val}")
    print(f"  Test:  N={N_test}   => benign={B_test}, high={H_test}")

    # Exclusion for test
    exclude_from_test = set(int(x) for x in exclude_from_test)
    eligible_for_test = [c for c in case_ids if c not in exclude_from_test]

    elig_benign = [c for c in eligible_for_test if case_to_label[c] == 0]
    elig_high = [c for c in eligible_for_test if case_to_label[c] == 1]

    if len(elig_benign) < B_test or len(elig_high) < H_test:
        raise RuntimeError(
            "Not enough eligible cases to build TEST with required benign/high counts.\n"
            f"Required TEST: benign={B_test}, high={H_test}\n"
            f"Eligible TEST: benign={len(elig_benign)}, high={len(elig_high)}\n"
            "Fix by reducing exclusions or loosening the ratio denominator (max_ratio_den)."
        )

   # -----------------------------
# Sample TEST with exact counts
# BUT: optionally force a benign family to be included in test (all-or-none)
# -----------------------------
    forced_test_benign = []

    if benign_families is not None and test_benign_family_idx is not None:
        if not (0 <= test_benign_family_idx < len(benign_families)):
            raise ValueError(f"test_benign_family_idx out of range: {test_benign_family_idx}")

        forced_test_benign = list(map(int, benign_families[test_benign_family_idx]))

        # Sanity: forced cases must exist + be benign
        missing = [c for c in forced_test_benign if c not in case_to_label]
        if missing:
            raise RuntimeError(f"Forced benign family contains unknown cases: {missing}")

        not_benign = [c for c in forced_test_benign if case_to_label[c] != 0]
        if not_benign:
            raise RuntimeError(f"Forced benign family contains non-benign cases: {not_benign}")

        # Must be eligible for test (i.e., not excluded)
        blocked = [c for c in forced_test_benign if c in exclude_from_test]
        if blocked:
            raise RuntimeError(
                f"Forced benign family cases are excluded from test via exclude_from_test: {blocked}\n"
                "Remove them from exclude_from_test when running benign-family testing."
            )

        # Need enough benign slots in test to fit the whole family
        if len(forced_test_benign) > B_test:
            raise RuntimeError(
                f"Test split benign target is B_test={B_test}, but forced family size={len(forced_test_benign)}.\n"
                "Increase test_ratio or loosen ratio constraints."
            )

    # Build benign test list: forced family + random other benign (if needed)
    elig_benign_set = set(elig_benign)
    for c in forced_test_benign:
        if c not in elig_benign_set:
            raise RuntimeError(f"Forced case {c} is not eligible benign for test (check exclusions).")

    remaining_elig_benign = [c for c in elig_benign if c not in set(forced_test_benign)]
    need_extra = B_test - len(forced_test_benign)

    extra_test_benign = []
    if need_extra > 0:
        extra_test_benign = rng.choice(remaining_elig_benign, size=need_extra, replace=False).tolist()

    test_benign = list(forced_test_benign) + list(extra_test_benign)

    # Sample test high-grade normally
    test_high = rng.choice(elig_high, size=H_test, replace=False).tolist()

    test_cases = set(test_benign + test_high)

    # Remaining pool
    remaining = [c for c in case_ids if c not in test_cases]
    rem_benign = [c for c in remaining if case_to_label[c] == 0]
    rem_high = [c for c in remaining if case_to_label[c] == 1]

    if len(rem_benign) < B_val or len(rem_high) < H_val:
        raise RuntimeError(
            "Not enough remaining cases to build VAL with required benign/high counts.\n"
            f"Required VAL: benign={B_val}, high={H_val}\n"
            f"Remaining:    benign={len(rem_benign)}, high={len(rem_high)}"
        )

    # Sample VAL with exact counts
    val_benign = rng.choice(rem_benign, size=B_val, replace=False).tolist()
    val_high = rng.choice(rem_high, size=H_val, replace=False).tolist()
    val_cases = set(val_benign + val_high)

    # TRAIN is the rest
    train_cases = set(case_ids) - test_cases - val_cases

    # Final sanity
    if len(train_cases) != N_train or len(val_cases) != N_val or len(test_cases) != N_test:
        raise RuntimeError(
            "Final split sizes mismatch.\n"
            f"Expected train/val/test: {N_train}/{N_val}/{N_test}\n"
            f"Actual   train/val/test: {len(train_cases)}/{len(val_cases)}/{len(test_cases)}"
        )

    bad = sorted(list(test_cases.intersection(exclude_from_test)))
    if bad:
        raise RuntimeError(f"Excluded cases ended up in TEST: {bad}")

    # Ratio printout (they will be extremely close / often identical up to rounding)
    def ratio_str(cset):
        b, h, n = _split_counts(list(cset), case_to_label)
        return f"{h}/{n} = {h/n:.6f}"

    print("\nFinal achieved high-grade ratios:")
    print(f"  Train: {ratio_str(train_cases)}")
    print(f"  Val:   {ratio_str(val_cases)}")
    print(f"  Test:  {ratio_str(test_cases)}")

    # Convert to slice-level lists
    train_slices, val_slices, test_slices = [], [], []
    for (case_id, slice_id), _y in slice_to_class.items():
        if case_id in train_cases:
            train_slices.append((case_id, slice_id))
        elif case_id in val_cases:
            val_slices.append((case_id, slice_id))
        elif case_id in test_cases:
            test_slices.append((case_id, slice_id))

    return train_slices, val_slices, test_slices, case_to_label


def main():
    ap = argparse.ArgumentParser(description="Create and save train/val/test case splits for MIL training.")
    ap.add_argument("--labels_csv", type=str, default=DATA_PATHS["labels_csv"])
    ap.add_argument("--patches_dir", type=str, default=DATA_PATHS["patches_dir"])
    ap.add_argument("--seed", type=int, default=TRAINING_CONFIG["random_state"])
    ap.add_argument("--save_dir", type=str, default=".")
    ap.add_argument("--benign_families_json", type=str, default=None,
               help="Path to benign_families.json with {'benign_families': [[...], ...]}")
    ap.add_argument("--test_benign_family_idx", type=int, default=None,
               help="If set, force benign_families[idx] to be included in TEST (all together).")
    ap.add_argument(
        "--exclude_from_test",
        type=str,
        default=",".join(str(x) for x in sorted(EXCLUDE_FROM_TEST_DEFAULT)),
        help="Comma-separated case IDs to exclude from test split",
    )
    ap.add_argument("--train_ratio", type=float, default=float(SPLIT_CONFIG["train_ratio"]))
    ap.add_argument("--val_ratio", type=float, default=float(SPLIT_CONFIG["val_ratio"]))
    ap.add_argument("--test_ratio", type=float, default=float(SPLIT_CONFIG["test_ratio"]))
    ap.add_argument(
        "--max_ratio_den",
        type=int,
        default=50,
        help="Max denominator for ratio approximation (smaller => easier feasibility, larger => closer to global ratio).",
    )
    args = ap.parse_args()

    exclude_from_test = set()
    for part in args.exclude_from_test.split(","):
        part = part.strip()
        if part:
            exclude_from_test.add(int(part))

    print("=" * 80)
    print("CREATING DATA SPLITS (EXCLUDE FROM TEST + MATCHED PROPORTIONS)")
    print("=" * 80)
    print(f"labels_csv:        {args.labels_csv}")
    print(f"patches_dir:       {args.patches_dir}")
    print(f"seed:              {args.seed}")
    print(f"save_dir:          {args.save_dir}")
    print(f"ratios (targets):  train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    print(f"exclude_from_test: {sorted(list(exclude_from_test))}")
    print(f"max_ratio_den:     {args.max_ratio_den}")

    labels = load_labels(args.labels_csv)
    print(f"\nLoaded {len(labels)} label rows")

    all_files = get_all_patch_files(args.patches_dir)
    print(f"Found {len(all_files)} files in patches_dir")

    patches = group_patches_by_slice(all_files, args.patches_dir)
    print(f"Grouped into {len(patches)} slices")

    slice_to_class = build_slice_to_class_map(patches, labels)
    print(f"Mapped {len(slice_to_class)} slices to classes")

    benign_families = None
    if args.benign_families_json:
        benign_families = load_benign_families(args.benign_families_json)
        print(f"Loaded {len(benign_families)} benign families from {args.benign_families_json}")
        if args.test_benign_family_idx is not None:
            print(f"Forcing benign family idx={args.test_benign_family_idx} into TEST: {benign_families[args.test_benign_family_idx]}")

    train_slices, val_slices, test_slices, case_to_label = split_by_case_with_constraints(
        slice_to_class=slice_to_class,
        exclude_from_test=exclude_from_test,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_ratio_den=args.max_ratio_den,
        benign_families=benign_families,
        test_benign_family_idx=args.test_benign_family_idx,
    )

    print("\nSlice counts after split:")
    print(f"  Train slices: {len(train_slices)}")
    print(f"  Val slices:   {len(val_slices)}")
    print(f"  Test slices:  {len(test_slices)}")

    # Build dicts for leak checks + printing lists
    train_case_dict, train_label_map = build_case_dict(train_slices, patches, slice_to_class)
    val_case_dict, val_label_map = build_case_dict(val_slices, patches, slice_to_class)
    test_case_dict, test_label_map = build_case_dict(test_slices, patches, slice_to_class)

    print("\n" + "-" * 40)
    print("LEAK CHECK")
    print("-" * 40)
    report_no_leak(train_case_dict, val_case_dict, test_case_dict)

    # Hard assertion: excluded cases not in test
    excluded_in_test = sorted(list(set(test_case_dict.keys()).intersection(exclude_from_test)))
    if excluded_in_test:
        raise RuntimeError(f"Excluded cases found in TEST case_dict: {excluded_in_test}")
    print("\n[OK] None of the excluded cases appear in TEST.")

    # Print case lists per split
    _print_split("train", list(train_case_dict.keys()), case_to_label)
    _print_split("val", list(val_case_dict.keys()), case_to_label)
    _print_split("test", list(test_case_dict.keys()), case_to_label)

    # Save splits
    os.makedirs(args.save_dir, exist_ok=True)
    train_cases = sorted(list(train_case_dict.keys()))
    val_cases = sorted(list(val_case_dict.keys()))
    test_cases = sorted(list(test_case_dict.keys()))

    save_data_splits(train_cases, val_cases, test_cases, save_dir=args.save_dir)

    out_path = os.path.join(args.save_dir, "data_splits.npz")
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Saved: {out_path}")
    print("Use it like:")
    print(f"  python main.py --load_splits {out_path}")


if __name__ == "__main__":
    main()