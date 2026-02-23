#!/usr/bin/env python3
"""
find_renamed_cases.py

Find files in SRC that appear to have been renamed in DST where ONLY the case number changed.

CONSTRAINTS (all must hold):
1) The original filename must NOT exist in DST.
2) The renamed (candidate) filename must NOT exist in SRC.
3) For any (old_case -> new_case) rename we accept:
   max patch index for old_case in SRC == max patch index for new_case in DST
   where patch index is the integer after 'patch' and before '.png'.

Outputs a CSV with two columns:
  original_filename, renamed_filename

Only rows where the name actually changed are included.

Run:
  python find_renamed_cases.py
Optional:
  python find_renamed_cases.py --src ... --dst ... --out renamed_files.csv
"""

import os
import re
import csv
import argparse
from typing import Optional, Tuple, Dict, List, Set


CASE_RE = re.compile(r"^(case_)(\d+)(_.+)$")               # prefix, case_num, rest
PATCH_RE = re.compile(r"patch(\d+)\.png$", re.IGNORECASE)  # patch index at end


def split_case(fname: str) -> Optional[Tuple[str, str, str]]:
    """Return (prefix 'case_', case_number, rest_of_name) if it matches else None."""
    m = CASE_RE.match(fname)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def extract_patch_index(fname: str) -> Optional[int]:
    """Return patch index int if found (patch<idx>.png at end), else None."""
    m = PATCH_RE.search(fname)
    if not m:
        return None
    return int(m.group(1))


def compute_case_max_patch(files: List[str]) -> Dict[str, int]:
    """
    Compute max patch index per case number from a list of filenames.
    Only considers files that match case_<num> and contain patch<idx>.png at end.
    """
    max_by_case: Dict[str, int] = {}
    for f in files:
        parts = split_case(f)
        if parts is None:
            continue
        _, case_num, _ = parts
        idx = extract_patch_index(f)
        if idx is None:
            continue
        prev = max_by_case.get(case_num)
        if prev is None or idx > prev:
            max_by_case[case_num] = idx
    return max_by_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="/projects/e32998/patches", help="Source folder (original names)")
    parser.add_argument("--dst", default="/projects/e32998/patches_Team5", help="Destination folder (renamed names)")
    parser.add_argument("--out", default="renamed_files.csv", help="Output CSV path")
    parser.add_argument("--progress_every", type=int, default=2000,
                        help="Print progress every N source files processed")
    args = parser.parse_args()

    print(f"[INFO] Source folder:       {args.src}")
    print(f"[INFO] Destination folder:  {args.dst}")
    print(f"[INFO] Output CSV:          {args.out}")

    # List only regular files
    src_files = [f for f in os.listdir(args.src) if os.path.isfile(os.path.join(args.src, f))]
    dst_files = [f for f in os.listdir(args.dst) if os.path.isfile(os.path.join(args.dst, f))]

    print(f"[INFO] Source file count:      {len(src_files)}")
    print(f"[INFO] Destination file count: {len(dst_files)}")

    src_set: Set[str] = set(src_files)
    dst_set: Set[str] = set(dst_files)

    # Precompute max patch index per case in each folder (for constraint 3)
    print("[INFO] Computing max patch index per case in SRC...")
    src_max_patch = compute_case_max_patch(src_files)
    print(f"[INFO] SRC cases with patch indices: {len(src_max_patch)}")

    print("[INFO] Computing max patch index per case in DST...")
    dst_max_patch = compute_case_max_patch(dst_files)
    print(f"[INFO] DST cases with patch indices: {len(dst_max_patch)}")

    # Index destination files by the "rest" portion (everything after case_<num>)
    dst_by_rest: Dict[str, List[str]] = {}
    kept_dst = 0
    skipped_dst = 0
    for f in dst_files:
        parts = split_case(f)
        if parts is None:
            skipped_dst += 1
            continue
        _, _, rest = parts
        dst_by_rest.setdefault(rest, []).append(f)
        kept_dst += 1

    print(f"[INFO] Destination files parsed for case-pattern: {kept_dst} (skipped: {skipped_dst})")
    print(f"[INFO] Unique 'rest' patterns in destination:     {len(dst_by_rest)}")

    rows: List[Tuple[str, str]] = []
    parsed_src = 0
    skipped_src = 0

    # Counters for constraints / outcomes
    orig_in_dst = 0                 # constraint 1 violated
    candidates_found = 0
    cand_in_src_blocked = 0         # constraint 2 violated
    case_max_mismatch_blocked = 0   # constraint 3 violated
    renamed_found = 0

    # Cache per (old_case, new_case) whether max-patch constraint passes, to avoid recomputing
    max_patch_ok_cache: Dict[Tuple[str, str], bool] = {}

    total = len(src_files)
    for i, f in enumerate(src_files, start=1):
        if i == 1 or i % args.progress_every == 0 or i == total:
            print(
                f"[PROGRESS] {i}/{total} | "
                f"parsed={parsed_src} skipped={skipped_src} "
                f"orig_in_dst={orig_in_dst} "
                f"candidates={candidates_found} "
                f"cand_in_src={cand_in_src_blocked} "
                f"max_mismatch={case_max_mismatch_blocked} "
                f"renamed={renamed_found}"
            )

        # Constraint 1: original file must NOT exist in destination
        if f in dst_set:
            orig_in_dst += 1
            continue

        parts = split_case(f)
        if parts is None:
            skipped_src += 1
            continue

        parsed_src += 1
        _, old_case, rest = parts

        # Find candidate renamed file(s) in destination with same "rest"
        cands = dst_by_rest.get(rest)
        if not cands:
            continue
        candidates_found += 1

        picked = None
        for cand in cands:
            cand_parts = split_case(cand)
            if cand_parts is None:
                continue
            _, new_case, _ = cand_parts

            if new_case == old_case:
                continue  # not a case-number rename

            # Constraint 2: renamed candidate must NOT exist in source
            if cand in src_set:
                cand_in_src_blocked += 1
                continue

            # Constraint 3: max patch indices must match for the case-pair
            key = (old_case, new_case)
            ok = max_patch_ok_cache.get(key)
            if ok is None:
                old_max = src_max_patch.get(old_case)
                new_max = dst_max_patch.get(new_case)
                ok = (old_max is not None) and (new_max is not None) and (old_max == new_max)
                max_patch_ok_cache[key] = ok
            if not ok:
                case_max_mismatch_blocked += 1
                continue

            picked = cand
            break

        if picked is None:
            continue

        rows.append((f, picked))
        renamed_found += 1

    # Write CSV
    print(f"[INFO] Writing {len(rows)} rows to CSV...")
    with open(args.out, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["original_filename", "renamed_filename"])
        writer.writerows(rows)

    print("[DONE]")
    print(f"[SUMMARY] Total source files:                 {len(src_files)}")
    print(f"[SUMMARY] Skipped (no case_<num> match):       {skipped_src}")
    print(f"[SUMMARY] Original present unchanged in dst:   {orig_in_dst}")
    print(f"[SUMMARY] Had rename candidate(s):             {candidates_found}")
    print(f"[SUMMARY] Candidate existed in src (blocked):  {cand_in_src_blocked}")
    print(f"[SUMMARY] Max-patch mismatch (blocked):        {case_max_mismatch_blocked}")
    print(f"[SUMMARY] Renamed mappings written:            {len(rows)}")
    print(f"[SUMMARY] Output CSV: {args.out}")


if __name__ == "__main__":
    main()