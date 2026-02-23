#!/usr/bin/env python3
"""
count_files_per_case.py

Counts number of files for each case in a folder.
Exports results to CSV.

Run:
    python count_files_per_case.py

Optional:
    python count_files_per_case.py --folder /projects/e32998/patches --out case_counts.csv
"""

import os
import re
import csv
import argparse
from collections import defaultdict

CASE_RE = re.compile(r"^case_(\d+)_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        default="/projects/e32998/patches_Team5",
        help="Folder containing files",
    )
    parser.add_argument(
        "--out",
        default="case_file_counts_newpatches.csv",
        help="Output CSV file",
    )
    args = parser.parse_args()

    folder = args.folder
    print(f"[INFO] Counting files in: {folder}")

    files = [
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
    ]

    case_counts = defaultdict(int)
    skipped = 0

    for f in files:
        match = CASE_RE.match(f)
        if match:
            case_num = match.group(1)
            case_counts[case_num] += 1
        else:
            skipped += 1

    print(f"[INFO] Total files scanned: {len(files)}")
    print(f"[INFO] Files matched to cases: {sum(case_counts.values())}")
    print(f"[INFO] Files skipped (no case_<num> pattern): {skipped}")
    print(f"[INFO] Unique cases found: {len(case_counts)}")

    # Write CSV
    with open(args.out, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["case_number", "file_count"])
        for case_num in sorted(case_counts, key=lambda x: int(x)):
            writer.writerow([case_num, case_counts[case_num]])

    print(f"[DONE] CSV written to: {args.out}")


if __name__ == "__main__":
    main()