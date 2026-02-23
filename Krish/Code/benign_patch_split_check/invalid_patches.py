#!/usr/bin/env python3

import os
import re
import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser(description="Find patches that do NOT follow naming convention")
    parser.add_argument(
        "--patches_dir",
        type=str,
        required=True,
        help="Directory containing patch .png files",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="invalid_patch_names.csv",
        help="CSV file to save invalid patch names",
    )
    args = parser.parse_args()

    patches_dir = args.patches_dir

    # Updated naming convention regex
    pattern = re.compile(
        r"""
        ^case_\d+_                     # case id
        (match|unmatched)_?\d+_        # match/unmatched with optional underscore
        (h&e|melan|sox10)              # stain
        (-labels)?                     # optional '-labels'
        \.?_patch\d+\.png$             # optional dot before _patch
        """,
        re.VERBOSE,
    )

    all_files = os.listdir(patches_dir)
    png_files = [f for f in all_files if f.lower().endswith(".png")]

    invalid_names = []

    for fname in png_files:
        if not pattern.match(fname):
            invalid_names.append(fname)

    print(f"\nTotal PNG files checked: {len(png_files)}")
    print(f"Invalid filenames found: {len(invalid_names)}")

    if invalid_names:
        print("\nExamples of invalid filenames:")
        for name in invalid_names[:10]:
            print("  ", name)

        df = pd.DataFrame({"invalid_patch_name": invalid_names})
        df.to_csv(args.output_csv, index=False)
        print(f"\nInvalid filenames saved to: {args.output_csv}")
    else:
        print("All filenames follow the naming convention ✔")


if __name__ == "__main__":
    main()