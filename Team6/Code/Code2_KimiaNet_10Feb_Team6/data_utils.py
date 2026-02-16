"""
Data utilities for MIL training
"""
import os
import re
import random
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Any, Optional

from config import DATA_PATHS, VALID_CLASSES, SPLIT_CONFIG


def load_labels(csv_path: str = None) -> pd.DataFrame:
    """Load and preprocess labels CSV"""
    if csv_path is None:
        csv_path = DATA_PATHS['labels_csv']
    
    # dropping case 65 as it has no class assigned
    labels = pd.read_csv(csv_path).drop(index=64, errors='ignore').reset_index(drop=True)
    return labels


def clean_all_patch_files(patches_dir: str = None) -> List[str]:
    """Enforce standard naming convention and drop duplicates"""
    if patches_dir is None:
        patches_dir = DATA_PATHS['patches_dir']

    all_files = os.listdir(patches_dir)
    invalid_file_names = []
    duplicate_files = []
    flexibility_needed_counter = 0
    flexibility_success_counter = 0
    replacements = {
        r'case_0+': r'case_', 
        r'(match|unmatched)(\d)': r'\1_\2', 
        r'-labels': r'', 
        r'\.(?!png)': r'', 
        r'(patch\d+).*(.png)': r'\1\2'
    }
    
    for filename in all_files:
        # match standard naming convention strictly
        if re.fullmatch(r'case_[^0]\d*_\w+_\d+_(h&e|melan|sox10)_patch\d+.png', filename): 
            continue
        
        # apply str replacement to filename to standard naming
        original_filename = filename
        flexibility_needed_counter += 1
        for non_standard, standard in replacements.items(): 
            filename = re.sub(non_standard, standard, filename)

        # check success and rename files if true
        if re.fullmatch(r'case_[^0]\d*_\w+_\d+_(h&e|melan|sox10)_patch\d+.png', filename): 
            if not os.path.exists(os.path.join(patches_dir, filename)): 
                os.replace(
                    os.path.join(patches_dir, original_filename), 
                    os.path.join(patches_dir, filename)
                )
                flexibility_success_counter += 1
            else: 
                duplicate_files.append(original_filename)
        else:
            invalid_file_names.append(original_filename)

    # Printing summary
    if invalid_file_names:
        print(f"Found and excluding {len(invalid_file_names)} files not following naming convention:")
        for f in invalid_file_names[:5]:  # Show first 5
            print(f"  {f}")
        if len(invalid_file_names) > 5:
            print(f"  ... and {len(invalid_file_names) - 5} more")
    if duplicate_files:
        print(f"Found and excluding {len(duplicate_files)} potentially duplicate files:")
        for f in duplicate_files[:5]:  # Show first 5
            print(f"  {f}")
        if len(duplicate_files) > 5:
            print(f"  ... and {len(duplicate_files) - 5} more")
    
    print(
        f"{flexibility_success_counter} out of {flexibility_needed_counter} " \
        "non-standard file names were successfully coerced to standard."
    )

    invalid_file_names.extend(duplicate_files)
    return invalid_file_names


def get_all_patch_files(invalid_files: List[str] = None, patches_dir: str = None) -> List[str]:
    """Get list of all patch files"""
    if patches_dir is None:
        patches_dir = DATA_PATHS['patches_dir']
    if invalid_files is None:
        return os.listdir(patches_dir)
    
    valid_files = [file for file in os.listdir(patches_dir) if file not in invalid_files]
    return valid_files


def make_pseudocases(labels: pd.DataFrame, valid_files: List[str], max_slices: int):
    filtered_labels = labels[labels['Class'].isin(VALID_CLASSES)].set_index('Case')
    filtered_labels['Class'] = filtered_labels['Class'].replace({1: 0, 3: 1, 4: 1})

    patch_dict = lambda: defaultdict(patch_dict)
    patch_data = patch_dict()
    for filename in valid_files:
        match = re.match(r'case_([^0]\d*)_(\w+_\d+)_(h&e|melan|sox10)_patch(\d+).png', filename)
        if int(match.group(1)) in filtered_labels.index:
            class_id = int(filtered_labels.loc[int(match.group(1)), 'Class'])
            case_id = match.group(1)
            stain_id = match.group(3)
            slice_id = match.group(2)
            patch_id = match.group(4)
            patch_data[class_id][case_id][stain_id][slice_id][patch_id] = filename

    benign_many_slices = pd.DataFrame()
    for case_id, stain_id in patch_data[0].items():
        num_slices = {}
        for stain_id, slice_id in stain_id.items(): 
            num_slices[(case_id, stain_id)] = len(slice_id)
        if max(num_slices.values()) > max_slices: 
            for (case, stain), slices in num_slices.items():
                benign_many_slices.loc[case, stain] = slices

    benign_many_slices = pd.DataFrame()
    for case_id, stain_id in patch_data[0].items():
        num_slices = {}
        for stain_id, slice_id in stain_id.items(): 
            num_slices[(case_id, stain_id)] = len(slice_id)
        if max(num_slices.values()) > 5: 
            for (case, stain), slices in num_slices.items():
                benign_many_slices.loc[case, stain] = slices
    benign_many_slices.sort_index(inplace=True)

    for case in benign_many_slices.index:
        stains_count = {}
        for stain_id, slice_id in patch_data[0][case].items():
            stains_count[stain_id] = len(slice_id)
        cases_target = int(np.ceil(max(stains_count.values())/max_slices))

        new_case_split = {}
        for stain_id, slice_id in patch_data[0][case].items():
            new_case_split[stain_id] = \
                np.array_split(list(slice_id.keys()), cases_target)

        for i in range(cases_target):
            pseudocase_id = (i+1)*1000+int(case)
            for stain_id, slice_id in patch_data[0][case].items():
                patch_data[0][pseudocase_id][stain_id] = \
                    {
                        slice: patch for slice, patch in slice_id.items()
                        if slice in new_case_split[stain_id][i]
                    } 
                for slice_id, patch_id in patch_data[0][pseudocase_id][stain_id].items():
                    for path in patch_id.values():
                        new_path = re.sub(r'case_\d+_', f'case_{pseudocase_id}_', path)
                        os.replace(
                            os.path.join(DATA_PATHS['patches_dir'], path), 
                            os.path.join(DATA_PATHS['patches_dir'], new_path)
                        )
            with open(DATA_PATHS['labels_csv'], 'a') as f:
                f.write(f"\n{pseudocase_id},1")
    print(f'These cases have at least one stain with more than {max_slices} slices per stain')
    print(benign_many_slices)


def group_patches_by_slice(valid_files: List[str], patches_dir: str) -> Dict[Tuple[int, str], List[str]]:
    """
    Group patches by case and slice ID
    Returns: {(case_id, slice_id): [patch_paths]}
    """
    case_slices = defaultdict(list)
    
    for filename in valid_files:
        match = re.match(r"case_(\d+)_(\w+_\d+)_", filename)
        case_id = int(match.group(1))
        slice_id = match.group(2)
        key = (case_id, slice_id)
        case_slices[key].append(os.path.join(patches_dir, filename))
    
    return case_slices


def build_slice_to_class_map(slices: Dict, labels: pd.DataFrame) -> Dict[Tuple[int, str], int]:
    """Build mapping from (case_id, slice_id) to class label"""
    slice_to_class = {}
    
    for case_id, slice_id in slices.keys():
        raw_label = labels.loc[labels['Case'] == case_id, 'Class']
        if not raw_label.empty and raw_label.item() in VALID_CLASSES:
            # Convert to binary: 1.0 -> 0 (benign), 3.0/4.0 -> 1 (high-grade)
            label = 0 if raw_label.item() == 1.0 else 1
            slice_to_class[(case_id, slice_id)] = label
    
    return slice_to_class


def split_by_case_stratified(slices_by_class: Dict, random_state: int = 42) -> Tuple[List, List, List]:
    """
    Split data by case to prevent leakage, maintaining class balance
    Returns: train_slices, val_slices, test_slices
    """
    # Build case -> label map and validate no mixed-label cases
    case_to_labels = defaultdict(set)
    for label, items in slices_by_class.items():
        for case_id, _ in items:
            case_to_labels[case_id].add(label)
    # Flatten to case list and aligned labels
    case_ids = []
    case_labels = []
    for cid, labs in case_to_labels.items():
        if len(labs) > 1:
            print(f"Warning: Case {cid} has mixed labels: {labs}")
        case_ids.append(cid)
        case_labels.append(next(iter(labs)))  # Take the first (should be only) label
    
    # Split cases with stratification
    train_ratio = SPLIT_CONFIG['train_ratio']
    val_ratio = SPLIT_CONFIG['val_ratio']
    
    # First split: train vs temp (val + test)
    case_train, case_temp, y_train, y_temp = train_test_split(
        case_ids, case_labels, 
        test_size=(1 - train_ratio), 
        stratify=case_labels, 
        random_state=random_state
    )
    
    # Second split: val vs test from temp
    val_size = val_ratio / (val_ratio + SPLIT_CONFIG['test_ratio'])
    case_val, case_test, _, _ = train_test_split(
        case_temp, y_temp, 
        test_size=(1 - val_size), 
        stratify=y_temp, 
        random_state=random_state
    )
    
    case_train = set(case_train)
    case_val = set(case_val)
    case_test = set(case_test)
    
    # Map case splits back to slice-level lists
    train_slices, val_slices, test_slices = [], [], []
    for label, items in slices_by_class.items():
        for case_id, slice_key in items:
            if case_id in case_train:
                train_slices.append((case_id, slice_key))
            elif case_id in case_val:
                val_slices.append((case_id, slice_key))
            elif case_id in case_test:
                test_slices.append((case_id, slice_key))
            else:
                print(f'Critical error! Case {case_id} not in any split')
    
    return train_slices, val_slices, test_slices


def build_case_dict(slice_list: List[Tuple], patches: Dict, slice_to_class: Dict) -> Tuple[Dict, Dict]:
    """
    Build case dictionary and label map from slice list
    Returns: case_dict, label_map
    """
    case_dict = defaultdict(lambda: defaultdict(list))
    label_map = {}
    
    for case_id, slice_id in slice_list:
        if (case_id, slice_id) not in patches:
            continue
            
        patch_paths = patches[(case_id, slice_id)]
        
        # Group patches by stain (extract stain from filenames, not slice_id)
        stain_groups = defaultdict(list)
        for patch_path in patch_paths:
            stain = extract_stain_from_filename(patch_path)
            if stain:
                stain_groups[stain].append(patch_path)
        
        # Add each stain group as a separate slice
        for stain, stain_patches in stain_groups.items():
            case_dict[case_id][stain].append(stain_patches)
        
        # Set label for this case
        if (case_id, slice_id) in slice_to_class:
            label_map[case_id] = slice_to_class[(case_id, slice_id)]
    
    return dict(case_dict), label_map


def extract_stain_from_filename(filename: str) -> Optional[str]:
    """Extract stain type from patch filename"""
    filename_lower = filename.lower()
    if 'h&e' in filename_lower or '_he_' in filename_lower:
        return 'h&e'
    elif 'melan' in filename_lower:
        return 'melan'
    elif 'sox10' in filename_lower:
        return 'sox10'
    return None


def get_case_ids(case_dict: Dict) -> set:
    """Extract unique case IDs from case dictionary"""
    return set(case_dict.keys())


def get_all_paths(case_dict: Dict) -> set:
    """Extract all patch paths from case dictionary"""
    paths = set()
    for case_data in case_dict.values():
        for stain_data in case_data.values():
            for slice_paths in stain_data:
                paths.update(slice_paths)
    return paths


def check_disjoint_sets(set1: set, set2: set, name1: str, name2: str) -> Tuple[bool, set]:
    """Check if two sets are disjoint and return overlap"""
    overlap = set1.intersection(set2)
    return len(overlap) == 0, overlap


def report_no_leak(train_case_dict: Dict, val_case_dict: Dict, test_case_dict: Dict):
    """Report data leakage analysis"""
    # Case-level analysis
    train_cases = get_case_ids(train_case_dict)
    val_cases = get_case_ids(val_case_dict)
    test_cases = get_case_ids(test_case_dict)
    
    print("Cases per split:", len(train_cases), len(val_cases), len(test_cases))
    
    ok_tv, leak_tv = check_disjoint_sets(train_cases, val_cases, "train", "val")
    ok_tt, leak_tt = check_disjoint_sets(train_cases, test_cases, "train", "test")
    ok_vt, leak_vt = check_disjoint_sets(val_cases, test_cases, "val", "test")
    
    # Path-level analysis
    train_paths = get_all_paths(train_case_dict)
    val_paths = get_all_paths(val_case_dict)
    test_paths = get_all_paths(test_case_dict)
    
    print("Paths per split:", len(train_paths), len(val_paths), len(test_paths))
    
    ok_tv_p, leak_tv_p = check_disjoint_sets(train_paths, val_paths, "train", "val")
    ok_tt_p, leak_tt_p = check_disjoint_sets(train_paths, test_paths, "train", "test")
    ok_vt_p, leak_vt_p = check_disjoint_sets(val_paths, test_paths, "val", "test")
    
    # Summary
    def summarise(ok, leak, label):
        if ok:
            print(f"No leakage between {label}.")
        else:
            print(f"[LEAK!!!! Nooo] {label} overlap count = {len(leak)}")
    
    summarise(ok_tv, leak_tv, "train & val (cases)")
    summarise(ok_tt, leak_tt, "train & test (cases)")
    summarise(ok_vt, leak_vt, "val & test (cases)")
    summarise(ok_tv_p, leak_tv_p, "train & val (paths)")
    summarise(ok_tt_p, leak_tt_p, "train & test (paths)")
    summarise(ok_vt_p, leak_vt_p, "val & test (paths)")


def summarize_case_dict(case_dict: Dict, label_map: Dict = None, split_name: str = "train") -> pd.DataFrame:
    """
    Create summary DataFrame with per-case statistics
    """
    records = []
    
    for case_id, stains in case_dict.items():
        record = {"case_id": case_id, "split": split_name}
        total_patches = 0
        
        for stain in ("h&e", "melan", "sox10"):
            slice_lists = stains.get(stain, [])
            num_slices = len(slice_lists)
            num_patches = sum(len(paths) for paths in slice_lists)
            record[f"{stain}_slices"] = num_slices
            record[f"{stain}_patches"] = num_patches
            record[f"{stain}_missing"] = int(num_patches == 0)
            total_patches += num_patches
        
        record["total_patches"] = total_patches
        if label_map and case_id in label_map:
            record["label"] = label_map[case_id]
        else:
            record["label"] = None
        
        records.append(record)
    
    return pd.DataFrame.from_records(records)