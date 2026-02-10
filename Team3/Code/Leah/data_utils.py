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
    
    labels = pd.read_csv(csv_path).drop(index=64, errors='ignore').reset_index(drop=True)
    return labels


def get_all_patch_files(patches_dir: str = None) -> List[str]:
    """Get list of all patch files"""
    if patches_dir is None:
        patches_dir = DATA_PATHS['patches_dir']
    
    all_files = os.listdir(patches_dir)
    return all_files


def group_patches_by_slice(all_files: List[str], root_dir: str) -> Dict[Tuple[int, str], List[str]]:
    """
    Group patches by case and slice ID
    Returns: {(case_id, slice_id): [patch_paths]}
    """
    case_slices = defaultdict(list)
    invalid_file_names = []
    flexibility_needed_counter = 0
    
    for filename in all_files:
        if filename.endswith(".png"):
            # Try standard naming convention first
            match = re.match(r"case_(\d+)_([a-z]+_\d+)_", filename)
            if match:
                case_id = int(match.group(1))
                slice_id = match.group(2)
                key = (case_id, slice_id)
                case_slices[key].append(os.path.join(root_dir, filename))
                continue
            
            # Try without underscore between match/unmatched and number
            match = re.match(r"case_(\d+)_([a-z]+\d+)_", filename)
            if match:
                case_id = int(match.group(1))
                slice_id = match.group(2)
                # Add underscore between letters and numbers
                slice_id = re.sub(r'([A-Za-z])(\d)', r'\1_\2', slice_id)
                key = (case_id, slice_id)
                case_slices[key].append(os.path.join(root_dir, filename))
                flexibility_needed_counter += 1
                continue
            
            invalid_file_names.append(os.path.join(root_dir, filename))
    
    # Print summary
    if invalid_file_names:
        print(f"Found {len(invalid_file_names)} files not following naming convention:")
        for f in invalid_file_names[:5]:  # Show first 5
            print(f"  {f}")
        if len(invalid_file_names) > 5:
            print(f"  ... and {len(invalid_file_names) - 5} more")
    else:
        print(f"All {flexibility_needed_counter} non-standard file names were handled.")
    
    return case_slices

###LEAH EDIT#########################
###NEW UTILITY FUNCTION TO SPLIT BENIGN CASES WITH EXCESS SLICES INTO TRAIN TO BALANCE THE NUMBER OF SLICES PER CLASS
def split_cases_with_excess_slices(case_slices: Dict[Tuple[int, str], List[str]], max_slices: int=5, random_state: int=42) -> Dict[Tuple[int, str], List[str]]:
    """
    Split benign cases with more than 5 slices into more psuedocases in training set to balance the number of slices per class
    Returns: new_case_slices: {(new_case_id, slice_id): [patch_paths]} where new_case_id is either the original case_id or a pseudo-case ID for split cases (i.e. 26_g0, 26_g1, etc.)
    """
    random.seed(random_state)

    #Group case/stain by slices
    case_stain_to_slices = defaultdict(list)
    for (case_id, slice_id), paths in case_slices.items():
        stain = '_'.join(slice_id.split('_')[:-1])  # Extract stain from slice_id
        case_stain_to_slices[(case_id, stain)].append((slice_id, paths))

    #Identify cases to split 
    cases_to_split = set()
    for (case_id, stain), slices in case_stain_to_slices.items():
        if len(slices) > max_slices:
            cases_to_split.add((case_id, stain))
            print(f"Case {case_id} with stain {stain} has {len(slices)} slices and will be split.")

    #Perform splitting
    new_case_slices = {}
    case_to_stains = defaultdict(dict)
    for (case_id, stain), slices in case_stain_to_slices.items():
        case_to_stains[case_id][stain] = slices
    
    #Process/Split each case
    for case_id, stain_dict in case_to_stains.items():
        if case_id not in cases_to_split:
            # No splitting needed, just add to new_case_slices
            for stain, slices in stain_dict.items():
                for slice_id, paths in slices:
                    new_case_slices[(case_id, slice_id)] = paths
        else:
            slices_in_case = max(len(slices) for slices in stain_dict.values())
            num_groups = (slices_in_case + max_slices - 1) // max_slices  # Calculate number of groups needed

            stain_partitions = {}
            for stain, slices in stain_dict.items():
                random.shuffle(slices)  # Shuffle slices to randomize grouping
                stain_partitions[stain] = [slices[i:i + max_slices] for i in range(0, len(slices), max_slices)]    

            for group_idx in range(num_groups):
                pseudo_case_id = f"{case_id}_g{group_idx}"
                
                for stain, partitions in stain_partitions.items():
                    for slice_id, patch_paths in partitions[group_idx]:
                        new_case_slices[(pseudo_case_id, slice_id)] = patch_paths
    return new_case_slices

####NEED TO CREATE NEW HELPER NOW THAT WE HAVE CREATED PSUEDOCASES SO THAT OTHER FUNCTIONS DON'T BREAK WHEN THEY EXPECT CASE IDS TO BE INTEGERS
def extract_original_case_id(case_id):
    """
    Extract original case ID from pseudo-case ID.
    Examples: 26 → 26, '26_g0' → 26, '26_g1' → 26
    """
    if isinstance(case_id, str) and '_g' in case_id:
        # Extract the numeric part before '_g'
        return int(case_id.split('_g')[0])
    return case_id

def build_slice_to_class_map(patches: Dict, labels: pd.DataFrame) -> Dict[Tuple[int, str], int]:
    """Build mapping from (case_id, slice_id) to class label"""
    slice_to_class = {}
    
    for (case_id, slice_id), paths in patches.items():
        original_case_id = extract_original_case_id(case_id) ##edits here to make sure that it looks up and compares the right things
        raw_label = labels.loc[labels['Case'] == original_case_id, 'Class']
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
            original_case_id = extract_original_case_id(case_id)
            case_to_labels[original_case_id].add(label)
    
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
            original_case_id = extract_original_case_id(case_id)
            
            if original_case_id in case_train:
                train_slices.append((case_id, slice_key))  # Keep pseudo-case ID in slices
            elif original_case_id in case_val:
                val_slices.append((case_id, slice_key))
            elif original_case_id in case_test:
                test_slices.append((case_id, slice_key))
            else:
                print(f'Critical error! Case {original_case_id} not in any split')
    
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