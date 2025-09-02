"""
Simplified data mixture creator that works with a total token budget.

Usage:
python3 scripts/tools/create_data_mixture_v2.py \
    --folders /path/to/folder1 /path/to/folder2 \
    --weights 0.6 0.4 \
    --max_tokens 1000000000 \
    --output /path/to/output
"""

import argparse
import os
import random
from pathlib import Path
from typing import List
import numpy as np

SEED = 1234

def get_bin_files(path_to_folder: str, exclude_paths: List[str] = []) -> List[str]:
    """Get all .bin files from a folder, excluding files from exclude_paths."""
    # Build set of excluded real file paths
    exclude_file_set = set()
    for exclude_path in exclude_paths:
        for dp, _, fn in os.walk(os.path.expanduser(exclude_path), followlinks=True):
            for f in fn:
                if Path(f).suffix.lower().endswith(".bin"):
                    real_path = os.path.realpath(os.path.join(dp, f))
                    exclude_file_set.add(real_path)

    # Gather all .bin files in the given folder
    files = [
        os.path.join(dp, f)
        for dp, _, fn in os.walk(os.path.expanduser(path_to_folder), followlinks=True)
        for f in fn
        if Path(f).suffix.lower().endswith(".bin")
    ]

    # Filter out excluded files
    filtered_files = [
        f for f in files
        if os.path.realpath(f) not in exclude_file_set
    ]

    return filtered_files

def estimate_tokens_from_file_size(file_size_bytes: int) -> int:
    """Estimate number of tokens from file size (rough approximation)."""
    # Assuming 4 bytes per token (typical for tokenized data)
    return file_size_bytes // 4

def select_files_for_budget(files: List[str], target_tokens: int) -> List[str]:
    """Select files to reach target token count within ±10% tolerance."""
    selected_files = []
    current_tokens = 0
    tolerance = 0.03  # ±3% tolerance
    
    # First pass: select files until we exceed target
    for file in files:
        file_size = os.path.getsize(file)
        file_tokens = estimate_tokens_from_file_size(file_size)
        
        if current_tokens + file_tokens <= target_tokens * (1 + tolerance):
            selected_files.append(file)
            current_tokens += file_tokens
        else:
            break
    
    # If we're under the minimum threshold, try to add more files
    min_tokens = target_tokens * (1 - tolerance)
    if current_tokens < min_tokens:
        # Continue adding files until we reach minimum threshold
        for file in files:
            if file in selected_files:
                continue
                
            file_size = os.path.getsize(file)
            file_tokens = estimate_tokens_from_file_size(file_size)
            
            # Only add if it doesn't push us over the maximum
            if current_tokens + file_tokens <= target_tokens * (1 + tolerance):
                selected_files.append(file)
                current_tokens += file_tokens
                
                if current_tokens >= min_tokens:
                    break
    
    return selected_files

def create_symlink_mixture(folders: List[str], weights: List[float], max_tokens: int, 
                          output_folder: str, exclude_paths: List[str] = []):
    """Create a data mixture with symlinks based on token budget."""
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Calculate token budget for each folder
    folder_budgets = [int(weight * max_tokens) for weight in weights]
    
    # Get all files from each folder
    all_folder_files = []
    for folder in folders:
        folder_files = sorted(get_bin_files(folder, exclude_paths))
        random.Random(SEED).shuffle(folder_files)
        all_folder_files.append(folder_files)
    
    # Select files for each folder based on budget
    selected_files_per_folder = []
    actual_tokens_per_folder = []
    
    for folder_files, budget in zip(all_folder_files, folder_budgets):
        selected_files = select_files_for_budget(folder_files, budget)
        selected_files_per_folder.append(selected_files)
        
        # Calculate actual tokens used
        actual_tokens = sum(estimate_tokens_from_file_size(os.path.getsize(f)) 
                           for f in selected_files)
        actual_tokens_per_folder.append(actual_tokens)
    
    # Create symlinks
    used_files = set()
    for folder, selected_files in zip(folders, selected_files_per_folder):
        dataset_name = os.path.basename(folder)
        output_dataset_folder = os.path.join(output_folder, dataset_name)
        
        for original_file in selected_files:
            symlink_file = os.path.join(output_dataset_folder, 
                                      Path(original_file).relative_to(folder))
            Path(os.path.dirname(symlink_file)).mkdir(parents=True, exist_ok=True)
            
            used_files.add(original_file)
            if not os.path.exists(symlink_file):
                os.symlink(f"{original_file[:-4]}.bin", f"{symlink_file[:-4]}.bin")
                os.symlink(f"{original_file[:-4]}.idx", f"{symlink_file[:-4]}.idx")
    
    # Calculate final statistics
    total_actual_tokens = sum(actual_tokens_per_folder)
    actual_weights = [tokens / total_actual_tokens for tokens in actual_tokens_per_folder]
    
    # Create summary
    tolerance = 0.03
    min_target = max_tokens * (1 - tolerance)
    max_target = max_tokens * (1 + tolerance)
    within_tolerance = min_target <= total_actual_tokens <= max_target
    
    summary = f"""Dataset mixture created in {output_folder}
Total files: {len(used_files)}
Total tokens: {total_actual_tokens:,} ({total_actual_tokens/1e9:.2f}B)
Target tokens: {max_tokens:,} ({max_tokens/1e9:.2f}B)
Tolerance range: {min_target:,} - {max_target:,} ({min_target/1e9:.2f}B - {max_target/1e9:.2f}B)
Within ±10% tolerance: {'✓' if within_tolerance else '✗'}
Token Collected / Target: {total_actual_tokens/max_tokens*100:.1f}%

Folder breakdown:"""
    
    for folder, budget, actual, weight, actual_weight in zip(
        folders, folder_budgets, actual_tokens_per_folder, weights, actual_weights):
        folder_name = os.path.basename(folder)
        summary += f"\n  {folder_name}: {actual:,} tokens ({actual_weight:.1%}) [target: {budget:,}]"
    
    print(summary)
    
    # Save summary to file
    with open(os.path.join(output_folder, "dataset_mixture_summary.txt"), 'w') as outfile:
        outfile.write(summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a mixture of bin files using symlinks with token budget.")
    parser.add_argument("--folders", nargs='+', required=True, 
                       help="List of folders containing bin files recursively")
    parser.add_argument("--weights", nargs='+', type=float, required=True, 
                       help="Weights for each folder")
    parser.add_argument("--max_tokens", type=int, required=True,
                       help="Maximum total tokens for the mixture")
    parser.add_argument("--output", required=True, 
                       help="Output folder for symlinks")
    parser.add_argument("--exclude", nargs='*', default=[],
                       help="List of folder paths to exclude any files under them recursively")
    
    args = parser.parse_args()

    if len(args.folders) != len(args.weights):
        raise ValueError("Number of folders and weights must be the same.")
    
    # Normalize weights
    weights = [float(i)/sum(args.weights) for i in args.weights]
    
    # Log the planned mixture
    mixture = {folder: round(weight, 4) for folder, weight in zip(args.folders, weights)}
    print(f"Creating data mixture from {dict(sorted(mixture.items()))}...")
    print(f"Target total tokens: {args.max_tokens:,}")
    
    create_symlink_mixture(args.folders, weights, args.max_tokens, args.output, 
                          exclude_paths=args.exclude)
