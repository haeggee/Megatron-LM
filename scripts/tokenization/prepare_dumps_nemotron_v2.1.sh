#!/bin/bash
DATA_BASE="/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-CC-v2.1/"

for dir in "$DATA_BASE"/*/; do
    subfolder_name=$(basename "$dir")
    output_folder="datasets/Nemotron-CC-v2.1-${subfolder_name}"
    
    echo "Processing subfolder: $subfolder_name"
    echo "Output destination:   $output_folder"

    python3 scripts/tokenization/prepare_dumps.py \
      --dataset-folder "$dir" \
      --preprocessing-metadata-folder "$output_folder" \
      --n-dumps 20
done

