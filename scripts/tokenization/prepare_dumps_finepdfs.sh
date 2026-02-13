DATA_BASE="/capstor/store/cscs/swissai/infra01/datasets/HuggingFaceFW/finepdfs-edu/snapshot/data"

# Use the first argument as the target, or default to "*" (all folders) if empty
TARGET="${1:-"*"}"

echo "Targeting pattern: $TARGET"

# The loop now uses the variable: either specific folder or wildcard
for dir in "$DATA_BASE"/$TARGET/; do
    # Check if directory exists (handles cases where the specific folder name is wrong)
    if [ -d "$dir" ]; then
        subfolder_name=$(basename "$dir")
        output_folder="datasets/finepdfs-edu-${subfolder_name}"
        
        echo "------------------------------------------------"
        echo "Processing subfolder: $subfolder_name"
        echo "Output destination:   $output_folder"

        python3 scripts/tokenization/prepare_dumps.py \
          --dataset-folder "$dir" \
          --preprocessing-metadata-folder "$output_folder" \
          --n-dumps 40
    else
        echo "Warning: Directory not found for pattern '$TARGET'"
    fi
done