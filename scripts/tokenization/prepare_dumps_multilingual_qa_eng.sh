DATA_BASE="/capstor/scratch/cscs/bmessmer/diverse_qa_per_language_v2/clean/data/"

TARGET="${1:-"eng_Latn"}"

echo "Targeting pattern: $TARGET"

# The loop now uses the variable: either specific folder or wildcard
for dir in "$DATA_BASE"/$TARGET/; do
    # Check if directory exists (handles cases where the specific folder name is wrong)
    if [ -d "$dir" ]; then
        subfolder_name=$(basename "$dir")
        output_folder="datasets/mdiverse-${subfolder_name}"
        
        echo "------------------------------------------------"
        echo "Processing subfolder: $subfolder_name"
        echo "Output destination:   $output_folder"

        python3 scripts/tokenization/prepare_dumps.py \
          --dataset-folder "$dir" \
          --preprocessing-metadata-folder "$output_folder" \
          --n-dumps 10
    else
        echo "Warning: Directory not found for pattern '$TARGET'"
    fi
done