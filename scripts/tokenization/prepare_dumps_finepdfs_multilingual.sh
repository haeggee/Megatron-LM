#!/bin/bash

DATA_BASE="/capstor/store/cscs/swissai/infra01/datasets/HuggingFaceFW/finepdfs-edu/snapshot/data"
OUTPUT_FOLDER="datasets/finepdfs-edu-multilingual"

echo "Processing root folder: $DATA_BASE"
echo "Output destination:     $OUTPUT_FOLDER"
echo "Excluding pattern:      eng_Latn"

python3 scripts/tokenization/prepare_dumps.py \
  --dataset-folder "$DATA_BASE" \
  --preprocessing-metadata-folder "$OUTPUT_FOLDER" \
  --filter-out "eng_Latn" \
  --n-dumps 150