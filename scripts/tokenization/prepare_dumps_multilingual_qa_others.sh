#!/bin/bash

DATA_BASE="/capstor/scratch/cscs/bmessmer/diverse_qa_per_language_v2/clean/data/"
OUTPUT_FOLDER="datasets/mdiverse-qa-multilingual"

echo "Processing root folder: $DATA_BASE"
echo "Output destination:     $OUTPUT_FOLDER"
echo "Excluding pattern:      eng_Latn"

python3 scripts/tokenization/prepare_dumps.py \
  --dataset-folder "$DATA_BASE" \
  --preprocessing-metadata-folder "$OUTPUT_FOLDER" \
  --filter-out "eng_Latn" \
  --n-dumps 10