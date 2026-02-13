DATA_BASE="/iopsstor/scratch/cscs/bmessmer/finetranslations/data/output"

OUTPUT_FOLDER="datasets/finetranslations"

python3 scripts/tokenization/prepare_dumps.py \
  --dataset-folder "$DATA_BASE" \
  --preprocessing-metadata-folder "$OUTPUT_FOLDER" \
  --n-dumps 30
  