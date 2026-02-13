DATA_BASE="/iopsstor/scratch/cscs/bmessmer/pleias_synth/data/output"

OUTPUT_FOLDER="datasets/PleIAs-SYNTH"

python3 scripts/tokenization/prepare_dumps.py \
  --dataset-folder "$DATA_BASE" \
  --preprocessing-metadata-folder "$OUTPUT_FOLDER" \
  --n-dumps 20
  



