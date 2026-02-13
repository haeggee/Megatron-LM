DATA_BASE="/capstor/store/cscs/swissai/infra01/users/vsabolce/apertus_moe/datasets/swissai-fineweb-2_0_1-quality_10-filterrobots/data/output"

OUTPUT_FOLDER="datasets/swissai-fineweb-2_0_1-quality_10-filterrobots-rehydrated"

python3 scripts/tokenization/prepare_dumps.py \
  --dataset-folder "$DATA_BASE" \
  --preprocessing-metadata-folder "$OUTPUT_FOLDER" \
  --n-dumps 60
  