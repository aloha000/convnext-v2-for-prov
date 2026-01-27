#!/usr/bin/env bash
set -e

# Usage:
#   bash running.sh
# Notes:
#   - Defaults to 1 GPU (NPROC=1). Set NPROC=4 to use 4 GPUs on this node.
#   - One run per (dims, embed_dim) pair; logs go to each run's out dir.

# ---------- Minimal DDP controls ----------
NPROC=2            # GPUs per node
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

# ---------- Search space ----------
all_metro_fcmae_dims=("8,16,32,64" "14,28,28,28" "21,42,42,42" "21,42,84,84" "21,42,84,168")
all_embed_dim=(64 28 42 84 168)

# Sanity check
if [ "${#all_metro_fcmae_dims[@]}" -ne "${#all_embed_dim[@]}" ]; then
  echo "Error: metro_fcmae_dims and embed_dim length mismatch." >&2
  exit 1
fi

# ---------- Fixed args (edit as needed) ----------
FARM_TYPE="wind"
DATA_DIR=""/inspire/ssd/project/sais-mtm/public/qlz/data/PowerEstimateData/gongjia_processed_data/self_supervised/processed_wind""

TOTAL_BATCH_SIZE=$((2048 / 16))
PER_PROC_BATCH_SIZE=16
UPDATE_FREQ=$(( TOTAL_BATCH_SIZE / PER_PROC_BATCH_SIZE / NPROC ))
echo "$UPDATE_FREQ"  # 1
INPUT_SIZE=11
MASK_RATIO=0.6
BLR=1.5e-4
EPOCHS=100
WARMUP_EPOCHS=1
DECODER_DEPTH=1
SAVE_CKPT_FREQ=1
SAVE_CKPT_NUM=-1
NUM_WORKERS=8

# ---------- Sweep ----------
for i in "${!all_embed_dim[@]}"; do
  embed_dim="${all_embed_dim[$i]}"
  dims="${all_metro_fcmae_dims[$i]}"
  SEED=$((42 + i))

  OUTDIR="./checkpoints/wind-whole-pretrain-${dims}"
  LOGFILE="${OUTDIR}/train.log"
  mkdir -p "${OUTDIR}"

  echo "==> dims=${dims} | decoder_embed_dim=${embed_dim} | out=${OUTDIR}"

  torchrun \
    --nproc_per_node "${NPROC}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    main_pretrain_metro_whole.py \
      --data_dir "${DATA_DIR}" \
      --decoder_depth "${DECODER_DEPTH}" \
      --decoder_embed_dim "${embed_dim}" \
      --metro_fcmae_dims "${dims}" \
      --metro_feat_channel 21 \
      --save_ckpt_freq "${SAVE_CKPT_FREQ}" \
      --save_ckpt_num "${SAVE_CKPT_NUM}" \
      --batch_size "${PER_PROC_BATCH_SIZE}" \
      --update_freq "${UPDATE_FREQ}" \
      --input_size "${INPUT_SIZE}" \
      --mask_ratio "${MASK_RATIO}" \
      --blr "${BLR}" \
      --epochs "${EPOCHS}" \
      --warmup_epochs "${WARMUP_EPOCHS}" \
      --farm_type "${FARM_TYPE}" \
      --num_workers "${NUM_WORKERS}" \
      --pin_mem true \
      --eval_freq 1 \
      --auto_resume true \
      --seed "${SEED}" \
      --output_dir "${OUTDIR}" \
      --log_dir "${OUTDIR}" \
      2>&1 | tee -a "${LOGFILE}"
done
