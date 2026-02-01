#!/usr/bin/env bash
set -e

# Usage:
#   bash running.sh
# Notes:
#   - Defaults to 1 GPU (NPROC=1). Set NPROC=4 to use 4 GPUs on this node.
#   - One run per (dims, embed_dim) pair; logs go to each run's out dir.

# ---------- Minimal DDP controls ----------
NPROC=1            # GPUs per node
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

# ---------- Search space ----------
# all_metro_fcmae_dims=( "4,8,16,16" "4,8,16,32" "8,16,16,16" "8,16,32,32" "8,16,32,64")
# all_embed_dim=(16 32 16 32 64)
# all_metro_fcmae_dims=( "4,8,16,16" "4,8,16,32" "8,16,32,64")
all_metro_fcmae_dims=( "4,8,16,16" )
# all_embed_dim=(16 32 64)
all_embed_dim=(16)

# Sanity check
if [ "${#all_metro_fcmae_dims[@]}" -ne "${#all_embed_dim[@]}" ]; then
  echo "Error: metro_fcmae_dims and embed_dim length mismatch." >&2
  exit 1
fi

# ---------- Fixed args (edit as needed) ----------
FARM_TYPE="wind"
DATA_DIR=""/inspire/ssd/project/sais-mtm/public/qlz/linan/dl/ConvNeXt-V2/self_supervised/processed_wind_1.2_2""

# todo: 调小patch size，调小batch size

PROVINCE="Guangdong" 
TOTAL_BATCH_SIZE=$((2048 / 16))
PER_PROC_BATCH_SIZE=8
PATCH_SIZE=4
UPDATE_FREQ=$(( TOTAL_BATCH_SIZE / PER_PROC_BATCH_SIZE / NPROC ))
echo "$UPDATE_FREQ"  # 1
case "${PROVINCE}" in
  Guangdong)
    INPUT_SIZE="60,92"
    ;;
  Guangxi)
    INPUT_SIZE="60,92"
    ;;
  Yunnan)
    INPUT_SIZE="92,108"
    ;;
  Guizhou)
    INPUT_SIZE="56,76"
    ;;
  Hainan)
    INPUT_SIZE="28,32"
    ;;
  *)
    echo "Error: unknown province ${PROVINCE}"
    exit 1
    ;;
esac

echo "Province=${PROVINCE}, INPUT_SIZE=${INPUT_SIZE}"

MASK_RATIO=0.6
BLR=1.5e-4
EPOCHS=100
WARMUP_EPOCHS=1
DECODER_DEPTH=1
SAVE_CKPT_FREQ=1
SAVE_CKPT_NUM=-1
NUM_WORKERS=0

# ---------- Sweep ----------
for i in "${!all_embed_dim[@]}"; do
  embed_dim="${all_embed_dim[$i]}"
  dims="${all_metro_fcmae_dims[$i]}"
  SEED=$((42 + i))

  OUTDIR="/inspire/ssd/project/sais-mtm/public/qlz/linan/dl/ConvNeXt-V2/checkpoints_1.2/all-whole-pretrain-${PROVINCE}-${dims}"
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
      --metro_feat_channel 14 \
      --save_ckpt_freq "${SAVE_CKPT_FREQ}" \
      --save_ckpt_num "${SAVE_CKPT_NUM}" \
      --batch_size "${PER_PROC_BATCH_SIZE}" \
      --patch_size "${PATCH_SIZE}" \
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
      --auto_resume false \
      --seed "${SEED}" \
      --output_dir "${OUTDIR}" \
      --log_dir "${OUTDIR}" \
      2>&1 | tee -a "${LOGFILE}"
done
