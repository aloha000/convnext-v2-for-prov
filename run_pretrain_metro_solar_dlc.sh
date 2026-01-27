#!/bin/bash
set -e

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        . "/opt/conda/etc/profile.d/conda.sh"
    else
        export PATH="/opt/conda/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda init
conda info --envs
conda activate climate

cd  /inspire/ssd/project/sais-mtm/public/qlz/code/PowerEstimate/ConvNeXt-V2

### CONDA ENV SET, START TRAINING SETTING

# ---------- Minimal DDP controls ----------
NPROC=4            # GPUs per node
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

# ---------- Search space ----------
all_metro_fcmae_dims=("6,12,12,12" "9,18,18,18" "9,18,36,36" "9,18,36,72")
all_embed_dim=(12 18 36 72)
# Sanity check
if [ "${#all_metro_fcmae_dims[@]}" -ne "${#all_embed_dim[@]}" ]; then
  echo "Error: metro_fcmae_dims and embed_dim length mismatch." >&2
  exit 1
fi

# ---------- Fixed args (edit as needed) ----------
FARM_TYPE="solar"

TOTAL_BATCH_SIZE=$((2048 / 16))
PER_PROC_BATCH_SIZE=32
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

  OUTDIR="./checkpoints/solar-whole-pretrain-${dims}"
  LOGFILE="${OUTDIR}/train.log"
  mkdir -p "${OUTDIR}"

  echo "==> dims=${dims} | decoder_embed_dim=${embed_dim} | out=${OUTDIR}"

  torchrun \
    --nproc_per_node "${NPROC}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    main_pretrain_metro_whole.py \
      --decoder_depth "${DECODER_DEPTH}" \
      --decoder_embed_dim "${embed_dim}" \
      --metro_fcmae_dims "${dims}" \
      --metro_feat_channel 9 \
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
