all_metro_fcmae_dims=("3,6,12,24" "9,18,36,72" "18,36,72,144" "36,72,144,288")
all_embed_dim=(24 72 144 288)

# paired hyper-param check
if [ "${#all_metro_fcmae_dims[@]}" -ne "${#all_embed_dim[@]}" ]; then
  echo "Error: all_metro_fcmae_dims and all_embed_dim must have the same length." >&2
  exit 1
fi

for i in "${!all_embed_dim[@]}"; do
    
    embed_dim="${all_embed_dim[$i]}"
    dims="${all_metro_fcmae_dims[$i]}"

    python -m torch.distributed.launch --nproc_per_node=1 main_pretrain_metro_per_farm.py \
    --decoder_depth 1 --decoder_embed_dim $embed_dim \
    --metro_fcmae_dims $dims --metro_feat_channel 9 \
    --save_ckpt_freq 5 --save_ckpt_num -1 \
    --batch_size 64 --update_freq 8 \
    --input_size 11 \
    --mask_ratio 0.6 \
    --blr 1.5e-4 \
    --epochs 1600 \
    --warmup_epochs 40 \
    --data_root /inspire/ssd/project/sais-mtm/public/qlz/data/PowerEstimateData/PowerEstimateData-Per-Farm-Debug \
    --farm_type "solar" --farm_id "1329" \
    --output_dir ./checkpoints/solar-1329-$embed_dim/ 2>&1

done