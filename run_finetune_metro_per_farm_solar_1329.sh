all_metro_fcmae_dims=("9,18,36,72")
all_embed_dim=(72)
farm_type=wind
farm_id=1000

# paired hyper-param check
if [ "${#all_metro_fcmae_dims[@]}" -ne "${#all_embed_dim[@]}" ]; then
  echo "Error: all_metro_fcmae_dims and all_embed_dim must have the same length." >&2
  exit 1
fi

for i in "${!all_embed_dim[@]}"; do
    
    embed_dim="${all_embed_dim[$i]}"
    dims="${all_metro_fcmae_dims[$i]}"

    python -m torch.distributed.launch --nproc_per_node=1 main_finetune_metro.py \
    --metro_fcmae_dims $dims --decoder_dim $embed_dim --metro_feat_channel 18 \
    --save_ckpt_freq 1 --save_ckpt_num -1 \
    --batch_size 16 --update_freq 1 \
    --input_size 11 \
    --eval False \
    --epochs 100 \
    --dist_eval False \
    --finetune /inspire/ssd/project/sais-mtm/public/qlz/code/PowerEstimate/ConvNeXt-V2/checkpoints/wind-1000-72/checkpoint-199.pth \
    --blr 6.25e-4 \
    --warmup_epochs 1 \
    --data_root /inspire/ssd/project/sais-mtm/public/qlz/data/PowerEstimateData/PowerEstimateData-Per-Farm-Debug \
    --farm_type $farm_type --farm_id $farm_id \
    --output_dir ./checkpoints/gj-mlp-$farm_type-$farm_id-$embed_dim/ 2>&1

done