python -m torch.distributed.launch --nproc_per_node=1 main_pretrain.py \
--model convnextv2_base \
--batch_size 64 --update_freq 8 \
--blr 1.5e-4 \
--epochs 1600 \
--warmup_epochs 40 \
--data_path /inspire/ssd/project/sais-mtm/public/qlz/data/resnet-test/ \
--output_dir ./checkpoints/test/