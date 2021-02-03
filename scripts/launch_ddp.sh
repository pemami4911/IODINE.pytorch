#!/bin/sh

SEED=600
DDP_PORT=29500
cd ..
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_distributed.py with train_params.json dataset.data_path=/data/pemami/iclr2021 seed=$SEED training.batch_size=4 training.out_dir=/data/pemami/iclr2021/experiments training.run_suffix=IODINE-clevr6-efficiency-seed=$SEED Net.inference_iters=5 training.DDP_port=$DDP_PORT
