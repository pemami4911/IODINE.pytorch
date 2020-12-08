#!/bin/sh

SEED=601
DDP_PORT=29500
cd ..
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 train_distributed.py with train_params.json dataset.data_path=/blue/ranka/pemami seed=$SEED training.batch_size=3 training.out_dir=/blue/ranka/pemami/experiments training.run_suffix=IODINE-clevr6-small-scheduler-seed=$SEED Net.inference_iters=4
