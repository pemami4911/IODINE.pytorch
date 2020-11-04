# IODINE.pytorch
Unofficial PyTorch implementation of IODINE https://arxiv.org/abs/1903.00450

## Installation

We provide a conda environment file. Install dependencies with `conda env create -f multi_object.yml`.

## Data

The official Google Multi-Object Datasets are available as TFRecords here: https://github.com/deepmind/multi_object_datasets.
This code uses `h5py` versions of this data available here: [TODO] for download.

## Training

First create the following in a desired directory (with sufficient storage for holding model weights and tensorboard logs):
```
experiments/
    weights/
    tb/
    runs/
```
Hyperparameters and runtime parameters can be modified in `train_params.json`. To train the model, use the following command:
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_distributed.py with train_params.json dataset.data_path=$PATH_TO_DATASET_DIR seed=$SEED training.batch_size=8  training.out_dir=$DIR_TO_STORE_OUTPUTS training.run_suffix=$RUN_NAME --file_storage $DIR_TO_STORE_OUTPUTS/runs
```
For more GPUs, include their GPU IDs in the `CUDA_VISIBLE_DEVICES` and increase `nproc_per_node` to be equal to the number of GPUs.

## Evaluation

Compute the ARI score for 320 test images.

```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 eval_distributed.py with eval_params.json dataset.data_path=$PATH_TO_DATASET_DIR seed=$SEED test.checkpoint_dir=$DIR_WHERE_MODEL_WEIGHTS_ARE_SAVED test.checkpoint=$SAVED_MODEL_NAME
```
