#!/bin/bash
echo $CUDA_VISIBLE_DEVICES
echo $SLURM_NODELIST
echo $SLURM_NODEID

nvidia-smi

python -u demo_batch.py --gpu 0 --input_folder /home/jisoo6687/OSX/demo/cropped_frames --output_folder output/parameters --batch_size 32
# python -u demo.py