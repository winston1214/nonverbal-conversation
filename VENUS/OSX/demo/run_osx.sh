#!/bin/bash
# echo $CUDA_VISIBLE_DEVICES
# echo $SLURM_NODELIST
# echo $SLURM_NODEID

# rm -r /home/jisoo6687/OSX/demo/cropped_frames
# rm -r /home/jisoo6687/OSX/demo/frames
# rm -r /home/jisoo6687/OSX/demo/output
# rm -r /home/jisoo6687/OSX/demo/samples
# rm -r /home/jisoo6687/OSX/output

# tar -czvf /home/jisoo6687/OSX.tar.gz /home/jisoo6687/OSX
python demo_batch.py --input_folder /share0/MIR_LAB/nc_video/shard_1 --batch_size 256
