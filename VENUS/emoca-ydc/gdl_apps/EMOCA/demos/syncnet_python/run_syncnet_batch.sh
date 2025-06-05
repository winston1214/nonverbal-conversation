#!/bin/bash
echo $CUDA_VISIBLE_DEVICES
echo $SLURM_NODELIST
echo $SLURM_NODEID

nvidia-smi

# python -u run_syncnet_batch.py \
#     --input_frame_folder /home/jisoo6687/talkinghead_dataset_/Celebv_text/shard_v3_frame_fps25 \
#     --input_audio_folder /home/jisoo6687/talkinghead_dataset_/Celebv_text/raw_audio/shard_v3_audio_fps25 \
#     --batch_size 100 \
#     --save_path /home/jisoo6687/talkinghead_dataset_/Celebv_text \
#     --dataset CELEBV-TEXT

python -u run_syncnet_batch_crop_crema.py \
    --audio_path /home/MIR_LAB/CREMA-D/25fps/wav_25fps \
    --frame_path /home/MIR_LAB/CREMA-D/emotalk/images_white_bg \
    --batch_size 140 \
    --save_path /home/MIR_LAB/metric/new_LSE \
    --dataset CREMA-D \
    --save_name Emotalk_CREMA-D_fps30