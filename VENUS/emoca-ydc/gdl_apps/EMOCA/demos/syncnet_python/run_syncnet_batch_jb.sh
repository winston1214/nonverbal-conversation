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

python -u run_syncnet_batch_jb.py \
    --audio_path /home/MIR_LAB/jisoo/MEAD/new_test_audio_shard \
    --frame_path /MIR_NAS/jisookim/EMOTE/stage2_noDEE/results_2_700_MEAD/images \
    --batch_size 140 \
    --save_path /MIR_NAS/jisookim/EMOTE/metric/new_LSE \
    --dataset MEAD \
    --save_name emo2vec_MEAD

# sbatch -q base_qos -p suma_rtx4090 --gres=gpu:1 --job-name=2vec_MEAD run_syncnet_batch_jb.sh