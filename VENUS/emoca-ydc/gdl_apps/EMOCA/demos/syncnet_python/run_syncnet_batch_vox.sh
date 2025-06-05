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

# RVQ MEAD : /home/MIR_LAB/checkpoints/DEMOTE_VQ_topcondition/stage2/DEEPTalk_with_RVQVAE_stage2/MEAD_test
# emo MEAD : /MIR_NAS/jisookim/EMOTE/stage2_noDEE/results_2_700_MEAD/images

# python -u run_syncnet_batch_vox.py \
#     --audio_path  /home/MIR_LAB/jisoo/MEAD/new_test_audio \
#     --frame_path /home/MIR_LAB/checkpoints/DEMOTE_VQ_topcondition/stage2/DEEPTalk_with_RVQVAE_stage2/MEAD_test_epoch3_step1000/images \
#     --batch_size 140 \
#     --save_path /MIR_NAS/jisookim/EMOTE/metric/new_LSE \
#     --dataset DEMOTE_RVQ_MEAD

python -u run_syncnet_batch_vox.py \
    --audio_path  /home/MIR_LAB/jungbin/curated_data/emotional_benchmark/data/audio \
    --frame_path /MIR_NAS/jisookim/EMOTE/stage2_noDEE/results_2_700_EmoVox/images \
    --batch_size 140 \
    --save_path /MIR_NAS/jisookim/EMOTE/metric/new_LSE \
    --dataset DEMOTE_emo2vec_emovox_crop1

# sbatch -q a100_1_qos -p suma_a100_1 --gres=gpu:1 --job-name=RV_MEAD run_syncnet_batch_vox.sh
# sbatch -q big_qos -p suma_rtx4090 --gres=gpu:1 --job-name=nemo_MEAD run_syncnet_batch_vox.sh