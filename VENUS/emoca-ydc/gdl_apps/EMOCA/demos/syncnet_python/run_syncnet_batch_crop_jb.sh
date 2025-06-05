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

## CREMA-D
# audio : /home/MIR_LAB/CREMA-D/25fps/wav_25fps
# DEEPTalk : /home/MIR_LAB/checkpoints/DEMOTE_VQ_topcondition/stage2/stage2_onlyliploss0.001_avemo1e-4_MEAN_noselfCL_tauM2m0.1_model_DEMOTEVQ_GT_probDEE_mean_HFMP_lr0.0008steplr_topcondition_ALIBIPeriod25_gumbel_tauM2m0.1_stacktranslayer6_32_stage1/CREMA-D
# FaceFormer : /home/MIR_LAB/checkpoints/faceformer/faceformer_batch64/CREMA-D/images
# FaceDiffuser : /MIR_NAS/jisookim/results/facediffuser/CREMA-D/images
# EMOTE : /MIR_NAS/jisookim/results/EMOTE/CREMA-D/frames

## HDTF
# audio : /home/MIR_LAB/HDTF_/wav_subsets
# DEEPTalk : /home/MIR_LAB/checkpoints/DEMOTE_VQ_topcondition/stage2/stage2_onlyliploss0.001_avemo1e-4_MEAN_noselfCL_tauM2m0.1_model_DEMOTEVQ_GT_probDEE_mean_HFMP_lr0.0008steplr_topcondition_ALIBIPeriod25_gumbel_tauM2m0.1_stacktranslayer6_32_stage1/HDTF
# FaceFormer : /home/MIR_LAB/checkpoints/faceformer/faceformer_batch64/HDTF
# FaceDiffuser : /MIR_NAS/jisookim/results/facediffuser/HDTF/images
# EMOTE : /MIR_NAS/jisookim/results/EMOTE/HDTF_EMOTE/frames

python -u run_syncnet_batch_crop_jb.py \
    --audio_path /home/MIR_LAB/HDTF_/wav_subsets \
    --frame_path /home/MIR_LAB/checkpoints/DEMOTE_VQ_topcondition/stage2/stage2_onlyliploss0.001_avemo1e-4_MEAN_noselfCL_tauM2m0.1_model_DEMOTEVQ_GT_probDEE_mean_HFMP_lr0.0008steplr_topcondition_ALIBIPeriod25_gumbel_tauM2m0.1_stacktranslayer6_32_stage1/HDTF \
    --batch_size 140 \
    --save_path /MIR_NAS/jisookim/results/DEEPTalk \
    --dataset HDTF \
    --save_name DEEPTalk_HDTF_cut_resize300