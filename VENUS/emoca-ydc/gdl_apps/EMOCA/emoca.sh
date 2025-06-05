#!/bin/bash
# export PYTHONPATH=/MIR/nas1/nc/nonverbal-conversation/emoca-ydc/gdl_apps:$PYTHONPATH
# python demos/run_emoca_on_frames_any_level.py --save_path /MIR/nas1/nc/shard_17 --start_idx 1000 --end_idx 1100 --partition 0 --batch_size 128
save_path=$1
start_idx=$2
end_idx=$3
partition=$4
python demos/run_emoca_on_frames_any_level.py --save_path $save_path --start_idx $start_idx --end_idx $end_idx --partition $partition