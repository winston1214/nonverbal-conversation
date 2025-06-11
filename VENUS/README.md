# VENUS collection pipeline
<img src='https://github.com/winston1214/nonverbal-conversation/blob/main/imgs/VENUS_PIPELINE.png?raw=true'></img>

## (a) Data Collection and Filtering

#### 1. Search Channels or Videos

Use `--mode channels` to search for Youtube channels, or `--mode videos` to search for videos from those channels.

So, you must search channel first, and then search videos.

> ⚠️  Make sure there are duplicate channel IDs.

```bash
   python ytb_channel_video_search.py \
      --save_path $SAVE_PATH \
      --api $YOUR_API_KEY \
      --start_time $START_DATE \
      --end_time $END_DATE \
      --mode [channels | videos]
```
<!----
1. Channel search ⚠️  Make sure there are duplicate channel IDs.
   ```
   python ytb_channel_video_search.py --save_path $SAVE_PATH --api $YOUR_API_KEY --start_time &START_DATE --end_time $END_DATE --mode channels
   ```
2. video search 
   ```
   python ytb_channel_video_search.py --save_path $SAVE_PATH --api $YOUR_API_KEY --start_time &START_DATE --end_time $END_DATE --mode videos
   ```  
--->
3. thumbnail_filter.py
   ```
   python thumbnail_filter.py --save_path $SAVE_PATH --video_file_name $VIDEO_FILE_CSV_NAME --yolo_weight_path $YOLO_WEIGHT_PATH
   ```
2. Audio file download -> segment_id_list.csv (segment_time = 600 sec)
   ```
   python youtube_download.py --save_path $SAVE_PATH --mode wav --org_del 1
   ```
## (b) ASR Transcripts
4. Run whisper (STT) -> make filtering_file_list.csv
   ```
   python run_whisper.py --save_path $SAVE_PATH --hf_token $YOUR_HF_TOKEN
   ```
5. Cleaning folder to save disk memory -> make step1_segment_id_list.csv
   ```
   python cleaner.py --mode whisper --save_path $SAVE_PATH
   ```
6. Preprocessing for whisper results
   - mode 'whisper_result_preprocessing' : Process the results of Whisper to save 'final_word', 'final_seq', and 'utterance' as CSV files
   - mode 'make_csv_column' : Add 'utterance','sequence' columns and make 'step2_1_segment_id_list.csv'
   ```
   python whisper_result_preprocessing.py --mode ${mode} --shard_name ${shard_path}
   ```
6. Video download (Only video without audio)
   ```
   python youtube_download.py --mode video --save_path $SAVE_PATH
   ```
7. Segment video (segment time = 600 sec)
   ```
   python video_seg.py --save_path $SAVE_PATH
   ```
7. Check video (Check if the video download was successful) -> make step2_2_segment_id_list.csv
   ```
   python checking.py --save_path $SAVE_PATH --mode check_fps 
   ```
## (c) Identifying Speaker
8. RUN ASD (Active Speaker Detection)
   ```
   python step2_main.py --save_path $SAVE_PATH --batch_size $YOLO_BATCH_SIZE --weight_path $ASD_WEIGHT_PATH
   ```
9. Remove video with fixed frames
   ```
   python cleaner.py --save_path $SAVE_PATH --mode video
   ```
11. Cropping speaker -> make segment.csv
    ```
    python crop_person_single.py --mode full --save_path $SAVE_PATH
    ```
## (d) Extracting Nonverbal-Cues
14. Extract mesh parameter (Body language)
   ```
   cd OSX/demo/ && python demo_batch.py
   ```
15. Extract mesh parameter (Facial Expression)
   ```bash
   cd emoca-ydc/gdl_apps/EMOCA
   sh run_face.sh
   ```

<!----
   17. checking.py -> mode 'make_csv' or 'check_jpg_files'
       - mode 'check_jpg_files' or 'check_npy_files' : Check the results from 'crop_person' and 'OSX'
       - mode 'parallel' : Enable multithreading
       ```
       python checking.py --mode ${mode} --save_path ${shard_path} --parallel ${1 if parallel else 0}
       ```
--->
