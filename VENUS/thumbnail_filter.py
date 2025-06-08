from ultralytics import YOLO
import numpy as np
from PIL import Image
import requests
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import glob
import argparse


def thumb_filtering(save_path, video_file_name, yolo_weight_path):
    # csv_name = glob.glob(os.path.join(save_path, '*.csv'))[0].split('/')[-1]
    video_df = pd.read_csv(os.path.join(save_path, video_file_name))
    file_list = sorted(video_df['video_id'].tolist())
    person_model = YOLO('/home/winston1214/workspace/nc/video_feature_extract/yolov8m.pt')
    video_list = sorted(set(map(lambda x: x[:11], file_list)))
    new_file_list = []
    for i in tqdm(video_list):
        image_url = f"https://img.youtube.com/vi/{i}/original.jpg"
        image = np.array(
            Image.open(
                requests.get(image_url, stream=True).raw
                )
        )
        results = person_model(image, verbose=False, classes = 0)
        for r in results:
            xyxy = r.boxes.xyxy.detach().cpu().numpy().tolist()
        if len(xyxy) > 0:
            new_file_list.append(i)
    result_list = [item for item in file_list if any(substring in item for substring in new_file_list)]
    thumbnail_filter = pd.DataFrame({'video_id':result_list})
    merge_df = pd.merge(video_df, thumbnail_filter, on='video_id').reset_index(drop=True)
    merge_df.to_csv(os.path.join(save_path, 'video_thumbnail_filtering_file_list.csv'), index=False)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--video_file_name', type=str)
    parser.add_argument('--yolo_weight_path', type=str)
    args = parser.parse_args()
    thumb_filtering(args.save_path, args.video_file_name, args.yolo_weight_path)
    
