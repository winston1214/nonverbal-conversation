import os
import pandas as pd
import shutil
import cv2
import numpy as np
import random
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import glob
from tqdm import tqdm
import argparse
import pickle
from vid_utils import video_file_format

class WhisperCleaner:
    def __init__(self, save_path):
        self.save_path = save_path
        self.filtering = pd.read_csv(os.path.join(self.save_path, 'filtering_file_list.csv'))
        self.org = pd.read_csv(os.path.join(self.save_path, 'org_filtering_file_list.csv'))
        self.not_english = pd.read_csv(os.path.join(self.save_path, 'not_english.csv'))
        self.before_segment_df = pd.read_csv(os.path.join(self.save_path, 'segment_id_list.csv'))
    def extract_remove_list(self):
        many = self.org[self.org['filter_boolean'] == False]['file'].tolist()
        
        not_twice = []
        for i in tqdm(self.filtering['file'].tolist()):
            try:
                seq = pd.read_csv(os.path.join(self.save_path, 'segment',i , f'seq_{i}.csv'))
                if len(seq['speaker'].unique()) != 2:
                    not_twice.append(i)
            except:
                pass
        remove_list = self.not_english['name'].tolist() + not_twice + many
        return remove_list
    def run_remove(self,remove_list):
        for i in tqdm(remove_list):
            shutil.rmtree(os.path.join(self.save_path, 'segment', i))
        real_segment_list = sorted(os.listdir(os.path.join(self.save_path, 'segment')))
        real_segment_df = pd.DataFrame({'segment_id' : real_segment_list})
        merge_df = pd.merge(self.before_segment_df, real_segment_df, on = 'segment_id')
        merge_df.to_csv(os.path.join(self.save_path, 'step1_segment_id_list.csv'), index=False)
    def step1_filtering_video_num(self):
        step1_filtering = pd.read_csv(os.path.join(self.save_path, 'step1_segment_id_list.csv'))
        video_num = len(step1_filtering['video_id'].unique())
        seg_num = step1_filtering.shape[0]
        print(f'video_num: {video_num}, seg_num : {seg_num}')


class StatVideoCleaner:
    def __init__(self,save_path):
        self.save_path=save_path
        self.segment_path=os.path.join(self.save_path,'segment')

    def __filter__(self):
        segment_paths=glob.glob(os.path.join(self.segment_path,'*'))
        to_remove_segments=[]
        for segment_path in tqdm(segment_paths):
            segment_name=os.path.basename(segment_path)
            
            word_csv=pd.read_csv(os.path.join(segment_path,f'final_word_{segment_name}.csv'))
           
            pckl_path=os.path.join(segment_path,'active_frame_list.pckl')
            if os.path.exists(pckl_path):
                with open(pckl_path,'rb') as fr:
                    data=pickle.load(fr)
                if len(data)<len(word_csv):
                    to_remove_segments.append(segment_name)
            else:
                to_remove_segments.append(segment_name)
        return to_remove_segments

    def __make_new_csv__(self,to_remove_segments):
    
        step2_2_csv=pd.read_csv(os.path.join(self.save_path,'step2_2_segment_id_list.csv'))
        print(f'Length of not moving videos : {len(to_remove_segments)}/{len(step2_2_csv)}')
        if input('Make step2_3.csv? : (y or n)')=='y':

            step2_3_csv=step2_2_csv[~step2_2_csv['segment_id'].isin(to_remove_segments)]
        
            step2_3_csv=step2_3_csv.reset_index(drop=True)
            step2_3_csv.to_csv(os.path.join(self.save_path,'step2_3_segment_id_list.csv'),index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--threshold', type=float, default=0.95)
    parser.add_argument('--mode',type=str, default='whisper')
    args = parser.parse_args()
    save_path = args.save_path
    if args.mode == 'whisper':
        cleaner = WhisperCleaner(save_path)
        remove_list = cleaner.extract_remove_list()
        cleaner.run_remove(remove_list)
        cleaner.step1_filtering_video_num()


    elif args.mode =='video':
        stat_filter=StatVideoCleaner(args.save_path)
        to_remove_segments=stat_filter.__filter__()
        stat_filter.__make_new_csv__(to_remove_segments)
        
        
