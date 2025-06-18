import os
import pandas as pd
import re
import argparse
import warnings
from tqdm import tqdm
import numpy as np
import shutil
warnings.filterwarnings('ignore')

def move_file_with_overwrite(src, dst):
    dst_full_path=os.path.join(dst,os.path.basename(src))
    if os.path.exists(dst_full_path):
        os.remove(dst_full_path)  
    shutil.move(src, dst)  
    
def has_number(s):
    # number pattern
    pattern = re.compile(r'\d')
    return bool(pattern.search(s))

def remove_rows_with_only_special_characters(word, seq): # remove special characters (both word and seq)
    na_word_idx = word[word['word'].isna()].index
    for na_idx in na_word_idx:
        word['word'].iloc[na_idx] = 'None'
    pattern = r'^[^a-zA-Z0-9\s]+$'  # special characters only
    word_cleaned = word[~word['word'].str.match(pattern)]

    word_noised = word[word['word'].str.match(pattern)]
    
    if len(word_noised):
        noise = word_noised['word'].tolist()

        idx_set = set()
        for idx, i in enumerate(seq['text']):
            for no in noise:
                if no in i:
                    idx_set.add(idx)
        idx_set = sorted(idx_set)
        noise = sorted(set(noise))
        for j in idx_set:
            for no in noise:
                noise_text = seq['text'].iloc[j]
                noise_text_list = noise_text.split()
                noise_text_list = [word for word in noise_text_list if no != word]
                noise_text_join = ' '.join(noise_text_list)
                seq['text'].iloc[j] = noise_text_join
    
    return word_cleaned.reset_index(drop=True), seq

def utterance_split(seq_df):
    dic_ls = []
    text = ''
    before_speaker = None

    for idx, row in seq_df.iterrows():
        current_speaker = row['speaker']
        
        if before_speaker is None and current_speaker is not None:
            start = row['start']

        if before_speaker != current_speaker:
            if before_speaker is not None: 
                end = seq_df.iloc[idx-1]['end']
                dic = {'start': start, 'end': end, 'speaker': before_speaker, 'text': text.strip()}
                dic_ls.append(dic)

            # initialize
            before_speaker = current_speaker
            start = row['start']
            text = row['text']
        else:
            text += ' ' + row['text']

    
    if before_speaker is not None:
        end = seq_df['end'].iloc[-1]
        dic = {'start': start, 'end': end, 'speaker': before_speaker, 'text': text.strip()}
        dic_ls.append(dic)

    result_df = pd.DataFrame(dic_ls)
    return result_df
    

def make_step2_2(shard_name,error_lists):
    step2_1_csv=pd.read_csv(os.path.join(shard_name,'step2_1_segment_id_list.csv'))
     
    step2_2_csv=step2_1_csv[~step2_1_csv['segment_id'].isin(error_lists)].reset_index(drop=True)
    
    step2_2_csv.to_csv(os.path.join(shard_name,'step2_2_segment_id_list.csv'),index=False)