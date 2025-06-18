import os
import pandas as pd
import re
import argparse
import warnings
from tqdm import tqdm
import numpy as np
import shutil
from itertools import chain
from whisper_preprocess_utils import *
warnings.filterwarnings('ignore')


    
def whisper_preprocessing(shard_name, segment_name):
    try: 
        word_data = pd.read_csv(os.path.join(shard_name, 'segment',segment_name, 'word_'+segment_name+'.csv'))
        seq_data = pd.read_csv(os.path.join(shard_name, 'segment',segment_name, 'seq_'+segment_name+'.csv'))
    except:
        #return 0
        word_data=pd.read_csv(os.path.join(shard_name,'whisper_org','word_'+segment_name+'.csv'))
        seq_data=pd.read_csv(os.path.join(shard_name,'whisper_org','seq_'+segment_name+'.csv'))

    error_issue=None
    
    if seq_data.shape[0]==0:
        error_issue='no contents in seq csv'
        return segment_name, error_issue
        
    opening_music = [] # remove opening music
    for idx, i in enumerate(seq_data['text']):
        if has_number(i)  and len(i.split())== 1:
            opening_music.append(idx)
        else:
            break
            
    remove_idx = []
    for music_idx in opening_music:
        if word_data['word'].iloc[music_idx] != seq_data['text'].iloc[music_idx] : 
            remove_idx.append(music_idx)
    if remove_idx:
        for r in remove_idx:
            opening_music.remove(r)


    
    word_data.drop(opening_music, axis=0, inplace = True)
    word_data = word_data.reset_index(drop=True)

    seq_data = seq_data.drop(opening_music)
    seq_data = seq_data.reset_index(drop=True)

    word_data, seq_data = remove_rows_with_only_special_characters(word_data, seq_data) # remove special characters

    last_index = word_data[word_data['end'] == seq_data['end'].iloc[-1]].index
    try: 
        last_index_number = word_data.iloc[last_index[0]+1]
        
        if len(seq_data.iloc[-1]['text'].split()) == 1: # remove closing music
            word_data = word_data[:last_index[0]+1]
    except:
        pass


    for idx, te in enumerate(seq_data['text']):
        if not te:
            continue
        first = te.split()[0]
        if has_number(first):
            try:
                second = te.split()[1]
                seq_start_time = seq_data.iloc[idx]['start']
                seq_end_time = seq_data.iloc[idx]['end']
                for index, row in word_data.iterrows(): # find first word index
                    if seq_start_time <= row['start']  <= seq_end_time and second == row['word']:
                        second_idx = index
                        break
                word_data['start'][second_idx-1] = seq_data.iloc[idx]['start']
                word_data['end'][second_idx-1] = word_data.iloc[second_idx]['start']
            except:
                pass
            
    
    found_in_rows = []
    num_word = word_data[word_data.isna().any(axis=1)].index
    for num in num_word:
        word_to_find = word_data.iloc[num]['word']
        start = word_data.iloc[num-1]['start']
        idx = 2
        while np.isnan(start): # nan check
            start = word_data.iloc[num-idx]['start']
            idx += 1
        for index, row in seq_data.iterrows():
            if row['start'] <= start <= row['end'] and word_to_find in row['text']:
                found_in_rows.append(index)

    f_rows_to_delete=[]
    for num_word, f_row in zip(num_word, found_in_rows):
        if f_row in f_rows_to_delete:
            continue
        text = seq_data.iloc[f_row]['text']
        word = word_data.iloc[num_word]
        seq_start_time = seq_data.iloc[f_row]['start']
        seq_end_time = seq_data.iloc[f_row]['end']
        ## word interpolation
        text_split = text.split()
        first_word = text_split[0]
        last_word = text_split[-1]
        
        if has_number(first_word) == 1:
            seq_start_time = seq_data.iloc[f_row-1]['end']
            for index, row in word_data.iterrows():
                if row['end'] == seq_start_time:
                    first_idx = index +1

        
        elif has_number(first_word) == 0: # first word is not number
            for index, row in word_data.iterrows(): # find first word index
                if row['start'] ==  seq_start_time: # and first_word in row['word']:
                    first_idx = index
                    break
        if has_number(last_word) == 1: # number : start-nan, end-nan
            if f_row+1>=len(seq_data):
                word_data=word_data.drop(list(word_data.iloc[-len(text_split):].index)).reset_index(drop=True)
                seq_data=seq_data.drop(f_row).reset_index(drop=True)
                f_rows_to_delete.append(f_row)
                continue
                
            else:
                seq_end_time = seq_data.iloc[f_row+1]['start']
                
                for index, row in word_data.iterrows(): # find first word index
                    if row['start'] ==  seq_end_time :# and last_word in row['word']:
                        end_idx = index - 1
                        break

        elif has_number(last_word) != 1:
            for index, row in word_data.iterrows(): # find last word index
                if row['end'] ==  seq_end_time and last_word in row['word']:
                    end_idx = index
                    break
        
        try: 
            word_data.iloc[[i for i in range(first_idx,end_idx+1)]] = word_data.iloc[[i for i in range(first_idx,end_idx+1)]].interpolate(method='linear')
        except:
            error_issue='Interpolation Error'
           
            
        
        ## speaker interpolation
        speaker = seq_data.iloc[f_row]['speaker']
        word_data['speaker'][num_word] = speaker
        word_data['speaker'] = word_data['speaker'].ffill()

    
    # save file and word_seq processing
    whisper_org_path=os.path.join(shard_name,'whisper_org')
    if not os.path.exists(whisper_org_path):
        os.makedirs(whisper_org_path,exist_ok=True)

    word_data['speaker'] = word_data['speaker'].ffill()
        
    # seq_df processing -> reconstruct text
    seq_text=sum((text.split() for text in seq_data['text']),[])

    
    if len(word_data) != len(seq_text):
        df2 = word_seq_missing_index(shard_name, segment_name, word_data,seq_data) # new word_df
        if len(df2) != len(seq_text): # still length is not match
            error_issue='length inconsistency between word and seq'
        df2.to_csv(os.path.join(shard_name, 'segment',segment_name, f'final_word_{segment_name}.csv'), index=False)
        
    else:
           
        word_data.to_csv(os.path.join(shard_name, 'segment',segment_name, f'final_word_{segment_name}.csv'), index=False)
        seq_data.to_csv(os.path.join(shard_name, 'segment',segment_name, f'final_seq_{segment_name}.csv'), index=False)
        utterance_split(seq_data).to_csv(os.path.join(shard_name, 'segment',segment_name, f'utterance_{segment_name}.csv'), index=False)

        org_word_df_path=os.path.join(shard_name,'segment',segment_name,f'word_{segment_name}.csv')
        org_seq_df_path=os.path.join(shard_name,'segment',segment_name,f'seq_{segment_name}.csv')
        
        if os.path.exists(org_word_df_path) and os.path.exists(org_seq_df_path):
            move_file_with_overwrite(org_word_df_path,whisper_org_path)
            move_file_with_overwrite(org_seq_df_path,whisper_org_path)
            
    return segment_name,error_issue

def word_seq_missing_index(shard_name, segment_name, df,seq_df):

     
     seq_dict={}
     word_dict={}
     i=0
     for _,sequence in seq_df.iterrows():
         seq_words=sequence['text'].split()
         for seq_word in seq_words:
             seq_dict[i]=seq_word
             i+=1
     for i, word in df.iterrows():
         word_dict[i]=word['word']
  
    
     missing_indexes = []
     seq_index = 0
 
     for word_index, word in word_dict.items():
        found = False
        
        
        while seq_index < len(seq_dict):
            if word == seq_dict[seq_index]:  
                found = True
                seq_index += 1
               
                break
                
            else:
                break
                
        
        if not found:
            missing_indexes.append(word_index)

     df = df.drop(missing_indexes).reset_index(drop=True)
     return df




def process_segment(shard_name,segments):
    error_list=[]
    for segment_id in tqdm(segments):
        segment_name,error_issue=whisper_preprocessing(shard_name,segment_id)
        if error_issue:
            error_list.append((segment_name,error_issue))

    return error_list

def make_csv_column(shard_name,segments):
    seg_list = sorted(pd.read_csv(os.path.join(shard_name, 'step2_2_segment_id_list.csv'))['segment_id'].tolist())
    remove_list = []
    for seg in tqdm(segments):
        try:
            word = pd.read_csv(os.path.join(shard_name, 'segment', seg, f'final_word_{seg}.csv'))
            seq = pd.read_csv(os.path.join(shard_name, 'segment', seg, f'final_seq_{seg}.csv'))
            utt = pd.read_csv(os.path.join(shard_name, 'segment', seg, f'utterance_{seg}.csv'))
        except FileNotFoundError:
            remove_list.append((seg,'File not found'))
            continue
            
        word_seq_map = []
        for idx in range(len(seq)):
            seq_start = seq.iloc[idx]['start']
            seq_end = seq.iloc[idx]['end']
            try:
                word_start_idx = word[word['start'] == seq_start].index[0]
            except:
                remove_list.append((seg,'Front number issue'))
                
                break
            word_end_idx = word[word['end'] == seq_end].index
            if len(word_end_idx) == 1:
                word_end_idx = word_end_idx[0]
            elif len(word_end_idx) > 1:
                word_end_idx = word_end_idx[-1]
            word_seq_map.append([idx] * (word_end_idx - word_start_idx + 1))
        ws_map = list(chain(*word_seq_map))
        if seg in [x[0] for x in remove_list]:
            continue
            
        seq_utt_map = []
        for idx in range(len(utt)):
            utt_start = utt.iloc[idx]['start']
            utt_end = utt.iloc[idx]['end']
            seq_start_idx = seq[seq['start'] == utt_start].index[0]
            seq_end_idx = seq[seq['end'] == utt_end].index
            if len(seq_end_idx) == 1:
                seq_end_idx = seq_end_idx[0]
            elif len(seq_end_idx) > 1:
                seq_end_idx = seq_end_idx[-1]
            seq_utt_map.append([idx] * (seq_end_idx - seq_start_idx + 1))
        su_map = list(chain(*seq_utt_map))
        try:
            word['sequence'] = ws_map
            seq['utterance'] = su_map
            word.to_csv(os.path.join(shard_name, 'segment', seg, f'final_word_{seg}.csv'), index=False)
            seq.to_csv(os.path.join(shard_name, 'segment', seg, f'final_seq_{seg}.csv'), index=False)
        except:
            remove_list.append((seg,'mapping issue'))
    #with open(error_make_csv_column_path,'w') as f:
       # f.writelines([f'{shard_name}/segment/{remove_seg}\n' for remove_seg in remove_list])
            
    return remove_list

def handle_error_lists(shard_name,error_list):

    total_error_segs=[x[0] for x in error_list]
    total_error_issues=[x[1] for x in error_list]

    error_data = {
        'segment_id': total_error_segs,
        'issue': total_error_issues
    }
    error_df = pd.DataFrame(error_data)

    print(f'Error segments making step2_2 csv : {len(list(set(total_error_segs)))} ')
    # make step2_2 csv
    make_step2_2(shard_name,total_error_segs)
    
    # Save to CSV
    error_csv_path = os.path.join(shard_name,'whisper_preprocess_error.csv')
    error_df.to_csv(error_csv_path, index=False)
    print(f'Error list CSV saved to {error_csv_path}')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard_name', type=str, default='tmp')
    args = parser.parse_args()
    file_list = sorted(os.listdir(os.path.join(args.shard_name,'segment')))
    

    error_list = process_segment(args.shard_name, file_list)
    handle_error_lists(args.shard_name,error_list) # step_2_2 segment_id_list.csv
    remove_list=make_csv_column(args.shard_name,file_list)
    remove_segs=[x[0] for x in remove_list]
    error_make_csv_column_path= os.path.join(args.shard_name, 'error_make_csv_column.txt')


    with open(error_make_csv_column_path,'w') as f:
        for remove_seg in remove_segs:
            remove_paths = os.path.join(args.shard_name,'segment',remove_seg)
            f.write(f'{remove_paths}\n')
        # f.writelines([f'{args.shard_name}/segment/{remove_seg}\n' for remove_seg in remove_segs])

    step2_2_csv=pd.read_csv(os.path.join(args.shard_name,'step2_2_segment_id_list.csv'))
    print(f'Lengths of remove list : {len(remove_segs)}')
    step2_2_csv=step2_2_csv[~step2_2_csv['segment_id'].isin(remove_segs)].reset_index(drop=True)
    step2_2_csv.to_csv(os.path.join(args.shard_name,'step2_2_segment_id_list.csv'),index=False)

