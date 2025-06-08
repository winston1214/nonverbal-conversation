import os
import pandas as pd
from tqdm import tqdm
import argparse
import glob
import shutil
import subprocess
import concurrent.futures

ERROR_LOG_PATH = "./error.log"

def download_full_video(save_path, max_workers=8):
    data = pd.read_csv(os.path.join(save_path, 'step1_segment_id_list.csv'))
    file_list = sorted(set(data['video_id'].tolist()))
    os.makedirs(os.path.join(save_path, 'full_video'), exist_ok = True)
    output_path = os.path.join(save_path, 'full_video')
    file_list = data['video_id'].tolist()
    uni_file = sorted(set(list(map(lambda x: x[:11], file_list))))
    p_bar = tqdm(range(len(uni_file)))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
        futures = [executor.submit(run_downloader, output_path, uid, f'https://www.youtube.com/watch?v={uid}') for uid in uni_file]
        for future in concurrent.futures.as_completed(futures):
            success, output_path, uid, url, *error = future.result()
            if not success:
                with open(os.path.join(save_path, ERROR_LOG_PATH), 'a') as error_file:
                    error_message = error[0] if error else 'Unknown Error'
                    error_file.write(f'{uid}, {url} | {error_message}\n')
            p_bar.update(1)




def download_wav_file(data_name,save_folder, segment_time, org_del = False):
    data = pd.read_csv(data_name)
    os.makedirs(save_folder, exist_ok=True)
    output_folder = os.path.join(save_folder, 'mono')
    os.makedirs(output_folder, exist_ok=True)
    segment_dir = os.path.join(save_folder, 'segment')
    os.makedirs(segment_dir, exist_ok=True)
    
    for i in tqdm(range(len(data))):
        error_uid = []
        uid = data['video_id'][i]
        ytb_link = 'https://www.youtube.com/watch?v=' + uid

        os.system(f'yt-dlp -P {output_folder} -o "{uid}.wav" -x --audio-format wav {ytb_link}')

        output_dir = f'{output_folder}/{uid}.wav'
        if os.path.isfile(output_dir):
            #change stereo to mono
            os.system(f'ffmpeg -i {output_dir} -ac 1 {output_folder}/{uid}_1.wav -y')
            os.remove(output_dir)
            os.system(f'ffmpeg -i {output_folder}/{uid}_1.wav -ar 16000 {save_folder}/copy_{uid}_1.wav -loglevel panic') # sampling rate = 16000
            os.system(f'ffmpeg -i {save_folder}/copy_{uid}_1.wav -ss 60 -acodec copy {output_folder}/crop_{uid}_1.wav -loglevel panic') # crop 1 min
            os.system(f'ffmpeg -i {output_folder}/crop_{uid}_1.wav -f segment -segment_time {segment_time} -c copy {segment_dir}/{uid}_%03d.wav -loglevel panic') # segment wav file
            os.remove(f'{save_folder}/copy_{uid}_1.wav')
            os.remove(f'{output_folder}/crop_{uid}_1.wav')
            os.remove(sorted(glob.glob(f'{segment_dir}/{uid}*.wav'))[-1]) # remove last wav file (because len(wav_file) < segment_time)
            seg_list = sorted(glob.glob(f'{segment_dir}/{uid}*.wav'))
            for seg in seg_list:
                seg_name = seg.split('/')[-1][:-4]
                seg_file = seg.split('/')[-1]
                os.makedirs(os.path.join(segment_dir, seg_name), exist_ok=True)
                shutil.move(seg, os.path.join(segment_dir, seg_name,seg_file))

        if org_del:
            try:
                os.remove(f'{output_folder}/{uid}_1.wav')
            except:
                error_uid.append(uid)
    if org_del:
        os.rmdir(output_folder)
    error_uid_df = pd.DataFrame(error_uid, columns=['error_uid'])
    error_uid_df.to_csv(os.path.join(save_folder, 'error_uid.csv'), index=False)
        
def make_csv(args):
    segment_list = sorted(os.listdir(os.path.join(args.save_folder, 'segment')))
    data = pd.read_csv(args.data_name)
    video_id = list(map(lambda x: x[:11], segment_list))
    dic = {}
    for vid in data['video_id']:
        if vid in video_id:
            cnt = video_id.count(vid)
            dic[vid] = cnt
    new_rows = []
    for _, row in data.iterrows():
        video_id = row['video_id']
        if video_id in dic:
            for i in range(dic[video_id]):
                segment_id = f"{video_id}_{i:03}"
                new_rows.append([row['channel_id'], video_id, segment_id])
    new_df = pd.DataFrame(new_rows, columns=['channel_id', 'video_id', 'segment_id'])
    return new_df
    
def video_split(save_path, segment_time):
    output_path = os.path.join(save_path, 'full_video')
    os.makedirs(output_path, exist_ok=True)
    file_list=os.listdir(output_path)
    uni_file = sorted(set(list(map(lambda x: x[:11], file_list))))
    for idx, uid in tqdm(enumerate(uni_file)):
        
        try:
            
            save_file_name = glob.glob(os.path.join(output_path, f'{uid}.*'))[0]
            file_format = save_file_name.split('.')[-1]
            full_video_path=f'{output_path}/{uid}.{file_format}'
            
        except:
            with open(os.path.join(save_path, 'error_video_download.txt'),'a') as f:
                f.write(uid + '\n')
            pass
            
        if os.path.exists(full_video_path):
            uid_seg_paths=sorted(glob.glob(os.path.join(save_path, f'segment/{uid}_*')))
            uid_seg_list=[uid_seg_path.split('/')[-1] for uid_seg_path in uid_seg_paths]
            
            
            start_time=60
            segment_index=0
            # 비디오 전체 길이
            duration_cmd = f"ffprobe -v error -show_format -show_streams '{full_video_path}'"
            duration_output = os.popen(duration_cmd).read()
            duration_info = duration_output.split('\n')
            duration = None
            for line in duration_info:
                if "duration" in line:
                    duration = float(line.split('=')[1])
                    break
            if duration is None:
                 with open(os.path.join(save_path, 'error_video_download.txt'),'a') as f:
                    f.write(uid + '\n')
                 pass
                    
            while start_time<duration:
               # seg_name = os.path.basename(seg_video).split('.')[0]
                segment_filename = f"{uid}_{segment_index:03d}.mp4"
                segment_foldername=segment_filename.split('.')[0]
                if segment_foldername not in uid_seg_list:
                    start_time+=segment_time
                    segment_index+=1
                    continue
                os.system(f"ffmpeg -ss {start_time} -i {full_video_path} -t {segment_time} -r 25 -c:v libx264 {save_path}/segment/{segment_foldername}/{segment_filename} -y -loglevel panic")
                start_time+=segment_time
                segment_index+=1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='check_extract_video_list.csv')
    parser.add_argument('--save_path', type=str, default='tmp')
    parser.add_argument('--segment_time', type=int, default=60*10)
    parser.add_argument('--org_del', type=int, default=0)
    parser.add_argument('--max_workers', default=8, type=int)
    parser.add_argument('--mode', type=str)
    args = parser.parse_args()
    
    if args.mode == 'wav':
        download_wav_file(args.data_name, args.save_path, args.segment_time, args.org_del)
        segment_id_list = make_csv(args)
        segment_id_list.to_csv(os.path.join(args.save_folder, 'segment_id_list.csv'), index=False)
    elif args.mode == 'video':
        download_full_video(args.save_path)
        video_split(args.save_path, args.segment_time)
        
