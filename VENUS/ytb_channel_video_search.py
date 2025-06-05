import os

# import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors

from datetime import datetime
import pandas as pd
import argparse
from itertools import product
from tqdm import tqdm



scopes = ["https://www.googleapis.com/auth/youtube.force-ssl"]
def channel_search(api_key,
                publishedAfter,
                publishedBefore,
                page_token=None,):
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    developer_key = api_key

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=developer_key)

    request = youtube.search().list(
        part="snippet",
        maxResults=50,
        q="podcast",
        regionCode="US",
        relevanceLanguage="en",
        pageToken=page_token,
        type="channel"
    )
    response = request.execute()
    
    return response

def search_videos_from_channel(channel_id: str,
                               video_N: int,
                               output_dir: str,):
    
    os.system(f'yt-dlp --flat-playlist --playlist-end {video_N} --print-to-file "id" {output_dir} "https://www.youtube.com/channel/{channel_id}/videos"')
    try:
        with open(output_dir,'r') as f:
            videos = f.readlines()
            videos = list(map(lambda x: x.strip(), videos))
        os.remove(output_dir)
    except:
        videos = None
    return videos


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_time',type=str,required=True,help='YYYYMMDD')
    parser.add_argument('--end_time',type=str,required=True, help='YYYYMMDD')
    parser.add_argument('--video_N',type=int,default = 100, help='video num')
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--api',type=str)
    parser.add_argument('--mode', type=str, help='channels or videos')
    args = parser.parse_args()
    print(args)

    if args.mode == 'channels':
        start_year, start_month, start_day = int(args.start_time[:4]), int(args.start_time[4:6]), int(args.start_time[6:])
        end_year, end_month, end_day = int(args.end_time[:4]), int(args.end_time[4:6]), int(args.end_time[6:])
        start_date = datetime(start_year, start_month, start_day)
        end_date = datetime(end_year, end_month, end_day)
        published_after = start_date.isoformat() + 'Z'
        publishedBefore = end_date.isoformat() + 'Z'
    
    
        trials = 50 # Day max trial is 50
        api_key = args.api
        channel_ids = []
        next_token_ls = []
        next_token = None
        for trial in tqdm(range(trials)):
            res_n = channel_search(api_key,
                                published_after,
                                publishedBefore,
                                page_token=next_token,)
    
            
            for res in res_n['items']:
                channel_ids.append(res['id']['channelId'])
                next_token_ls.append(None)
            try:
                next_token = res_n['nextPageToken']
                next_token_ls.pop(-1)
                next_token_ls.append(next_token)
            except:
                next_token_ls.pop(-1)
                next_token_ls.append('Done')
                break
    
        df = pd.DataFrame({'channel_id':channel_ids,'next_token':next_token_ls})
        df.to_csv(os.path.join(args.save_dir,f'{args.start_time}_{args.end_time}_channels.csv'), index=False)
  elif args.mode == 'videos':
        df = pd.read_csv(os.path.join(args.save_dir,f'{args.start_time}_{args.end_time}_channels.csv')

        dic = {}
        for channel in df['channel_id']:
            videos = search_videos_from_channel(channel, args.video_N, 'tmptmp.txt')
            if videos:
                dic[channel] = videos
        combinations = [(channel, video) for channel, videos in dic.items() for video in videos]
        data = pd.DataFrame(combinations, columns=['channel_id','video_id'])
        data.to_csv(os.path.join(args.save_dir,f'{args.start_time}_{args.end_time}_channels_videos.csv',index=False)
