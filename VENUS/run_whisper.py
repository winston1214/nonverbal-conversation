import os
import whisperx
import glob
from pyannote.audio import Pipeline
import torch
import pandas as pd
import argparse
from tqdm import tqdm

class ExtractAudio:
    def __init__(self, HF_TOKEN, use_whisper = False):
        self.HF_TOKEN = HF_TOKEN
        self.device = 'cuda'
        compute_type = 'float16'
        self.batch_size = 32
        model_name = "pyannote/speaker-diarization-3.0"
        if use_whisper:
            self.whisper_model = whisperx.load_model("large-v3", self.device, compute_type=compute_type)
        self.diarize_model = Pipeline.from_pretrained(model_name, use_auth_token=HF_TOKEN).to(torch.device(self.device))

    def pyannote_filtering(self,wav_file, SAMPLE_RATE = 16000):
        audio = whisperx.load_audio(wav_file)
        audio_data = {
            'waveform': torch.from_numpy(audio[None, :]),
            'sample_rate': SAMPLE_RATE
        }
        segments = self.diarize_model(audio_data)
        diarize_df = pd.DataFrame(segments.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
        diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)

        return diarize_df

    def run_whisper(self, wav_file):

        audio = whisperx.load_audio(wav_file) # not segment
        result = self.whisper_model.transcribe(audio, batch_size=self.batch_size)
        lang_ls = []
        lang = result['language']
        if lang == 'en':
            model_a, metadata = whisperx.load_align_model(language_code=lang, device=self.device)
            
            result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)
            diarize_segments = self.pyannote_filtering(wav_file)
            result = whisperx.assign_word_speakers(diarize_segments, result)

            result_seg = list(filter(lambda x: len(x) == 5, result['segments']))
            

            start_time = list(map(lambda x: x['start'], result_seg))
            end_time = list(map(lambda x: x['end'], result_seg))
            speaker = list(map(lambda x: x['speaker'], result_seg))
            text = list(map(lambda x: x['text'].strip(), result_seg))
            for _ in range(len(start_time)):
                lang_ls.append(lang)
            seq_df = pd.DataFrame({'start':start_time,'end':end_time,'speaker':speaker,'text':text, 'language':lang_ls})
            word_df = pd.DataFrame(result['word_segments'])
            return seq_df, word_df
        else:
            return None, None


                
        
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
                # end = row['end']
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
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--segment_csv', type=str, default='segment_id_list.csv')
    parser.add_argument('--filtering_file', type=str, default='filtering_file_list')
    parser.add_argument('--save_path', type=str, default='shard0/')
    parser.add_argument('--hf_token', type=str)
    args = parser.parse_args()

    HF_TOKEN = args.hf_token

    # pyannote filtering
    extractor = ExtractAudio(HF_TOKEN, use_whisper = 0)
    segment_file_csv = pd.read_csv(os.path.join(args.save_path, args.segment_csv))
    file_list = segment_file_csv['segment_id']
    
    filter_boolean = []
    num_speaker = []
    for wav_file_name in tqdm(file_list):
        wav_file = os.path.join(args.save_path, 'segment',wav_file_name, wav_file_name + '.wav')
        diarize_df = extractor.pyannote_filtering(wav_file)
        num_speaker.append(len(diarize_df['speaker'].unique()))
        if len(diarize_df['speaker'].unique()) == 2:
            filter_boolean.append(True)
        else:
            filter_boolean.append(False)
    file_df = pd.DataFrame({'file':file_list, 'filter_boolean':filter_boolean, 'num_speaker':num_speaker})
    file_df.to_csv(os.path.join(args.save_path, f'org_{args.filtering_file}.csv'), index=False) # optional (checking purpose)
    file_df = file_df[file_df['filter_boolean'] == True].reset_index(drop=True)
    file_df.drop('filter_boolean', axis=1, inplace=True)
    file_df.to_csv(os.path.join(args.save_path, f'{args.filtering_file}.csv'), index=False) # optional (checking purpose)

    # running whisper
    extractor = ExtractAudio(HF_TOKEN, use_whisper = 1)
    not_english = []
    for wav_name in tqdm(file_df['file']):
        wav_file = os.path.join(args.save_path, 'segment', wav_name, wav_name + '.wav')
        seq_df, word_df = extractor.run_whisper(wav_file)
        if seq_df is None:
            not_english.append(wav_name)
        else:    
            seq_df.to_csv(os.path.join(args.save_path, 'segment', wav_name, f'seq_{wav_name}.csv'), index=False)
            word_df.to_csv(os.path.join(args.save_path, 'segment', wav_name, f'word_{wav_name}.csv'), index=False)
            utterance_split(seq_df).to_csv(os.path.join(args.save_path, 'segment', wav_name, f'utterance_{wav_name}.csv'), index=False)
    not_english_df = pd.DataFrame({'name':not_english})
    not_english_df.to_csv(os.path.join(args.save_path, f'not_english.csv'), index=False)
