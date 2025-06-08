import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.signal import savgol_filter
from tqdm import tqdm


class VENUSDataset(Dataset):
    def __init__(self, dataset, mode):
        self.dataset = dataset
        self.mode = mode
        if self.mode == 'face': 
            self._face_prepare_data()
        elif self.mode == 'body':
            self._body_prepare_data()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        
    def _face_prepare_data(self):
        param = []
        name_list = []

        for example in tqdm(self.dataset):
            facial_expression = example['facial_expression']
            utt_ids, frames, features = facial_expression['utt_id'], facial_expression['frame'], facial_expression['features'] 

            param_num = 53
            utt_param = []
            utt_name = []
            prev_utt_id = None
            
            for utt_id, frame, feature in zip(utt_ids, frames, features):
                utt_id = int(utt_id)
                frame = int(frame.split('_')[0])
                expr = feature[:50]
                jaw = feature[53:56]
                face_features = torch.tensor(np.concatenate([expr, jaw]))

                if prev_utt_id is None:
                    prev_utt_id = utt_id
                
                if prev_utt_id != utt_id:
                    paramdata = torch.cat(utt_param).reshape(-1, param_num)
                    paramdata_length = paramdata.shape[0]
                    if paramdata_length >= 3:  # minimum window_length >= 3
                        window_length = min(5, paramdata_length if paramdata_length % 2 == 1 else paramdata_length - 1) 
                        polyorder = min(2, window_length - 1)  # polyorder < window_length
                        filtered_param = torch.tensor(savgol_filter(paramdata, window_length, polyorder, axis=0))
                        param.append(filtered_param)
                    else:
                        param.append(paramdata)
                    name_list.append(utt_name)

                    utt_param = [face_features]
                    utt_name =  [f"{example['video_id']}_{example['segment_id']}_{utt_id}_{frame}"]
                    prev_utt_id = utt_id
                else:
                    utt_param.append(face_features)
                    utt_name.append(f"{example['video_id']}_{example['segment_id']}_{utt_id}_{frame}")

            if utt_param:
                paramdata = torch.cat(utt_param).reshape(-1, param_num)
                paramdata_length = paramdata.shape[0]
                if paramdata_length >= 3:
                    window_length = min(5, paramdata_length if paramdata_length % 2 == 1 else paramdata_length - 1)
                    polyorder = min(2, window_length - 1)
                    param.append(torch.tensor(savgol_filter(paramdata, window_length, polyorder, axis=0)))
                else:
                    param.append(paramdata)
                name_list.append(utt_name)

        self.param_data = param
        self.name_data = name_list

    def _body_prepare_data(self):
        param = []
        name_list = []

        for example in tqdm(self.dataset):
            body_language = example['body_language']
            utt_ids, frames, features = body_language['utt_id'], body_language['frame'], body_language['features'] 

            param_num = 117
            utt_param = []
            utt_name = []
            prev_utt_id = None
            
            for utt_id, frame, feature in zip(utt_ids, frames, features):
                utt_id = int(utt_id)
                frame = int(frame.split('_')[0])
                tmp = feature[39:156]
                upper_body = tmp[:27]
                lhand = tmp[27:72]
                rhand = tmp[72:]
                body_features = torch.tensor(np.concatenate([upper_body, lhand, rhand]))

                if prev_utt_id is None:
                    prev_utt_id = utt_id
                
                if prev_utt_id != utt_id:
                    paramdata = torch.cat(utt_param).reshape(-1, param_num)
                    paramdata_length = paramdata.shape[0]
                    if paramdata_length >= 3:  # minimum window_length >= 3
                        window_length = min(5, paramdata_length if paramdata_length % 2 == 1 else paramdata_length - 1)  # adjust odd number
                        polyorder = min(2, window_length - 1)  # polyorder <  window_length
                        filtered_param = torch.tensor(savgol_filter(paramdata, window_length, polyorder, axis=0))
                        param.append(filtered_param)
                    else:
                        param.append(paramdata)
                    name_list.append(utt_name)

                    utt_param = [body_features]
                    utt_name =  [f"{example['video_id']}_{example['segment_id']}_{utt_id}_{frame}"]
                    prev_utt_id = utt_id
                else:
                    utt_param.append(body_features)
                    utt_name.append(f"{example['video_id']}_{example['segment_id']}_{utt_id}_{frame}")

            if utt_param:
                paramdata = torch.cat(utt_param).reshape(-1, param_num)
                paramdata_length = paramdata.shape[0]
                if paramdata_length >= 3:
                    window_length = min(5, paramdata_length if paramdata_length % 2 == 1 else paramdata_length - 1)  
                    polyorder = min(2, window_length - 1)  
                    param.append(torch.tensor(savgol_filter(paramdata, window_length, polyorder, axis=0)))
                else:
                    param.append(paramdata)
                name_list.append(utt_name)

        self.param_data = param
        self.name_data = name_list


    def __len__(self):
        return len(self.param_data)
    
    def __getitem__(self, idx):
        return self.param_data[idx], self.name_data[idx]


def custom_collate_fn(batch):
    '''
    batch : [List] (#len utt frames, #dim param)
    input_tensor : (#batch, #window_size, #dim param) -> window_size = 512
    mask_tensor : (#batch, #window_size) -> window_size = 512
    '''
    # max_utt_length = max(utterance_param.shape[0] for utterance_param in batch)
    # max_utt_length = min(max_utt_length, 512) # window_size = 512
    max_utt_length = 512
    input_tensor = torch.zeros((len(batch), max_utt_length, batch[0][0].shape[-1])) # For zero padding
    mask_tensor = torch.zeros((len(batch), max_utt_length)) # For masking (not calculating loss)
    names_tensor = []

    for sample_index, batch_items in enumerate(batch):
        utt_params, names = batch_items
        L, E = utt_params.shape
        window_size = min(L, max_utt_length)
        input_tensor[sample_index, :window_size] = utt_params[:window_size, :]
        mask_tensor[sample_index, :window_size] = 1

        padded_names = names[:window_size] + [''] * (max_utt_length - len(names[:window_size]))
        names_tensor.append(padded_names)
        
    batch = {"inputs":input_tensor, "masks": mask_tensor, 'names': names_tensor}
                
    return batch

# if __name__ == '__main__'  :
#     from datasets import load_dataset
#     dataset = load_dataset('winston1214/VENUS-5K', split= 'train')
#     dataset = VENUSDataset(dataset, 'face')
