import os
import sys
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
from config import cfg
import cv2
from tqdm import tqdm
import torchvision
import natsort
import glob
import pandas as pd
import pickle
import shutil
from collections import defaultdict

def pad_to_square(image):

    _, height, width = image.shape
    
    new_size = 512
    scale = new_size / max(width, height)
    

    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = transforms.Resize((new_height, new_width))(image)
    

    padding_left = (new_size - new_width) // 2
    padding_right = new_size - new_width - padding_left
    padding_top = (new_size - new_height) // 2
    padding_bottom = new_size - new_height - padding_top
    
    padded_image = transforms.Pad((padding_left, padding_top, padding_right, padding_bottom))(resized_image)
    
    return padded_image



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids', default='0')
    parser.add_argument('--input_folder', type=str, default='frames/')
    parser.add_argument('--output_folder', type=str, default='output')
    parser.add_argument('--encoder_setting', type=str, default='osx_l', choices=['osx_b', 'osx_l'])
    parser.add_argument('--decoder_setting', type=str, default='normal', choices=['normal', 'wo_face_decoder', 'wo_decoder'])
    parser.add_argument('--pretrained_model_path', type=str, default='../pretrained_models/osx_l.pth.tar')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--visualize', action='store_true', help='choose whether or not to visualize mesh')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=100)
    parser.add_argument('--partition', type=int, default=0)
    parser.add_argument('--rest', type=int, default=0)
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    return args

def main(img_paths, transform, args) :
    grouped = defaultdict(list)
    for file_path in img_paths:
        group_key = '/'.join(file_path.split('/')[:-2])
        grouped[group_key].append(file_path.strip())
    all_keys = list(grouped.keys())
    n = len(all_keys)
    chunk_size = (n + 5) // 6 
    start_idx = chunk_size * args.partition
    end_idx = min(start_idx + chunk_size, n)
    selected_keys = all_keys[start_idx:end_idx]
    new_grouped = {k: grouped[k] for k in selected_keys}

    if args.rest:
        keys_list = list(new_grouped.keys())[args.rest:]
        new_grouped = {key: new_grouped[key] for key in keys_list}

    for group_key, group_img_paths in tqdm(new_grouped.items(), desc="Processing", unit="step"):
        group_img_paths = natsort.natsorted(group_img_paths)
        processed_count = (len(group_img_paths) // args.batch_size) * args.batch_size
        rest_group_img_paths = group_img_paths[processed_count:]
        
        data_loader = torch.utils.data.DataLoader(group_img_paths, batch_size=args.batch_size, drop_last = True)
        rest_dataloader = torch.utils.data.DataLoader(rest_group_img_paths, batch_size=1, drop_last = False)
        os.makedirs(os.path.join(group_key, 'osx'), exist_ok=True)
        utt_osx_result = []
        utt_num = os.path.basename(group_key)

        for _, batch_img_paths in enumerate(data_loader) :
            batch_images = []
            valid_image_paths = []
            for img_path in batch_img_paths :
                try:
                    torch_img = torchvision.io.read_image(img_path)
                    batch_images.append(torch_img)
                    valid_image_paths.append(img_path)
                except:
                    pass
            if len(batch_images) == 0:
                continue
            images = []
            for batch_image in batch_images :
                images.append(transform(batch_image).to(torch.float32)/255)
            images = torch.stack(images, dim=0)
            images = images.cuda()
            inputs = {'img': images}
            targets = {}
            meta_info = {}

            with torch.no_grad():

                real_batch_size = images.shape[0]
                if real_batch_size % args.batch_size != 0:
                    padding_size = args.batch_size - (real_batch_size % args.batch_size)
                    padding = torch.zeros((padding_size, *images.shape[1:]), device=images.device)
                    padded_images = torch.cat([images, padding], dim=0)
                    inputs = {'img': padded_images}
                    out = demoer.model(inputs, targets, meta_info, 'pose_inference')
                    for key in out.keys():
                        out[key] = out[key][:real_batch_size]
                    
                else:
                    out = demoer.model(inputs, targets, meta_info, 'pose_inference')
            for i, batch_img in enumerate(valid_image_paths) :
                params = {}
                param_numpy = np.array([])
                for key in out.keys():
                    params = out[key][i].detach().cpu().numpy()
                    param_numpy = np.concatenate((param_numpy, params))
                frame_num = os.path.basename(batch_img).split('.')[0]
                utt_osx_result.append((utt_num, frame_num, param_numpy))

        ## rest process
        for _, rest_batch_img_paths in enumerate(rest_dataloader) :
            batch_images = []
            for img_path in rest_batch_img_paths :
                try:
                    torch_img = torchvision.io.read_image(img_path)
                    batch_images.append(torch_img)
                except:
                    continue
            if len(batch_images) == 0:
                continue
            images = []
            for batch_image in batch_images :
                images.append(transform(batch_image).to(torch.float32)/255)
            images = torch.stack(images, dim=0)
            images = images.cuda()
            inputs = {'img': images}
            targets = {}
            meta_info = {}

            with torch.no_grad():
                out = demoer.model(inputs, targets, meta_info, 'pose_inference') # param dict

            for i, rest_batch_img in enumerate(rest_batch_img_paths) :
                params = {}
                param_numpy = np.array([])
                for key in out.keys():
                    params = out[key][i].detach().cpu().numpy()
                    param_numpy = np.concatenate((param_numpy, params))
                frame_num = os.path.basename(rest_batch_img).split('.')[0]
                utt_osx_result.append((utt_num, frame_num, param_numpy))
        
        
        save_path = os.path.join(group_key, 'osx', str(utt_num)+'.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(utt_osx_result, f)
        

if __name__ == "__main__" :

    args = parse_args()
    cfg.set_args(args.gpu_ids)
    cudnn.benchmark = True

    # load model
    cfg.set_additional_args(encoder_setting=args.encoder_setting, decoder_setting=args.decoder_setting, pretrained_model_path=args.pretrained_model_path)
    from common.base import Demoer
    demoer = Demoer()
    demoer._make_model()
    model_path = args.pretrained_model_path
    assert osp.exists(model_path), 'Cannot find model at ' + model_path
    print('Load checkpoint from {}'.format(model_path))

    demoer.model.eval()
    
    # file_df = pd.read_csv(os.path.join(args.input_folder, 'segment.csv'))
    # file_list = sorted(file_df['segment_id'].tolist())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(args.input_folder, f'img_paths_{args.start_idx}_{args.end_idx}.txt'), 'r') as f:
        img_paths = f.readlines()
    transform = transforms.Compose([transforms.Lambda(pad_to_square)])
    main(img_paths, transform, args)


    