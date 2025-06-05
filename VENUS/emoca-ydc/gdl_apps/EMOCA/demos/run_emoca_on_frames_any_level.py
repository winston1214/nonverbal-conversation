import torch
import time
import os
import numpy as np
from torch.utils.data.dataloader import DataLoader
# Keep in mind that we are using face_alignment_/face_alignment_1 as face_alignment
from face_alignment_ import face_alignment_1 as face_alignment
from bbox_utils import *
import argparse
from landmark_utils import *
# from gdl_apps.EMOCA.utils.io import save_obj, save_images, save_codes, test
from gdl_apps.EMOCA.utils.load import load_model
import gdl
from pathlib import Path
from tqdm import auto
import torchvision.transforms as t 
import torchvision
from tqdm import tqdm
import glob
from collections import defaultdict
import natsort
import pickle
def pad_to_square(image):

    _, height, width = image.shape
    
    new_size = 512
    scale = new_size / max(width, height)
    
    
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = t.Resize((new_height, new_width))(image)
    
    
    padding_left = (new_size - new_width) // 2
    padding_right = new_size - new_width - padding_left
    padding_top = (new_size - new_height) // 2
    padding_bottom = new_size - new_height - padding_top
    
    
    padded_image = t.Pad((padding_left, padding_top, padding_right, padding_bottom))(resized_image)
    
    return padded_image


def print_gpu_memory(device):
    # in bytes
    free_memory,total_memory = torch.cuda.mem_get_info(device=device)
    # in GB
    free_memory_gb = free_memory / (1024 ** 3)
    total_memory_gb = total_memory / (1024 ** 3)
    used_memory_gb = total_memory_gb - free_memory_gb
    memory_percentage = (used_memory_gb / total_memory_gb) * 100
    print(f'Used GPU Memory: {used_memory_gb:.3f} GB ({memory_percentage:.2f}%)')
    
def print_shape(any_obj, name=None):
    """
    print shape of any object (torch.Tensor, np.ndarray, list, dict)
    name : name of any object in string
    """
    if name:
        print(f'---shape of {name}---')
    if isinstance(any_obj, list):
        print(f'list len: {len(any_obj)}')
        if any_obj and (isinstance(any_obj[0], torch.Tensor) or isinstance(any_obj[0], np.ndarray)):
            print(f'element type: {type(any_obj[0])}')
            print(f'element shape: {any_obj[0].shape}')
    elif isinstance(any_obj, (torch.Tensor, np.ndarray)):
        print(f'shape: {any_obj.shape}')
    elif isinstance(any_obj, dict):
        print(f'dict len: {len(any_obj)}')
        print(f'keys: {any_obj.keys()}')
        for k, v in any_obj.items():
            print_shape(v, k)
    else:
        print(f'unknown type: {type(any_obj)}')
    

def sfd_to_fan_bbox_batch(sfd_bboxlist):
    '''
    sfd_bboxlist : list of torch.Tensor (BS,1,5)
    '''
    fan_bboxes = []
    centers = []
    scales = []
    for sfd_bbox in sfd_bboxlist :
        sfd_bbox = sfd_bbox[0]
        bbox, center, scale = sfd_bbox_to_fan_bbox(sfd_bbox[:4])
        fan_bboxes.append(bbox)
        centers.append(center)
        scales.append(scale)
    fan_bboxes = torch.stack(fan_bboxes, dim=0)
    centers = torch.stack(centers, dim=0)
    scales = torch.stack(scales, dim=0).unsqueeze(-1)
    return fan_bboxes, centers, scales

@ torch.no_grad()
def get_2d_landmarks_batch(cropped_batch, device, centers, scales, face_model):
    cropped_batch.div_(255.0)
    heatmap_batch = face_model.face_alignment_net(cropped_batch) # (BS,68,64,64)
    pts, pts_img, scores = get_preds_fromhm(heatmap_batch, device, centers, scales)
    pts_img = pts_img.view(cropped_batch.shape[0], 68, 2)
    return pts_img


def crop_ori_image_with_landmarks(landmarks, img_batch):
    # define boundingboxes
    left = torch.min(landmarks[:,:,0],dim=1).values
    right = torch.max(landmarks[:,:,0],dim=1).values
    top = torch.min(landmarks[:,:,1],dim=1).values
    bottom = torch.max(landmarks[:,:,1],dim=1).values
    
    # crop image
    img_batch = img_batch / 255.
    old_size, center = bbox2point(left, right, top, bottom)
    size = (old_size*1.25).int()
    dst_image = bbpoint_warp_resize(img_batch, center, size, 224, landmarks=landmarks)
    if len(dst_image.shape) == 3:
        dst_image = dst_image.view(1,3,224,224)
    dst_batch = {"image": dst_image}
    return dst_batch

@ torch.no_grad()
def run_emocav2_on_frames(args, emoca, face_model,img_paths, device, batch_size =16, transform=None) :
    grouped = defaultdict(list)
    for file_path in img_paths:
        group_key = '/'.join(file_path.split('/')[:-2])
        grouped[group_key].append(file_path.strip())
    all_keys = list(grouped.keys())
    n = len(all_keys)
    chunk_size = (n + 3) // 4
    start_idx = chunk_size * args.partition
    end_idx = min(start_idx + chunk_size, n)
    selected_keys = all_keys[start_idx:end_idx]
    new_grouped = {k: grouped[k] for k in selected_keys}

    if args.rest:
        keys_list = list(new_grouped.keys())[args.rest:]
        new_grouped = {key: new_grouped[key] for key in keys_list}


    for group_key, group_img_paths in tqdm(new_grouped.items(), desc="Processing", unit="step"):
        group_img_paths = natsort.natsorted(group_img_paths)
        image_loader = DataLoader(group_img_paths, batch_size=batch_size, shuffle=False, drop_last = False)
        utt_flame_results = []
        utt_num = os.path.basename(group_key)

        for idx, batch in enumerate(image_loader) :
            tensor_batch=[]
            valid_paths = []
            for img_path in batch :
                try:
                    image = torchvision.io.read_image(img_path.strip())
                    if transform is not None:
                        image = transform(image)

                    tensor_batch+=[image]
                    valid_paths.append(img_path)
                except:
                    continue
            if len(tensor_batch) == 0:
                continue
            img_batch = torch.stack(tensor_batch, dim=0).to(device, dtype=torch.float32)

            sfd_bboxlist = face_model.face_detector.detect_from_batch(img_batch)

            # sfd to fan bbox
            try:
                fan_bboxes, centers, scales = sfd_to_fan_bbox_batch(sfd_bboxlist)
            except:
                for value in sfd_bboxlist:
                    if value:
                        previous_value = value
                        break

                for i, value in enumerate(sfd_bboxlist):
                    if not value:
                        sfd_bboxlist[i] = previous_value  
                    else:
                        previous_value = value 

                fan_bboxes, centers, scales = sfd_to_fan_bbox_batch(sfd_bboxlist)

            
            # crop detected faces with fan_bboxes
            cropped_batch = crop_batch(img_batch, fan_bboxes)

            pts_img = get_2d_landmarks_batch(cropped_batch, device, centers, scales,face_model)

            dst_batch = crop_ori_image_with_landmarks(pts_img, img_batch)

            vals = emoca.encode(dst_batch, training=False)
            vals = emoca.decode(vals, training=False)

            
            # save flame params
            # save_img_paths = [idx*batch_size:(idx+1)*batch_size]
            for idx, save_path in enumerate(batch):
                if idx < len(valid_paths):
                    flame_param = torch.cat([vals["expcode"][idx], vals["posecode"][idx], vals["shapecode"][idx]])
                    frame_number = os.path.basename(save_path).split('.')[0]
                    np_flame = flame_param.cpu().numpy()
                    utt_flame_results.append((utt_num, frame_number, np_flame))
        os.makedirs(os.path.join(group_key, 'face'), exist_ok=True)
        with open(os.path.join(group_key, 'face', utt_num+'.pkl'), 'wb') as f:
            pickle.dump(utt_flame_results, f)
                
    return 'success'

        

def decode(emoca, values, training=False):

    with torch.no_grad():
        values = emoca.decode(values, training=training)
        uv_detail_normals = None
        if 'uv_detail_normals' in values.keys():
            uv_detail_normals = values['uv_detail_normals']
        visualizations, grid_image = emoca._visualization_checkpoint(
            values['verts'],
            values['trans_verts'],
            values['ops'],
            uv_detail_normals,
            values, 
            0,
            "",
            "",
            save=False
        )

    return values, visualizations

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
    
def parse_args():
    pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = pythonpath.split(':')[0]
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, help="directory path with frames to reconstruct.", required=True)
    parser.add_argument('--path_to_models', type=str, default=os.path.join(pythonpath, "assets/EMOCA/models"))
    parser.add_argument('--mode', type=str, default="detail", choices=["detail", "coarse"], help="Which model to use for the reconstruction.")
    parser.add_argument('--save_images', type=str2bool, default=False, help="If true, output images will be saved")
    parser.add_argument('--save_codes', type=str2bool, default=True, help="If true, output FLAME values for shape, expression, jaw pose will be saved")
    parser.add_argument('--save_mesh', type=str2bool, default=False, help="If true, output meshes will be saved")
    parser.add_argument('--model_name', type=str, default='EMOCA_v2_lr_mse_20', help='Name of the model to use. Currently EMOCA or DECA are available.')
    # add a string argument with several options for image type
    parser.add_argument('--image_type', type=str, default='geometry_detail', 
        choices=["geometry_detail", "geometry_coarse", "out_im_detail", "out_im_coarse"], 
        help="Which image to use for the reconstruction video.")
    parser.add_argument('--processed_subfolder', type=str, default=None, 
        help="If you want to resume previously interrupted computation over a video, make sure you specify" \
            "the subfolder where the got unpacked. It will be in format 'processed_%Y_%b_%d_%H-%M-%S'")
    parser.add_argument('--cat_dim', type=int, default=0, 
        help="The result video will be concatenated vertically if 0 and horizontally if 1")
    parser.add_argument('--include_rec', type=str2bool, default=True, 
        help="The reconstruction (non-transparent) will be in the video if True")
    parser.add_argument('--include_transparent', type=str2bool, default=True, 
        help="Apart from the reconstruction video, also a video with the transparent mesh will be added")
    parser.add_argument('--include_original', type=str2bool, default=True, 
        help="Apart from the reconstruction video, also a video with the transparent mesh will be added")
    parser.add_argument('--black_background', type=str2bool, default=False, help="If true, the background of the reconstruction video will be black")
    parser.add_argument('--use_mask', type=str2bool, default=True, help="If true, the background of the reconstruction video will be black")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--disable_exception', action='store_true', default=False, help="If true, the exception will not be saved, use it when debugging")
    parser.add_argument('--print_time', action='store_true', default=False, help="If true print all time information")
    parser.add_argument('--print_shape', action='store_true', default=False, help="If true print shape of all tensors")
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=100)
    parser.add_argument('--partition', type=int, default=0)
    parser.add_argument('--rest', type=int, default=0)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    device = args.device
    print(f'device {device} is used')
    ## args
    path_to_models = args.path_to_models
    model_name = args.model_name
    mode = args.mode
    
    print(args)

    print("loading face alignment model...")
    face_alignment_model = face_alignment.FaceAlignment_(face_alignment.LandmarksType.TWO_D, device=str(device), 
                                                flip_input=False, face_detector='sfd',
                                                face_detector_kwargs={"filter_threshold":0.9})
    emoca, conf = load_model(path_to_models, model_name, mode)
    emoca.to(device)
    emoca.eval()
    with open(os.path.join(args.save_path, f'img_paths_{args.start_idx}_{args.end_idx}.txt'), 'r') as f:
        img_paths = f.readlines()

    transform = t.Compose([t.Lambda(pad_to_square)])
    
    run_emocav2_on_frames(args, emoca, face_alignment_model, img_paths, device, args.batch_size, transform = transform)
    print('DONE')


if __name__ == "__main__":
    
    main()