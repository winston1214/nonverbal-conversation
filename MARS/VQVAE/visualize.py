import torch
import os
from dataclasses import dataclass
from os.path import join as pjoin
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm
import pickle
from torch.utils.data import DataLoader
import tyro

from flame_pytorch.config import VertexArguments
from flame_pytorch.rendering import save_render
from models.vq.model import RVQVAE
from options.vq_option import VQVAEOptions
from flame_pytorch.config import VertexArguments
from venus_dataset_huggingface import VENUSDataset, custom_collate_fn
from flame_pytorch.osx_rendering import save_render_motion
import cv2
import smplx
@dataclass
class CombinedArgs( VQVAEOptions, VertexArguments):
    pass


def load_model(model_path, opt):  
    if opt.mode == "face":
        dim_pose = 53  # Expression(50) + Jaw(3)
    elif opt.mode == "body":
        dim_pose = 117  # Upper body + right hand + left hand
    elif opt.mode == "full":
        dim_pose = 170 

    model = RVQVAE(opt,
                 dim_pose,
                 opt.nb_code,
                 opt.code_dim,
                 opt.code_dim,
                 opt.down_t,
                 opt.stride_t,
                 opt.width,
                 opt.depth,
                 opt.dilation_growth_rate,
                 opt.vq_act,
                 opt.vq_norm)
    model.load_state_dict(torch.load(model_path)['vq_model'])
    model = model.to(opt.device)
    model.eval()
    return model


def save_face_render_video(model, data, save_path, opt):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, (opt.video_width * 2, opt.video_height))
    with torch.no_grad():
        inputs = data['inputs'].to(opt.device)
        masks = data['masks'].to(opt.device)
        pred_motion, _, _ = model(inputs, masks)
        inputs = inputs.squeeze(0)
        pred_motion = pred_motion.squeeze(0)
        for gt, pred in zip(inputs, pred_motion):
            pred_image = save_render(opt, pred.unsqueeze(0))
            gt_image = save_render(opt, gt.unsqueeze(0))
            draw_pred = ImageDraw.Draw(pred_image)
            draw_gt = ImageDraw.Draw(gt_image)
            draw_pred.text((100, 100), 'pred', fill='red')
            draw_gt.text((100, 100), 'gt', fill='blue')

            image = Image.new('RGB', (pred_image.width * 2, pred_image.height), (255, 255, 255))
            image.paste(pred_image, (0, 0))
            image.paste(gt_image, (pred_image.width, 0))
            video_writer.write(np.array(image))
    video_writer.release()
    print(f'{save_path} saved!')

def save_body_render_video(model, data, save_path, opt, smplx_model_dir):
    smplx_model = smplx.create(smplx_model_dir, model_type='smplx', gender='neutral', use_face_contour=True, ext='pkl', use_pca=False, num_expression_coeffs=50).to(opt.device)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, (opt.video_width * 2, opt.video_height))
    with torch.no_grad():
        inputs = data['inputs'].to(opt.device)
        masks = data['masks'].to(opt.device)
        pred_motion, _, _ = model(inputs, masks)
        inputs = inputs.squeeze(0)
        pred_motion = pred_motion.squeeze(0)
        for gt, pred in tqdm(zip(inputs, pred_motion)):
            pred_image = save_render_motion(opt, pred.unsqueeze(0), smplx_model, mode='osx')
            gt_image = save_render_motion(opt, gt.unsqueeze(0), smplx_model, mode='osx')
            draw_pred = ImageDraw.Draw(pred_image)
            draw_gt = ImageDraw.Draw(gt_image)
            draw_pred.text((100, 100), 'pred', fill='red')
            draw_gt.text((100, 100), 'gt', fill='blue')

            image = Image.new('RGB', (pred_image.width * 2, pred_image.height), (255, 255, 255))
            image.paste(pred_image, (0, 0))
            image.paste(gt_image, (pred_image.width, 0))
            video_writer.write(np.array(image))
    video_writer.release()
    print(f'{save_path} saved!')


if __name__ == "__main__":
    opt = tyro.cli(CombinedArgs)
    opt.save_root = pjoin(opt.checkpoints_dir, opt.mode, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    with open(f'{opt.mode}_test_dataset.pkl', 'rb') as f:
        test_dataset = pickle.load(f)
    random_idx = np.random.randint(0, len(test_dataset))
    # random_idx = 1943/
    print(random_idx)
    
    random_choice_dataset = custom_collate_fn([test_dataset[random_idx]])

    model = load_model(pjoin(opt.model_dir, 'best.tar'), opt)
    save_name = test_dataset[random_idx][1][0][:15]
    os.makedirs(pjoin(opt.save_root, 'visualize'), exist_ok=True)
    save_path = os.path.join(opt.save_root, 'visualize', f'{save_name}.mp4')
    if opt.mode == 'face':
        save_face_render_video(model, random_choice_dataset, save_path, opt)
    elif opt.mode == 'body':
        smplx_model_dir = '../../VENUS/OSX/common/utils/human_model_files'
        # smplx_model = smplx.create(smplx_model_dir, model_type='smplx', gender='neutral', use_face_contour=True, ext='pkl', use_pca=False, num_expression_coeffs=50).to(opt.device)
        save_body_render_video(model, random_choice_dataset, save_path, opt, smplx_model_dir)






