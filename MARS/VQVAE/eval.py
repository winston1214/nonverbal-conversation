import os
import sys
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models.vq.model import RVQVAE
from options.vq_option import VQVAEOptions
from venus_dataset_huggingface import VENUSDataset, custom_collate_fn
from metrics import *
import tyro
from dataclasses import dataclass
from flame_pytorch.config import VertexArguments
from os.path import join as pjoin

from tqdm import tqdm
os.environ["OMP_NUM_THREADS"] = "1"
import pickle
import math
import numpy as np



def evaluate_codebook_usage_from_data(model, dataloader, nb_code, device):

    model.eval()
    num_layers = len(model.quantizer.layers)
    
    # Collect all codebook indices for each layer
    all_indices = [[] for _ in range(num_layers)]
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = batch['inputs'].to(device)
            masks = batch['masks'].to(device)
            
            # Get the codebook indices for each layer by using the quantizer
            code_idx, _ = model.quantizer.quantize(model.encoder(model.preprocess(inputs)), 
                                                 mask=model.create_encoder_mask(masks),
                                                 return_latent=True)
            
            # code_idx 형태: [B, T, num_layers]
            for i in range(num_layers):
                layer_indices = code_idx[..., i]
                all_indices[i].append(layer_indices.cpu())
    # Calculate the codebook usage statistics for each layer
    layer_stats = []
    total_dead_codes = 0
    
    for layer_idx in range(num_layers):

        layer_indices = torch.cat(all_indices[layer_idx], dim=0).flatten()
        
        # Calculate the usage ratio for each codebook index
        counts = torch.bincount(layer_indices, minlength=nb_code)
        usage_ratio = counts.float() / (counts.sum() + 1e-10)
        
        # Calculate the entropy
        entropy = -(usage_ratio * torch.log(usage_ratio + 1e-10)).sum().item()
        normalized_entropy = entropy / math.log(nb_code)
        
        # Calculate the perplexity
        perplexity = torch.exp(torch.tensor(entropy)).item()
        
        # The number of dead codes
        dead_codes = (counts == 0).sum().item()
        total_dead_codes += dead_codes
        
        
        
        layer_stats.append({
            'usage_ratio': usage_ratio.numpy(),
            'normalized_entropy': normalized_entropy,
            'perplexity': perplexity,
            'dead_codes': dead_codes,
            'utilization_rate': 1.0 - (dead_codes / nb_code)
        })
    
    
    mean_entropy = np.mean([stats['normalized_entropy'] for stats in layer_stats])
    mean_perplexity = np.mean([stats['perplexity'] for stats in layer_stats])
    mean_perplexity_ratio = mean_perplexity / nb_code
    total_codes = num_layers * nb_code
    utilization_rate = 1.0 - (total_dead_codes / total_codes)
    

    
    return {
        'mean_entropy': mean_entropy,
        'mean_perplexity': mean_perplexity,
        'mean_perplexity_ratio': mean_perplexity_ratio,
        'utilization_rate': utilization_rate
    }


@dataclass
class CombinedArgs(VQVAEOptions, VertexArguments):
    pass

def load_model(model_path):  
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


opt = tyro.cli(CombinedArgs)
print(opt)
if opt.mode == 'face':
    dim_pose = 53
elif opt.mode == 'body':
    dim_pose = 117


opt.save_root = pjoin(opt.checkpoints_dir, opt.mode, opt.name)
opt.model_dir = pjoin(opt.save_root, 'model')
opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))

model = load_model(pjoin(opt.model_dir, 'best.tar')) # latest.tar?


with open(f'{opt.mode}_test_dataset.pkl', 'rb') as f:
    test_dataset = pickle.load(f)
test_loader = DataLoader(test_dataset, batch_size=opt.vq_batch_size, drop_last=False, num_workers=8, collate_fn=custom_collate_fn, shuffle=False)

usage_stats = evaluate_codebook_usage_from_data(model, test_loader, opt.nb_code, opt.device)

total_l2 = []
total_fid = []
total_diversity = []
total_diversity_gt = []
total_var = []
total_var_gt = []
total_windowed_l2v = []  # New list for windowed L2V
processed = []
total_vertex_mse = []
total_vertex_mae = []
gt_motion_list = []
pred_motion_list = []

vertex_metric = VertexMetric(opt, opt.mode, opt.device)
feature_metric = FeatureMetric(opt)

for batch in tqdm(test_loader):
    inputs = batch['inputs'].to(opt.device)  # Ensure inputs are on the same device
    masks = batch['masks'].to(opt.device)
    with torch.no_grad():
        pred_motion, loss_commit, perplexity = model(inputs, masks)

        gt_motion_gpu = inputs[masks == 1]
        pred_motion_gpu = pred_motion[masks == 1]
        
        vmse, vmae = vertex_metric.calculate_vertex_metric(gt_motion_gpu, pred_motion_gpu)
        
        del gt_motion_gpu
        del pred_motion_gpu
        torch.cuda.empty_cache()
        total_vertex_mse.append(vmse)
        total_vertex_mae.append(vmae)
        gt_motion = inputs[masks == 1].cpu().detach().numpy()
        valid_preds = pred_motion[masks == 1].cpu().detach().numpy()


    batch_metrics = feature_metric.calculate_all_metrics(gt_motion, valid_preds)
    total_l2.append(batch_metrics['mse'])
    total_fid.append(batch_metrics['fid'])
    total_diversity_gt.append(batch_metrics['diversity_gt'])
    total_diversity.append(batch_metrics['diversity'])
    total_windowed_l2v.append(batch_metrics['windowed_l2v'])
    total_var_gt.append(batch_metrics['var_gt'])
    total_var.append(batch_metrics['var'])

print('codebook usage entropy', usage_stats['mean_entropy'])
print('codebook usage rate', usage_stats['utilization_rate'])
print("l2", np.mean(np.array(total_l2))) 
print("windowed avg.l2v", np.mean(np.array(total_windowed_l2v)))  # L2 Affect 
print("fid", np.mean(np.array(total_fid)))
print("diversity", np.mean(np.array(total_diversity)))
print("diversity GT", np.mean(np.array(total_diversity_gt)))
print("var", np.mean(np.array(total_var)))
print("var GT", np.mean(np.array(total_var_gt)))
print("vertex mse", np.mean(np.array(total_vertex_mse)))
print("vertex mae", np.mean(np.array(total_vertex_mae)))