import os
from os.path import join as pjoin

import torch
from torch.utils.data import DataLoader
import wandb

from models.vq.model import RVQVAE
from models.vq.vq_trainer import RVQTokenizerTrainer
from options.vq_option import arg_parse
from utils.fixseed import fixseed
import numpy as np

from venus_dataset_huggingface import VENUSDataset, custom_collate_fn
from datasets import load_dataset
import pickle


os.environ["OMP_NUM_THREADS"] = "1"




def wandb_init(args):
    run = wandb.init(settings=wandb.Settings(_service_wait=300), project=args.project_name, name=args.exp_name, config=args)
    return run

if __name__ == "__main__":
    opt = arg_parse(True)
    print(opt)
    fixseed(opt.seed)
    if not hasattr(opt, 'no_wandb') or not opt.no_wandb:
        pass

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    print(f"Using Device: {opt.device}")

    opt.save_root = pjoin(opt.checkpoints_dir, opt.mode, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')

    os.makedirs(opt.model_dir, exist_ok=True)

    if opt.mode == "face":
        dim_pose = 53  # Expression(50) + Jaw(3)
    elif opt.mode == "body":
        dim_pose = 117  # Upper body + right hand + left hand
    elif opt.mode == "full":
        dim_pose = 170 

    else:
        raise KeyError('Dataset Does not Exists')
    if not hasattr(opt, 'no_wandb') or not opt.no_wandb:
        run = wandb_init(opt)
    net = RVQVAE(opt,
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

    pc_vq = sum(param.numel() for param in net.parameters())
    
    



    if os.path.exists(f'{opt.mode}_train_dataset.pkl'):
        print("Loading dataset from pkl")
        with open(f'{opt.mode}_train_dataset.pkl', 'rb') as f:
            train_dataset = pickle.load(f)
    if os.path.exists(f'{opt.mode}_test_dataset.pkl'):
        with open(f'{opt.mode}_test_dataset.pkl', 'rb') as f:
            test_dataset = pickle.load(f)
    else:
        print("New loading dataset")
        train_small_dataset = load_dataset("winston1214/VENUS-5K", split="train")
        test_small_dataset = load_dataset("winston1214/VENUS-5K", split="test")
        train_dataset = VENUSDataset(train_small_dataset, opt.mode)
        test_dataset = VENUSDataset(test_small_dataset, opt.mode)
        with open(f'{opt.mode}_train_dataset.pkl', 'wb') as f:
            pickle.dump(train_dataset, f)
        with open(f'{opt.mode}_test_dataset.pkl', 'wb') as f:
            pickle.dump(test_dataset, f)

    print("train_dataset: ", len(train_dataset))
    print("test_dataset: ", len(test_dataset))


    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=False, num_workers=8,
                              shuffle=True, pin_memory=True, collate_fn=custom_collate_fn)
    test_loader= DataLoader(test_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=8,
                            shuffle=False, pin_memory=True, collate_fn=custom_collate_fn)

    trainer = RVQTokenizerTrainer(opt, vq_model=net)

    try:
        trainer.train(train_loader, test_loader)
    except ZeroDivisionError as e:
        print("ZeroDivisionError occurred:", e)
    except Exception as e:
        print("An unexpected error occurred:", e)
