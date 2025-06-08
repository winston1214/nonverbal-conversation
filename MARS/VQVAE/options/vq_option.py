import os
from dataclasses import dataclass, field
from typing import List, Optional, Literal
import torch
import tyro


@dataclass
class VQVAEOptions:
    """VQVAE model options"""
    
    # Data loader options
    mode: str = field(default="face", metadata={"help": "Dataset mode (Directory)", "tyro_name": "mode"})
    vq_batch_size: int = field(default=256, metadata={"help": "Batch size", "tyro_name": "batch-size"})
    window_size: int = field(default=64, metadata={"help": "Learning motion length", "tyro_name": "window-size"})
    gpu_id: int = field(default=0, metadata={"help": "GPU ID", "tyro_name": "gpu-id"})
    
    # Optimization options
    max_epoch: int = field(default=100, metadata={"help": "Total number of epochs to train", "tyro_name": "max-epoch"})
    warm_up_iter: int = field(default=2000, metadata={"help": "Warm-up iterations", "tyro_name": "warm-up-iter"})
    lr: float = field(default=2e-4, metadata={"help": "Maximum learning rate", "tyro_name": "lr"})
    milestones: List[int] = field(default_factory=lambda: [150000, 250000], metadata={"help": "Learning rate schedule (iterations)", "tyro_name": "milestones"})
    gamma: float = field(default=0.1, metadata={"help": "Learning rate decay rate", "tyro_name": "gamma"})
    
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay", "tyro_name": "weight-decay"})
    commit: float = field(default=0.02, metadata={"help": "Commitment loss hyperparameter", "tyro_name": "commit"})
    loss_vel: float = field(default=0.5, metadata={"help": "Velocity loss hyperparameter", "tyro_name": "loss-vel"})
    recons_loss: str = field(default="l1_smooth", metadata={"help": "Reconstruction loss", "tyro_name": "recons-loss"})
    velocity_loss: str = field(default="l1_smooth", metadata={"help": "Velocity loss", "tyro_name": "velocity-loss"})
    jaw_lambda: float = field(default=1.0, metadata={"help": "Jaw lambda", "tyro_name": "jaw-lambda"})
    expr_lambda: float = field(default=1.0, metadata={"help": "Expression lambda", "tyro_name": "expr-lambda"})
    body_lambda: float = field(default=1.0, metadata={"help": "Body lambda", "tyro_name": "body-lambda"})
    hands_lambda: float = field(default=1.0, metadata={"help": "Hands lambda", "tyro_name": "hands-lambda"})
    recon_lambda: float = field(default=1.0, metadata={"help": "Reconstruction lambda", "tyro_name": "recon-lambda"})
    vel_expr_lambda: float = field(default=1.0, metadata={"help": "Expression lambda for velocity loss", "tyro_name": "vel-expr-lambda"})
    vel_jaw_lambda: float = field(default=1.0, metadata={"help": "Jaw lambda for velocity loss", "tyro_name": "vel-jaw-lambda"})
    vel_body_lambda: float = field(default=1.0, metadata={"help": "Body lambda for velocity loss", "tyro_name": "vel-body-lambda"})
    vel_hands_lambda: float = field(default=1.0, metadata={"help": "Hands lambda for velocity loss", "tyro_name": "vel-hands-lambda"})
    reg_lambda: float = field(default=0.0, metadata={"help": "Regularization lambda", "tyro_name": "reg-lambda"})
    
    # VQVAE architecture options
    code_dim: int = field(default=512, metadata={"help": "Embedding dimension", "tyro_name": "code-dim"})
    nb_code: int = field(default=512, metadata={"help": "Number of codebooks", "tyro_name": "nb-code"})
    mu: float = field(default=0.99, metadata={"help": "Exponential moving average for codebook updates", "tyro_name": "mu"})
    down_t: int = field(default=3, metadata={"help": "Downsampling ratio", "tyro_name": "down-t"})
    stride_t: int = field(default=2, metadata={"help": "Stride size", "tyro_name": "stride-t"})
    width: int = field(default=512, metadata={"help": "Network width", "tyro_name": "width"})
    depth: int = field(default=3, metadata={"help": "Number of res blocks", "tyro_name": "depth"})
    dilation_growth_rate: int = field(default=3, metadata={"help": "Dilation growth rate", "tyro_name": "dilation-growth-rate"})
    output_emb_width: int = field(default=512, metadata={"help": "Output embedding width", "tyro_name": "output-emb-width"})
    vq_act: Literal["relu", "silu", "gelu"] = field(default="relu", metadata={"help": "Activation function", "tyro_name": "vq-act"})
    vq_norm: Optional[str] = field(default=None, metadata={"help": "Normalization method", "tyro_name": "vq-norm"})
    
    num_quantizers: int = field(default=1, metadata={"help": "Number of quantizers", "tyro_name": "num-quantizers"})
    shared_codebook: bool = field(default=False, metadata={"help": "Shared codebook", "tyro_name": "shared-codebook"})
    quantize_dropout_prob: float = field(default=0.2, metadata={"help": "Quantization dropout probability", "tyro_name": "quantize-dropout-prob"})
    # use_vq_prob: float = field(default=0.8, metadata={"help": "Probability of quantization", "tyro_name": "use-vq-prob"})
    
    ext: str = field(default="default", metadata={"help": "Reconstruction loss", "tyro_name": "ext"})
    
    # Other options
    name: str = field(default="test", metadata={"help": "Experiment name", "tyro_name": "name"})
    is_continue: bool = field(default=False, metadata={"help": "Continue training", "tyro_name": "is-continue"})
    checkpoints_dir: str = field(default="./checkpoints", metadata={"help": "Model save location", "tyro_name": "checkpoints-dir"})
    log_every: int = field(default=10, metadata={"help": "Log recording frequency", "tyro_name": "log-every"})
    save_latest: int = field(default=500, metadata={"help": "Latest model save frequency", "tyro_name": "save-latest"})
    save_every_e: int = field(default=2, metadata={"help": "Save model every n epochs", "tyro_name": "save-every-e"})
    eval_every_e: int = field(default=1, metadata={"help": "Evaluate and save results every n epochs", "tyro_name": "eval-every-e"})
    # early_stop_e: int = field(default=5, metadata={"help": "Early stopping epochs", "tyro_name": "early-stop-e"})
    feat_bias: float = field(default=5, metadata={"help": "GRU layer bias", "tyro_name": "feat-bias"})
    
    which_epoch: str = field(default="all", metadata={"help": "Name of this trial", "tyro_name": "which-epoch"})
    
    # Res Predictor related options
    vq_name: str = field(default="rvq_nq6_dc512_nc512_noshare_qdp0.2", metadata={"help": "Experiment name", "tyro_name": "vq-name"})
    # n_res: int = field(default=2, metadata={"help": "Number of res blocks", "tyro_name": "n-res"})
    # do_vq_res: bool = field(default=False, metadata={"help": "Perform VQ residual", "tyro_name": "do-vq-res"})
    seed: int = field(default=3407, metadata={"help": "Random seed", "tyro_name": "seed"})
    
    # wandb related options
    no_wandb: bool = field(default=False, metadata={"help": "Disable wandb logging", "tyro_name": "no-wandb"})
    project_name: str = field(default="rvqvae", metadata={"help": "wandb project name", "tyro_name": "project-name"})
    exp_name: Optional[str] = field(default=None, metadata={"help": "wandb experiment name", "tyro_name": "exp-name"})
    log_dir: str = field(default="./logs", metadata={"help": "Log directory", "tyro_name": "log-dir"})
    normalize: bool = field(default=False, metadata={"help": "Normalize data", "tyro_name": "normalize"})
    data_root: str = field(default="./data", metadata={"help": "Data directory", "tyro_name": "data-root"})
    ckpt_path: Optional[str] = field(default=None, metadata={"help": "Checkpoint path for evaluation", "tyro_name": "ckpt-path"})
    
    # Internal fields
    is_train: bool = field(default=False, metadata={"help": "Training mode", "tyro_name": "is-train"})

    # Optimization
    # part: int = field(default=0, metadata={"help": "Part", "tyro_name": "part"})

def arg_parse(is_train=False):
    """
    Parse command line arguments.
    
    Args:
        is_train (bool): Whether to train the model
        
    Returns:
        VQVAEOptions: Parsed options
    """
    opt = tyro.cli(VQVAEOptions)
    opt.is_train = is_train
    torch.cuda.set_device(opt.gpu_id)
    
    args = vars(opt)
    
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    
    if is_train:
        # Save to disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.mode, opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
    
    return opt