import torch
import wandb
from os.path import join as pjoin
import torch.nn.functional as F

import torch.optim as optim

import time
import numpy as np
from collections import OrderedDict, defaultdict



def def_value():
    return 0.0





class RVQTokenizerTrainer:
    def __init__(self, args, vq_model):
        self.opt = args
        self.vq_model = vq_model
        self.device = args.device
        self.jaw_lambda = args.jaw_lambda
        self.expr_lambda = args.expr_lambda
        self.vel_expr_lambda = args.vel_expr_lambda
        self.vel_jaw_lambda = args.vel_jaw_lambda
        self.body_lambda = args.body_lambda
        self.hands_lambda = args.hands_lambda
        self.vel_body_lambda = args.vel_body_lambda
        self.vel_hands_lambda = args.vel_hands_lambda

        if args.is_train:
            # wandb initialize
            
            # self.logger = SummaryWriter(args.log_dir)
            if args.recons_loss == 'l1':
                self.recon_criterion = torch.nn.L1Loss(reduction='none')
                self.vel_criterion = torch.nn.L1Loss(reduction='none')
            elif args.recons_loss == 'l1_smooth':
                self.recon_criterion = torch.nn.SmoothL1Loss(reduction='none')
                self.vel_criterion = torch.nn.SmoothL1Loss(reduction='none')
            elif args.recons_loss == 'l2':
                self.recon_criterion = torch.nn.MSELoss(reduction='none')
                self.vel_criterion = torch.nn.MSELoss(reduction='none')

        # self.critic = CriticWrapper(self.opt.dataset_name, self.opt.device)

    def forward(self, inputs, masks):
        # motions = batch_data.detach().to(self.device).float()
        
        pred_motion, loss_commit, perplexity = self.vq_model(inputs, masks)
        
        self.motions = inputs
        self.pred_motion = pred_motion
        self.masks = masks

        if self.opt.mode == 'face':
            feature_masks = masks.unsqueeze(-1)
            exp_loss_raw = self.recon_criterion(pred_motion[:,:,:50], self.motions[:,:,:50])
            jaw_loss_raw =  self.recon_criterion(pred_motion[:,:,50:], self.motions[:,:,50:])
            # Apply mask
            exp_loss = (exp_loss_raw * feature_masks).sum() / (feature_masks.sum() * exp_loss_raw.shape[-1] + 1e-8)
            jaw_loss = (jaw_loss_raw * feature_masks).sum() / (feature_masks.sum() * jaw_loss_raw.shape[-1] + 1e-8)
            loss_rec = self.expr_lambda * exp_loss + self.jaw_lambda * jaw_loss
        elif self.opt.mode == 'body': # TMP
            feature_masks = masks.unsqueeze(-1)
            body_raw = self.recon_criterion(pred_motion[:,:,:27], self.motions[:,:,:27])
            hands_raw = self.recon_criterion(pred_motion[:,:,27:], self.motions[:,:,27:])
                
            
            body_loss = (body_raw * feature_masks).sum() / (feature_masks.sum() * body_raw.shape[-1] + 1e-8)
            hands_loss = (hands_raw * feature_masks).sum() / (feature_masks.sum() * hands_raw.shape[-1] + 1e-8)
            loss_rec = self.body_lambda * body_loss + self.hands_lambda * hands_loss


        # Velocity loss
        vel_gt = self.motions[:, 1:, :] - self.motions[:, :-1, :]
        vel_pred = pred_motion[:, 1:, :] - pred_motion[:, :-1, :]
        # Float 마스크를 Boolean으로 변환 후 논리 연산 수행
        # bool_masks = masks > 0.5  # Float를 Boolean으로 변환
        vel_mask = masks[:, 1:] * masks[:, :-1]  # 논리 AND 연산
        
        if self.opt.mode == 'face':
        # vel_diff_expr = vel_pred[:, :, :50] - vel_gt[:, :, :50]
            vel_loss_expr_elemwise = self.recon_criterion(vel_pred[:, :, :50], vel_gt[:, :, :50])
            vel_mask = vel_mask.unsqueeze(-1)
            loss_vel_expr = (vel_loss_expr_elemwise * vel_mask).sum() / (vel_mask.sum() * vel_loss_expr_elemwise.shape[-1] + 1e-8)
            
            vel_loss_jaw_elemwise = self.recon_criterion(vel_pred[:, :, 50:], vel_gt[:, :, 50:])
            
            loss_vel_jaw = (vel_loss_jaw_elemwise * vel_mask).sum() / (vel_mask.sum() * vel_loss_jaw_elemwise.shape[-1] + 1e-8)
            loss_vel = self.vel_expr_lambda * loss_vel_expr + self.vel_jaw_lambda * loss_vel_jaw
        
        elif self.opt.mode == 'body': 
            vel_loss_body_elemwise = self.recon_criterion(vel_pred[:, :, :27], vel_gt[:, :, :27])
            vel_mask = vel_mask.unsqueeze(-1)
            loss_vel_body = (vel_loss_body_elemwise * vel_mask).sum() / (vel_mask.sum() * vel_loss_body_elemwise.shape[-1] + 1e-8)
            loss_vel_body = self.vel_body_lambda * loss_vel_body

            vel_loss_hands_elemwise = self.recon_criterion(vel_pred[:, :, 27:], vel_gt[:, :, 27:])
            loss_vel_hands = (vel_loss_hands_elemwise * vel_mask).sum() / (vel_mask.sum() * vel_loss_hands_elemwise.shape[-1] + 1e-8)
            loss_vel_hands = self.vel_hands_lambda * loss_vel_hands

            loss_vel = loss_vel_body + loss_vel_hands
        
        

        loss_vel = loss_vel * self.opt.loss_vel
        loss_commit = loss_commit * self.opt.commit
        loss_rec = loss_rec * self.opt.recon_lambda

        if self.opt.reg_lambda > 0:
            reg_loss = self.vq_model.quantizer.get_reg_loss()
            loss_reg = reg_loss * self.opt.reg_lambda
            loss = loss_vel + loss_commit + loss_rec + loss_reg
        else:
            loss_reg = torch.tensor(0.0, device=self.device)
            loss = loss_vel + loss_commit + loss_rec

        return loss, loss_rec, loss_vel, loss_commit, loss_reg, perplexity
    


    # @staticmethod
    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_vq_model.param_groups:
            param_group["lr"] = current_lr

        return current_lr

    def save(self, file_name, ep, total_it):
        state = {
            "vq_model": self.vq_model.state_dict(),
            "opt_vq_model": self.opt_vq_model.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.vq_model.load_state_dict(checkpoint['vq_model'])
        self.opt_vq_model.load_state_dict(checkpoint['opt_vq_model'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader):
        self.vq_model.to(self.device)

        self.opt_vq_model = optim.AdamW(self.vq_model.parameters(), lr=self.opt.lr, betas=(0.9, 0.99), weight_decay=self.opt.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt_vq_model, milestones=self.opt.milestones, gamma=self.opt.gamma)

        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))

        current_lr = self.opt.lr
        logs = defaultdict(def_value, OrderedDict())

        min_val_loss = float('inf')

        while epoch < self.opt.max_epoch:
            self.vq_model.train()
            for i, batch_data in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    current_lr = self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)
                inputs = batch_data['inputs'].to(self.device)
                masks = batch_data['masks'].to(self.device)
                loss, loss_rec, loss_vel, loss_commit, loss_reg, perplexity = self.forward(inputs, masks)
                self.opt_vq_model.zero_grad()
                loss.backward()
                self.opt_vq_model.step()

                if it >= self.opt.warm_up_iter:
                    self.scheduler.step()
                

                logs['loss'] += loss.item()
                logs['loss_rec'] += loss_rec.item()
                # Note it not necessarily velocity, too lazy to change the name now
                logs['loss_vel'] += loss_vel.item()
                logs['loss_commit'] += loss_commit.item()
                logs['loss_reg'] += loss_reg.item()
                logs['perplexity'] += perplexity.item()
                logs['lr'] += self.opt_vq_model.param_groups[0]['lr']

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    # wandb로 로그 기록
                    train_logs = {}
                    for tag, value in logs.items():
                        mean_value = value / self.opt.log_every
                        train_logs[f'Train/{tag}'] = mean_value
                        mean_loss[tag] = mean_value
                    wandb.log(train_logs, step=it)
                    logs = defaultdict(def_value, OrderedDict())
                    # print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                # if it % self.opt.save_latest == 0:
                #     self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            # if epoch % self.opt.save_every_e == 0:
            #     self.save(pjoin(self.opt.model_dir, 'E%04d.tar' % (epoch)), epoch, total_it=it)

            print('Validation time:')
            self.vq_model.eval()
            val_loss_rec = []
            val_loss_vel = []
            val_loss_commit = []
            val_loss = []
            val_perpexity = []
            val_loss_reg = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    inputs = batch_data['inputs'].to(self.device)
                    masks = batch_data['masks'].to(self.device)
                    loss, loss_rec, loss_vel, loss_commit, loss_reg, perplexity = self.forward(inputs, masks)
                    # val_loss_rec += self.l1_criterion(self.recon_motions, self.motions).item()
                    # val_loss_emb += self.embedding_loss.item()
                    val_loss.append(loss.item())
                    val_loss_rec.append(loss_rec.item())
                    val_loss_vel.append(loss_vel.item())
                    val_loss_commit.append(loss_commit.item())
                    val_perpexity.append(perplexity.item())
                    val_loss_reg.append(loss_reg.item())

            # Logging validation loss
            val_logs = {
                'Val/loss': sum(val_loss) / len(val_loss),
                'Val/loss_rec': sum(val_loss_rec) / len(val_loss_rec),
                'Val/loss_vel': sum(val_loss_vel) / len(val_loss_vel),
                'Val/loss_commit': sum(val_loss_commit) / len(val_loss),
                'Val/loss_reg': sum(val_loss_reg) / len(val_loss_reg),
                'Val/loss_perplexity': sum(val_perpexity) / len(val_loss_rec),
                'epoch': epoch
            }
            wandb.log(val_logs, step=it)

            current_recon_val_loss = sum(val_loss_rec) / len(val_loss_rec)

            print('Validation Loss: %.5f Reconstruction: %.5f, Velocity: %.5f, Commit: %.5f, Reg: %.5f' %
                  (sum(val_loss)/len(val_loss), sum(val_loss_rec)/len(val_loss), 
                   sum(val_loss_vel)/len(val_loss), sum(val_loss_commit)/len(val_loss), sum(val_loss_reg)/len(val_loss_reg)))

            if current_recon_val_loss < min_val_loss:
                min_val_loss = current_recon_val_loss
                
                state = {
                    "vq_model": self.vq_model.state_dict(),
                    "opt_vq_model": self.opt_vq_model.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    'epoch': epoch,
                    'it': it,
                    'val_loss': min_val_loss,
                    'val_loss_rec': sum(val_loss_rec) / len(val_loss_rec),
                    'val_loss_vel': sum(val_loss_vel) / len(val_loss_vel),
                    'val_loss_commit': sum(val_loss_commit) / len(val_loss),
                    'val_loss_reg': sum(val_loss_reg) / len(val_loss_reg),
                    'perplexity': sum(val_perpexity) / len(val_loss_rec),
                }
                torch.save(state, pjoin(self.opt.model_dir, 'best.tar'))
                print(f'New best model saved at epoch {epoch} with validation loss {min_val_loss:.5f}')
                
                
                wandb.log({
                    'Best/val_loss': min_val_loss,
                    'Best/epoch': epoch,
                }, step=it)

