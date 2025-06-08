import torch
import os
import numpy as np
from flame_pytorch.flame import FLAME
from flame_pytorch.PyRenderMeshSequenceRenderer import get_vertices_from_FLAME
from flame_pytorch.config import VertexArguments
from flame_pytorch.motion_util import merge_parameters_to_smplx
import torch.nn as nn
from scipy import linalg
import tyro
import smplx
class VertexMetric(nn.Module):
    def __init__(self, args, mode, device="cuda:0"):
        super(VertexMetric, self).__init__()
        self.args = args
        self.flame_bacth_size = 2048
        self.flame = FLAME(args, batch_size=self.flame_bacth_size).to(device).eval()
        self.device = device
        self.mode = mode
        self.smplx_model_path = '../VENUS/OSX/common/utils/human_model_files'
        self.smplx_model = smplx.create(self.smplx_model_path, model_type='smplx', gender='neutral', use_face_contour=True, ext='pkl', use_pca=False, num_expression_coeffs=50).to(self.device)
    
    def _get_vertices_flame(self, params):
        """
        params: (-1, 53)
        """
        with torch.no_grad():
            batch_size = self.flame_bacth_size
            total_samples = params.shape[0]
            
            vertices_list = []
            
            
            for i in range(0, total_samples, batch_size):
                # current batch size (last batch is smaller)
                current_batch = params[i:i+batch_size]
                current_size = current_batch.shape[0]
                
                if current_size < batch_size:
                    # Adjust the last element to match the batch size
                    padding = current_batch[-1].unsqueeze(0).repeat(batch_size - current_size, 1)
                    current_batch = torch.cat([current_batch, padding], dim=0)
                
                # Calculate transformed vertices
                batch_vertices = get_vertices_from_FLAME(self.flame, current_batch, self.args.with_shape)
                
                # Remove padding and save only actual data
                vertices_list.append(batch_vertices[:current_size])
            
            # Combine all results
            return torch.cat(vertices_list, dim=0)
        
    def _get_vertices_smplx(self, params):
        """
        params: (-1, 117)
        """

        with torch.no_grad():
            batch_size = 1024
            total_samples = params.shape[0]
            
            vertices_list = []
            
            for i in range(0, total_samples, batch_size):
                # current batch size (last batch is smaller)
                current_batch = params[i:i+batch_size]
                current_size = current_batch.shape[0]
                
                if current_size < batch_size:
                    # Adjust the last element to match the batch size
                    padding = current_batch[-1].unsqueeze(0).repeat(batch_size - current_size, 1)
                    current_batch = torch.cat([current_batch, padding], dim=0)
                
                # 현재 배치를 (batch_size, 117) 형태로 reshape
                current_batch = current_batch.reshape(-1, 117)
                
                # SMPLX 모델을 사용하여 vertices 계산
                gt_smplx_output, _ = merge_parameters_to_smplx(current_batch, self.smplx_model, mode='osx')

                batch_vertices = gt_smplx_output.vertices.cpu()
                
                # Remove padding and save only actual data
                vertices_list.append(batch_vertices[:current_size])
            
            # Combine all results
            return torch.cat(vertices_list, dim=0)

    def calculate_vertex_metric(self, pred_params, gt_params): # Vertex L2 distance

        if self.mode == 'face':
            pred_vertices = self._get_vertices_flame(pred_params)
            gt_vertices = self._get_vertices_flame(gt_params)
            vmse = torch.mean(torch.norm(pred_vertices - gt_vertices, dim=1)).item()
            vmae = torch.mean(torch.abs(pred_vertices - gt_vertices)).item()
            window_l2 = self.calculate_windowed_l2v_vertex(gt_vertices, pred_vertices, 25)
        elif self.mode == 'body':
            pred_vertices = self._get_vertices_smplx(pred_params)
            gt_vertices = self._get_vertices_smplx(gt_params)
            vmse = torch.mean(torch.norm(pred_vertices - gt_vertices, dim=1)).item()
            vmae = torch.mean(torch.abs(pred_vertices - gt_vertices)).item()
            window_l2 = self.calculate_windowed_l2v_vertex(gt_vertices, pred_vertices, 25)
        return vmse, vmae, window_l2

    def calculate_windowed_l2v_vertex(self, gt_vertices, pred_vertices, window_size):
        """
        Calculates the windowed L2 distance (MSE) based on vertices.

        """
        
        # gt_vertices: [The number of frames, The number of vertex, 3]
        n_frames, n_vertices, _ = gt_vertices.shape
        

        # Transform all vertices to a representative value per frame
        gt_v = gt_vertices.mean(dim=1)  # [number of frames, 3]
        pred_v = pred_vertices.mean(dim=1)  # [number of frames, 3]

        
        # Split the vertices into windows
        actual_window_size = min(window_size, n_frames)
        

        windowed_gt = gt_v.unfold(dimension=0, size=actual_window_size, step=actual_window_size)
        windowed_gt_mean = windowed_gt.mean(dim=1)  # [window num, 3]
        
        windowed_pred = pred_v.unfold(dimension=0, size=actual_window_size, step=actual_window_size)
        windowed_pred_mean = windowed_pred.mean(dim=1)  # [window num, 3]
        # Calculate the MSE for each window
        window_mse = ((windowed_gt_mean - windowed_pred_mean) ** 2).mean().item()
        
        return window_mse


class FeatureMetric(nn.Module):
    def __init__(self, args):
        super(FeatureMetric, self).__init__()
        self.args = args
    def MSE(self, pred_features, gt_features):
        return np.mean((pred_features - gt_features) ** 2)
    
    def calculate_activation_statistics(self, activations):
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        diff = mu1 - mu2
        

        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            covmean = np.real(covmean)

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)
    
    def calculate_FD(self, gt_motion, pred_motion):
        
        gt_mu, gt_cov = self.calculate_activation_statistics(gt_motion)
        pred_mu, pred_cov = self.calculate_activation_statistics(pred_motion)
        fid = self.calculate_frechet_distance(gt_mu, gt_cov, pred_mu, pred_cov)
        return fid
    def calculate_diversity(self, activation, K=1000, repeats=10):
        """
        Calculates the diversity of motion parameters.
        
        Args:
            activation: Array of motion parameters in shape (N, D)
            K: Number of pairs to randomly select (default: 1000)
            repeats: Number of repetitions (default: 10)
            
        Returns:
            float: Diversity score
        """
        assert len(activation.shape) == 2
        num_samples = activation.shape[0]
        
        
        if num_samples < 2:
            return 0.0
        
        max_pairs = num_samples * (num_samples - 1) // 2
        K = min(K, max_pairs) 
        
        diversity_scores = []
        
        for _ in range(repeats):
            pair_distances = []
            
            
            for _ in range(K):
                i, j = np.random.choice(num_samples, 2, replace=False)
                distance = np.linalg.norm(activation[i] - activation[j])
                pair_distances.append(distance)
                
            
            diversity_scores.append(np.mean(pair_distances))
        
        
        return np.mean(diversity_scores)
    
    def calculate_diversities(self, gt_motion, pred_motion, K=1000, repeats=10):
        gt_diversity = self.calculate_diversity(gt_motion, K, repeats)
        pred_diversity = self.calculate_diversity(pred_motion, K, repeats)
        return gt_diversity, pred_diversity

    def calculate_windowed_l2v(self, gt_motion, pred_motion, window_size):
        gt_v = gt_motion.mean(axis=1)
        pred_v = pred_motion.mean(axis=1)

        windowed_gt_v = torch.from_numpy(gt_v).view(-1, 1).unfold(dimension=0, size=min(window_size, gt_v.shape[0]), step=window_size)
        windowed_gt_v = windowed_gt_v.mean(dim=-1)
        windowed_pred_v = torch.from_numpy(pred_v).view(-1, 1).unfold(dimension=0, size=min(window_size, pred_v.shape[0]), step=window_size)
        windowed_pred_v = windowed_pred_v.mean(dim=-1).numpy()
        windowed_mse_v = ((windowed_gt_v - windowed_pred_v) ** 2).mean()

        return windowed_mse_v

    def motion_variance(self, motion):
        return np.var(motion, axis=0).mean()

    def calculate_all_metrics(self, gt_motion, pred_motion, window_size=25):
        metrics = {}
        
        # MSE
        metrics['mse'] = self.MSE(gt_motion, pred_motion)
        
        # FID
        metrics['fid'] = self.calculate_FD(gt_motion, pred_motion)
        
        # Diversity
        K = 1000
        repeats = 10
        gt_diversity, pred_diversity = self.calculate_diversities(gt_motion, pred_motion, K, repeats)
        metrics['diversity_gt'] = gt_diversity
        metrics['diversity'] = pred_diversity
        
        # window L2V 계산
        metrics['windowed_l2v'] = self.calculate_windowed_l2v(gt_motion, pred_motion, window_size)
        
        # Variance 계산
        gt_var = self.motion_variance(gt_motion)
        pred_var = self.motion_variance(pred_motion)
        metrics['var_gt'] = gt_var
        metrics['var'] = pred_var
        
        return metrics




