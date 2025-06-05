import os
import torch
from glob import glob
test_dir = '/share0/MIR_LAB/nc_video/shard_0/segment/aRi5o-81vpk_000/12/0/0'
# print(os.listdir(test_dir))
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params = []

inputs = {}
for fpath in glob(f"{test_dir}/*.npy"):
    param = np.load(fpath)
    fname = fpath.split('/')[-1].split('.')[0]
    inputs[fname] = torch.FloatTensor(param).unsqueeze(0).to(device)
    
    params.append(np.expand_dims(param, -1))
    print(f"{fpath} : {param.shape}")
    
''' OSX Parameters -> Mesh Vertices

'''
import os, sys
OSX_PATH = f'/home/dlwlgp/vq_video/OSX'
sys.path.append(OSX_PATH)
from main.OSX import get_model
model = get_model('test')
model = model.to(device)

inputs['cam_trans'] = torch.FloatTensor(np.array([
    [1,0,0],[0,1,0],[0,0,1]
])).unsqueeze(0).to(device)

# 정답 root_pose, body_pose (나머지는 모두 0으로 처리), lhand_pose, rhand_pose, jaw_pose, shape, expr, cam_trans #
# 필요한 모든 parameter을 입출력으로 VQ-VAE를 학습 시킨다면?
# with torch.no_grad():

joint_proj, joint_cam, mesh_cam = model(
    inputs = inputs, targets=None, meta_info=None, mode='get_coord_only'
)

print(f"Joint Proj shape : {joint_proj.shape}")
print(f"Joint Cam shape : {joint_cam.shape}")
print(f"Mesh Cam shape : {mesh_cam.shape}")