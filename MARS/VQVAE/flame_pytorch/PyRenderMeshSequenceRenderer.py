"""
Author: Radek Danecek
Copyright (c) 2023, Radek Danecek
All rights reserved.

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at emote@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
"""
import os 
# if 'DISPLAY' not in os.environ or os.environ['DISPLAY'] == '':
#     os.environ['PYOPENGL_PLATFORM'] = 'egl'
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
# os.environ['PYOPENGL_PLATFORM'] =  'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import numpy as np 
import trimesh 
import pyrender
import cv2
from flame_pytorch import FLAME, get_config
import torch
radian = np.pi / 180.0
class PyRenderMeshSequenceRenderer(object): 

    def __init__(self, 
            template_file,
            height=800., 
            width=800.,
            bg_color=None, 
            t_center=None, 
            rot=np.zeros(3), 
            tex_img=None, 
            z_offset=0,
            ) -> None:
        self.width = width 
        self.height = height

        self.camera_params = {
                        'c': np.array([width / 2, height /2 ]),
                        'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                        'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])
                        }

        self.frustum = {'near': 0.01, 'far': 3.0, 'height': height, 'width': width}

        self.primitive_material_gray = pyrender.material.MetallicRoughnessMaterial(
                alphaMode='BLEND',
                baseColorFactor=[0.3, 0.3, 0.3, 1.0],
                metallicFactor=0.8, 
                roughnessFactor=0.8 
            )

        self.primitive_material_red = pyrender.material.MetallicRoughnessMaterial(
                alphaMode='BLEND',
                baseColorFactor=[0.3, 0.0, 0.0, 1.0],
                metallicFactor=0.8, 
                roughnessFactor=0.8 
            )

        bg_color = bg_color or "black"

        if bg_color == "black":
            self.bg_color = [0, 0, 0]
        elif bg_color == "white":
            self.bg_color = [255, 255, 255]

        if not isinstance(self.bg_color, list):
            raise ValueError("bg_color must be a list of 3 integers in range [0, 255]")

        self.scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=self.bg_color)

        self.template_file = template_file
        self.template = trimesh.load_mesh(template_file)

        self.camera = pyrender.IntrinsicsCamera(
                                    fx=self.camera_params['f'][0],
                                    fy=self.camera_params['f'][1],
                                    cx=self.camera_params['c'][0],
                                    cy=self.camera_params['c'][1],
                                    znear=self.frustum['near'],
                                    zfar=self.frustum['far'])
        

        camera_pose = np.eye(4)
        camera_pose[:3,3] = np.array([0, 0, 1.0-z_offset])
        self.scene.add(self.camera, pose=[[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 1],
                                [0, 0, 0, 1]])
        angle = np.pi / 6.0
        pos = camera_pose[:3,3]
        intensity = 2.0
        light_color = np.array([1., 1., 1.])
        light = pyrender.DirectionalLight(color=light_color, intensity=intensity)

        light_pose = np.eye(4)
        light_pose[:3,3] = pos
        self.scene.add(light, pose=light_pose.copy())
        
        light_pose[:3,3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
        self.scene.add(light, pose=light_pose.copy())

        light_pose[:3,3] =  cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
        self.scene.add(light, pose=light_pose.copy())

        light_pose[:3,3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
        self.scene.add(light, pose=light_pose.copy())

        light_pose[:3,3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
        self.scene.add(light, pose=light_pose.copy())


        self.flags = pyrender.RenderFlags.SKIP_CULL_FACES

        self.rot = rot 
        self.t_center = t_center


    def render(self, verts, rot=None, t_center=None, valid=True):
        """
        verts: (N, 3)

        Returns: rendered image ndarray type uint8
        """
        if self.rot is None: 
            self.rot = np.zeros(3)

        rot = rot or self.rot
        
        if self.t_center is None:
            self.t_center = np.mean(verts, axis=0)

        t_center = t_center or self.t_center

        verts = cv2.Rodrigues(rot)[0].dot((verts-t_center).T).T+t_center
        # mesh_copy = trimesh.base.Trimesh(verts, self.template.faces)


        mesh = trimesh.base.Trimesh(verts, self.template.faces)


        self.render_mesh = pyrender.Mesh.from_trimesh(mesh, 
            material=self.primitive_material_gray if valid else self.primitive_material_red,
            smooth=True)
        self.mesh_node = self.scene.add(self.render_mesh, pose=np.eye(4))
        
        
        r = pyrender.OffscreenRenderer(viewport_width=self.frustum['width'], viewport_height=self.frustum['height'])
        color, _ = r.render(self.scene, flags=self.flags)
        # try:
        #     r = pyrender.OffscreenRenderer(viewport_width=self.frustum['width'], viewport_height=self.frustum['height'])
        #     color, _ = r.render(self.scene, flags=self.flags)
        # except:
        #     print('pyrender: Failed rendering frame')
        #     color = np.zeros((int(self.frustum['height']), int(self.frustum['width']), 3), dtype='uint8')
        
        self.scene.remove_node(self.mesh_node)

        return color.astype(np.uint8) 
    
def get_vertices_from_FLAME(flame, params, use_shape):

    exp_params = params[:, :50]
    if params.shape[1] == 156:
        jaw_params = params[:, 53:56] # this is the new version where we have 156 parameters
    elif params.shape[1] == 153:
        jaw_params = params[:, 50:53] # this is the old version where we have 153 parameters
    # for EMOTE inferenced param
    elif params.shape[1] == 103:
        jaw_params = params[:, 100:103]
    # for DEMOTE inferenced param
    elif params.shape[1] == 53:
        jaw_params = params[:, 50:53]
    if use_shape:
        shape_params = params[:, 56:156]
        assert shape_params.shape[1] == 100, f"shape parameter num for flame is {flame.shape_params}, but got {shape_params.shape[1]}"


    
    batch_size = exp_params.shape[0]
    shape_params = torch.zeros(batch_size, 100, dtype=torch.float32).cuda() 
    

    expression_params = exp_params.clone().detach().to(dtype=torch.float32).cuda()
    
    

    jaw_params = jaw_params.clone().detach().to(dtype=torch.float32).cuda()
    
    
    global_rotation = torch.tensor([0.0, 0.0 * radian, 0.0],dtype=torch.float32).view(1,3).repeat(batch_size,1).cuda()
    pose_params = torch.cat([global_rotation,jaw_params],dim=1).cuda()
    
    vertice, landmark = flame(shape_params, expression_params, pose_params)
    
    return vertice

    
def save_video(video_path, video_tensor, fourcc='png', fps=25):
    if isinstance(video_tensor, torch.Tensor):
        # torch to numpy
        video_tensor_np = video_tensor.cpu().numpy()
        video_tensor_np = video_tensor_np.transpose(0,2,3,1)
    else:
        video_tensor_np = video_tensor
    # video_tensor is in( T, C, H, W)
    # convert to (T, H, W, C)
    if video_tensor_np.dtype == np.float32 or video_tensor_np.dtype == np.float64:
        video_tensor_np = video_tensor_np.clip(0,1)
        # convert to uint8
        video_tensor_np = (video_tensor_np * 255)
    video_tensor_np = video_tensor_np.astype(np.uint8)
    # # rgb to bgr
    video_tensor_np = video_tensor_np[...,::-1]

    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*fourcc), fps,
                            (video_tensor_np.shape[1], video_tensor_np.shape[2])
                            )
    for frame in video_tensor_np:
        writer.write(frame)
    writer.release()