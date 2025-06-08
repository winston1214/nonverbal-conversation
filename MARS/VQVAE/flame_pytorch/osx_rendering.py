import os
os.environ['PYOPENGL_PLATFORM'] =  'egl'
import numpy as np 
import trimesh 
import pyrender
from flame_pytorch.flame import FLAME
import torch
from PIL import Image
from flame_pytorch.PyRenderMeshSequenceRenderer2 import get_vertices_from_FLAME
# for motion
from flame_pytorch.motion_util import merge_parameters_to_smplx
import warnings

from flame_pytorch.config import VertexArguments
import tyro

import smplx
import sys
import cv2
from tqdm import tqdm
from scipy.signal import savgol_filter

import numpy as np
import trimesh
import os
import imageio.v2 as imageio
import argparse
import imageio

warnings.filterwarnings("ignore")

def rotate_camera_pose(camera_pose, angle_degrees, axis='z'):
    """
    Rotate the camera pose by a given angle around a given axis.

    Parameters:
    camera_pose (np.ndarray): Original camera pose (shape: [4, 4])
    angle_degrees (float): Rotation angle in degrees
    axis (str): Axis to rotate around ('x', 'y', or 'z')

    Returns:
    np.ndarray: Rotated camera pose
    """
    angle_radians = np.radians(angle_degrees)
    
    if axis == 'x':
        R = np.array([
            [1, 0, 0],
            [0, np.cos(angle_radians), -np.sin(angle_radians)],
            [0, np.sin(angle_radians), np.cos(angle_radians)]
        ])
    elif axis == 'y':
        R = np.array([
            [np.cos(angle_radians), 0, np.sin(angle_radians)],
            [0, 1, 0],
            [-np.sin(angle_radians), 0, np.cos(angle_radians)]
        ])
    elif axis == 'z':
        R = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians), 0],
            [np.sin(angle_radians), np.cos(angle_radians), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Invalid axis value. Choose from 'x', 'y', or 'z'.")
    
    # Create a 4x4 rotation matrix
    R_homogeneous = np.eye(4)
    R_homogeneous[:3, :3] = R
    
    # Rotate the camera pose
    new_camera_pose = R_homogeneous @ camera_pose
    
    return new_camera_pose

def save_render(args, flame_param):
    radian = np.pi / 180.0
    flamelayer = FLAME(args)
    shape_params = torch.zeros(1, 100).cuda()
    
    
    expr = flame_param[:,:50]
    jaw = flame_param[:,50:]
    
    expression_params = torch.tensor(expr, dtype=torch.float32).cuda()
    jaw_params = torch.tensor(jaw, dtype=torch.float32).cuda()
    
    flamelayer.cuda().eval()
    parameters = torch.cat([expression_params, jaw_params], dim=1)
    vertice, landmark = get_vertices_from_FLAME(flamelayer, parameters, False)
    faces = flamelayer.faces
    
    i = 0
    vertices = vertice[i].detach().cpu().numpy().squeeze()
    joints = landmark[i].detach().cpu().numpy().squeeze()
    vertex_colors = np.ones([vertices.shape[0], 4]) * [0.5, 0.5, 0.5, 0.8]
    tri_mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors,
                            face_colors=[0.9, 0.1, 0.1, 1.0], process=False)
    tri_mesh.visual.vertex_colors = [0.5, 0.5, 0.5, 1.0]
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    camera_pose = np.array([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.4],  # Adjust Z position
                            [0.0, 0.0, 0.0, 1.0]])
    camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0, aspectRatio=1.0)
    camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
    scene = pyrender.Scene(nodes=[camera_node])
    scene.add(mesh)
    sm = trimesh.creation.uv_sphere(radius=0.005)
    sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
    tfs = np.tile(np.eye(4), (len(joints), 1, 1))
    tfs[:, :3, 3] = joints
    joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
    light = pyrender.DirectionalLight(color=[0.8, 0.8, 0.8], intensity=1)
    scene.add(light)
    renderer = pyrender.OffscreenRenderer(viewport_width=args.width, viewport_height=args.height)
    
    color, _ = renderer.render(scene)
    image = Image.fromarray(color)
    # image.save(f'vertices.png')
    return image


def save_render_motion(args, param, model_path, mode='osx') :
    """render flame model with pyrender
    (NOTE) faces can be obtained from FLAME.faces
    Args:
        vertices (torch.tensor): verticies (T, 5023, 3)
        faces (torch.tensor): template FLAME Faces (9976,3)
    
    """
    
    if mode == 'osx' :
        smplx_output, smplx_model = merge_parameters_to_smplx(param, model_path)
    elif mode == 'full' :
        smplx_output, smplx_model = merge_parameters_to_smplx(param, model_path, mode='full')

    vertices = smplx_output.vertices

    stacked_vertices = torch.stack([vertices]).squeeze(0)


    faces = smplx_model.faces
    bg_color = 'white'
    vertices_ = stacked_vertices[0].detach().cpu().numpy()

    tri_mesh = trimesh.Trimesh(vertices_, faces, process=False)
    tri_mesh.fix_normals()

    # material
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        roughnessFactor=0.5,
        baseColorFactor=(0.9, 0.9, 0.9, 1.0)
    )
    mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=material, smooth=True)

    # mesh coloring
    mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=False)

    # create camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi/3.5, aspectRatio=1.0)
    camera_pose = np.array([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 1.1],  # Adjust Z position
                            [0.0, 0.0, 0.0, 1.0]])
    # rotate camera pose
    camera_pose = rotate_camera_pose(camera_pose, -15, axis='x')

    y_offset = -0.05
    camera_pose[1, 3] += y_offset

    z_offset = 0.1
    camera_pose[2, 3] += z_offset

    camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
    if bg_color == 'white' :
        bgcolor = [1.0, 1.0, 1.0, 1.0]
    elif bg_color == 'black' :
        bgcolor = [0.0, 0.0, 0.0, 1.0]
    scene = pyrender.Scene(nodes=[camera_node],
                        ambient_light=[.2,.2,.2],
                        bg_color = bgcolor)
    scene.add(mesh)

    # create light
    directional_light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(directional_light, pose=camera_pose)

    # create renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=args.video_width, viewport_height=args.video_height)

    # render
    color, _ = renderer.render(scene)
    image = Image.fromarray(color)
    # Clean up
    renderer.delete()
    scene.clear()

    return image