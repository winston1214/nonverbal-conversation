import smplx
import numpy as np
import torch

def merge_parameters_to_smplx(param, smplx_model, exp_coeffs=50, mode='osx'):
    # Load SMPLX model
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # smplx_model = smplx.create(model_path, model_type='smplx', gender='neutral', use_face_contour=True, ext='pkl', use_pca=False, num_expression_coeffs=exp_coeffs).to(device)
    
    # Initialize SMPLX parameters with zeros
    smplx_params = {
        'betas': torch.zeros([10], device = device),  # shape coefficients
        'global_orient': torch.zeros([3], device = device),  # global orientation
        'body_pose': torch.zeros([21 * 3], device = device),  # body pose
        'left_hand_pose': torch.zeros([15 * 3], device = device),  # left hand pose
        'right_hand_pose': torch.zeros([15 * 3], device = device),  # right hand pose
        'jaw_pose': torch.zeros([3], device = device),  # jaw pose
        'leye_pose': torch.zeros([3], device = device),  # left eye pose
        'reye_pose': torch.zeros([3], device = device),  # right eye pose
        'expression': torch.zeros([50], device = device)  # facial expression
    }

    # Merge SMPL parameters
    seq_len = param.shape[0]
    if mode == 'full':
        emoca_param = param[:,:53]
        osx_param = param[:,53:]
    elif mode == 'osx':
        osx_param = param
    
    # if smpl_params:
    smplx_params['betas'] = torch.zeros([seq_len, 10], device = device)
    smplx_params['global_orient'] = torch.zeros([seq_len, 3], device = device)
    # pad for lower body
    # padded_arr = np.pad(osx_param[:, :27], ((0, 0), (36,0)), mode='constant')
    # body_pose = torch.tensor(padded_arr)
    body_pose = torch.nn.functional.pad(osx_param[:,:27], (36, 0), mode='constant', value=0)
    smplx_params['body_pose'] = body_pose

    # Merge MANO parameters
    # if mano_params:
    smplx_params['left_hand_pose'] = torch.tensor(osx_param[:, 27:72]).to(device)
    smplx_params['right_hand_pose'] = torch.tensor(osx_param[:, 72:]).to(device)

    # Merge FLAME parameters
    # if flame_params:
    smplx_params['leye_pose'] = torch.zeros([seq_len, 3], device = device)
    smplx_params['reye_pose'] = torch.zeros([seq_len, 3], device = device)
    if mode == 'osx' :
        smplx_params['jaw_pose'] = torch.zeros([seq_len, 3], device = device)
        smplx_params['expression'] = torch.zeros([seq_len, 50], device = device)
    elif mode == 'full' :
        smplx_params['jaw_pose'] = torch.tensor(emoca_param[:,50:53], device = device)
        smplx_params['expression'] = torch.tensor(emoca_param[:,:50], device = device)
    
    # print(f'betas : {smplx_params["betas"].shape}')
    # Create SMPLX output
    # smplx_params['betas'] = smplx_params['betas']
    # print(f'betas : {smplx_params["betas"].shape}')

    smplx_output = smplx_model(
        betas=smplx_params['betas'],
        global_orient=smplx_params['global_orient'],
        body_pose=smplx_params['body_pose'],
        left_hand_pose=smplx_params['left_hand_pose'],
        right_hand_pose=smplx_params['right_hand_pose'],
        jaw_pose=smplx_params['jaw_pose'],
        leye_pose=smplx_params['leye_pose'],
        reye_pose=smplx_params['reye_pose'],
        expression=smplx_params['expression']
    )

    return smplx_output, smplx_model