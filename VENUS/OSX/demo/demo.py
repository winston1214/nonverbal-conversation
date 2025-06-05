import os
import sys
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
from config import cfg
import copy
import cv2
import time
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids', default='0')
    parser.add_argument('--img_path', type=str, default='input.png')
    parser.add_argument('--output_folder', type=str, default='output')
    parser.add_argument('--encoder_setting', type=str, default='osx_l', choices=['osx_b', 'osx_l'])
    parser.add_argument('--decoder_setting', type=str, default='normal', choices=['normal', 'wo_face_decoder', 'wo_decoder']) ## wo_decoder은 encoder의 feature들을 사용해서 hand, face의 parameter을 구하는 방법 ##
    parser.add_argument('--pretrained_model_path', type=str, default='../pretrained_models/osx_l.pth.tar')
    parser.add_argument('--visualize', type=int)
    parser.add_argument('--postfix', type=str, help='postfix of the paths to save')
    
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    return args

args = parse_args()
cfg.set_args(args.gpu_ids)
cudnn.benchmark = True

# load model
cfg.set_additional_args(encoder_setting=args.encoder_setting, decoder_setting=args.decoder_setting, pretrained_model_path=args.pretrained_model_path)
from common.base import Demoer
demoer = Demoer()
demoer._make_model()
from common.utils.preprocessing import load_img, process_bbox, generate_patch_image
from common.utils.vis import render_mesh, save_obj
from common.utils.human_models import smpl_x
model_path = args.pretrained_model_path
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))

demoer.model.eval()
# img_paths = ['/home/jisoo6687/OSX/assets/-7rkMNhO2qA_001_541.jpg', '/home/jisoo6687/OSX/assets/-400XiJWSjY_000_0.jpg',
#                 '/home/jisoo6687/OSX/assets/1Hd3kzMcVEs_001_468.jpg', '/home/jisoo6687/OSX/assets/6CmxdI61HG8_009_678.jpg',
#                 '/home/jisoo6687/OSX/assets/mTzn9uVAEJU_006_1022.jpg', '/home/jisoo6687/OSX/assets/r3mqZlb5ilM_003_470.jpg',
#                 '/home/jisoo6687/OSX/assets/VIHz5QuRPpE_016_534.jpg', '/home/jisoo6687/OSX/assets/zunfnLD7uY4_012_230.jpg']
img_paths = ['/home/dlwlgp/vq_video/OSX/demo/input_2.jpg']
for i, img_path in enumerate(img_paths) :
    # prepare input image
    transform = transforms.ToTensor()
    # original_img = load_img(args.img_path)
    original_img = load_img(img_path)
    original_img_height, original_img_width = original_img.shape[:2] # (H,W,3) # 
    os.makedirs(args.output_folder, exist_ok=True)

    # detect human bbox with yolov5s
    # detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    detector = torch.hub.load('yolov5', 'custom', 'yolov5s.pt', source='local')
    
    with torch.no_grad():
        results = detector(original_img)
    person_results = results.xyxy[0][results.xyxy[0][:, 5] == 0]
    class_ids, confidences, boxes = [], [], []
    for detection in person_results:
        x1, y1, x2, y2, confidence, class_id = detection.tolist()
        class_ids.append(class_id)
        confidences.append(confidence)
        boxes.append([x1, y1, x2 - x1, y2 - y1])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    vis_img = original_img.copy()
    for num, indice in enumerate(indices):
        bbox = boxes[indice]  # x,y,h,w
        org_bbox = copy.deepcopy(bbox)
        bbox = process_bbox(bbox, original_img_width, original_img_height) ## bbox를 찾고 ##
        '''DEBUG (1) bbox change'''
        print(f"Original BBOX : {org_bbox} -> Processed BBOX : {bbox}")
        
        # 찾은 bbox 영역에 대해서 crop후 affine warping으로 원하는 OSX 모델에의 input_img_shape로 resizing을 해 준다. #
        # scale = 1.0, rotation = 0.0, do_flip = False #
        img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape)
        
        '''DEBUG (2) : save the cropped image'''
        print(f"IMG2BBOX : {img2bb_trans}")
        print(f"BBOX2IMG : {bb2img_trans}")

        plt.imshow(img.astype(np.float32)/255)
        dest_folder = os.path.join(args.output_folder, "bbox-crop");os.makedirs(dest_folder, exist_ok=True)
        plt.savefig(f"{dest_folder}/{num}_bbox.png")
        
        img = transform(img.astype(np.float32))/255
        img = img.cuda()[None,:,:,:]
        inputs = {'img': img}
        targets = {}
        meta_info = {}

        # mesh recovery
        with torch.no_grad():
            out = demoer.model(inputs, targets, meta_info, 'test')

        mesh = out['smplx_mesh_cam'].detach().cpu().numpy()
        mesh = mesh[0]
        print(f'mesh shape : {mesh.shape}')
        print(f'root pose : {out["smplx_root_pose"]}')
        print(f'body pose : {out["smplx_body_pose"]}')

        # save mesh
        # save_obj(mesh, smpl_x.face, os.path.join(args.output_folder, f'person_{i}_{num}.obj'))

        # render mesh
        # focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
        focal = [cfg.focal[0] / cfg.input_img_shape[1] * bbox[2], cfg.focal[1] / cfg.input_img_shape[0] * bbox[3]]
        
        # focal = [cfg.focal[0], cfg.focal[1]]
        # princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]
        princpt = [cfg.princpt[0] / cfg.input_img_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_img_shape[0] * bbox[3] + bbox[1]]
        
        # princpt = [cfg.princpt[0]+bbox[0], cfg.princpt[1]+bbox[1]]
        
        print(f'Focal distance : {focal}')
        print(f'Center coord : {princpt}')
        
        start = time.time()
        vis_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt})
        end = time.time()
        print(f'Render takes {end-start}')

    # save rendered image
    if args.visualize == 1:
        cv2.imwrite(os.path.join(args.output_folder, f"{args.postfix}_render_{img_path.split('/')[-1]}"), vis_img[:, :, ::-1])
