import os
import glob
import pickle
import torchvision
import cv2
import torch

# input_video = '/home/jisoo6687/OSX/demo/NF3NV-8zhQM_000.webm'
# os.makedirs('frames', exist_ok=True)
# output_video = f'frames/{os.path.basename(input_video).split(".")[0]}'
# cmd=f'ffmpeg -y -r 25 -i {input_video} -r 25 {output_video}_%03d.png'
# os.system(cmd)

images = glob.glob('frames/*.png')
os.makedirs('cropped_frames', exist_ok=True)
with open('/home/jisoo6687/OSX/demo/active_frame_list.pckl', 'rb') as f:
    anno = pickle.load(f)

for i, image_path in enumerate(images[:100]) :
    image = torchvision.io.read_image(image_path)
    bbox = anno[i]['bbox']
    # w = int(bbox[2]-bbox[0])
    # h = int(bbox[3]-bbox[1])
    # left = int(bbox[0])
    # top = int(bbox[1])
    x,y,w,h = bbox
    # image = image[:, int(y):int(y+h) ,int(x):int(x+w)].float()
    image = image[:, int(y):int(y+h) ,int(x):int(x+w)].permute(1,2,0).numpy()
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    print(image.shape)

    save_path = f'cropped_frames/{os.path.basename(image_path).split(".")[0]}.png'

    cv2.imwrite(save_path,image_bgr)
    # torchvision.utils.save_image(image, save_path)

    
