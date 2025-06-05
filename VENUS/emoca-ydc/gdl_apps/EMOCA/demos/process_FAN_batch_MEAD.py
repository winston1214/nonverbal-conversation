import torch
# from gdl.utils.FaceDetector import FAN
from PIL import Image
from torchvision import transforms
from skimage.io import imread
import glob
import time
import json
import os
import numpy as np
from torch.utils.data.dataloader import DataLoader
from tqdm import auto
# import face_alignment
from face_alignment_ import face_alignment
from bbox_utils import *
import argparse
from landmark_utils import *
from gdl_apps.EMOCA.utils.io import save_obj, save_images, save_codes, test
from gdl_apps.EMOCA.utils.load import load_model
import gdl
from pathlib import Path
from tqdm import auto



def load_dataset(args,image_folder) :
    
    # Load dataset and define output file name
    emotion_dict = {"neutral":1, "calm":2, "happy":3, "sad":4, "angry":5, "fear":6, "disgusted":7, "surprised":8, "contempt":9}
    image_tensors = {}
    
    folders = glob.glob(f'/mnt/storage/MEAD/frames/**') # /mnt/storage/Celebv_text/video_unzip/ ...
    for folder in folders : # Actor_01, Actor_02 ...
        print(f'Start {folder}')
        actor_name = os.path.basename(folder)
        emotions = glob.glob(f'{folder}/**')
        for emotion in emotions :
            emotion_name = emotion_dict[os.path.basename(emotion)]
            intensities = glob.glob(f'{emotion}/**')
            for intensity in intensities :
                intensity_name = int(os.path.basename(intensity).split('_')[-1])
                input_sessions = glob.glob(f'{intensity}/**') # ~/001
                os.makedirs(f'{args.output_folder}/{actor_name}', exist_ok=True)
                for input_session in input_sessions :
                    video_name = os.path.basename(input_session)
                    # set output file name
                    output_name = f'{actor_name}_{emotion_name}_{intensity_name}_{video_name}'
                    start_data = time.time()
                    input_frames = sorted(glob.glob(f'{input_session}/*.png'))
                    images_tensor = []
                    for image_path in input_frames :
                        image = Image.open(image_path)
                        transform = transforms.Compose([transforms.PILToTensor()])
                        image = transform(image)
                        images_tensor+=[image]
                    images_tensor = torch.stack(images_tensor, dim=0)
                    image_tensors[output_name] = images_tensor
                    
                    end_data = time.time()
                    print(f'saving {output_name}:{images_tensor.shape} takes {end_data-start_data}')
                    
    return image_tensors
    
    
def face_detect(args, emoca, face_model, dataset, device, batch_size =16) :
    outputs = dataset.keys()
    for output_file_name in outputs :
        images_ = dataset[output_file_name]
        flame_params_final=[]
        flame_params = []
        image_loader = DataLoader(images_, batch_size=batch_size, shuffle=False)
        start_total = time.time()
        for i, batch in enumerate(auto.tqdm(image_loader)) :
            batch = batch.to(device, dtype=torch.float32)
            # print(f'batch shape : {batch.shape}')
            # print(f'batch is on device {batch.device}')
            start = time.time()
            # detect faces
            start_detect = time.time()
            with torch.no_grad() :
                bboxlist = face_model.face_detector.detect_from_batch(batch)
            end_detect = time.time()
            print(f'Face detect takes {end_detect-start_detect}')

            # crop detected faces
            start_crop = time.time()
            fan_bboxes = []
            centers = []
            scales = []
            for sfd_bbox in bboxlist :
                sfd_bbox = sfd_bbox[0]
                bbox, center, scale = sfd_bbox_to_fan_bbox(sfd_bbox[:4])
                
                fan_bboxes.append(bbox)
                centers.append(center)
                scales.append(scale)
            fan_bboxes = torch.stack(fan_bboxes, dim=0)
            centers = torch.stack(centers, dim=0)
            scales = torch.stack(scales, dim=0).unsqueeze(-1)
            cropped_batch = crop_batch(batch, fan_bboxes) # (BS,3,256,256)
            end_crop = time.time()
            print(f'Crop takes {end_crop-start_crop}')
            # print('cropped batch shape :', cropped_batch.shape) 
            # print(f'the cropped batch is on device {cropped_batch.device}')
            
            # detect 2d landmarks
            # cropped_batch = cropped_batch
            # cropped_batch = cropped_batch.to(device=device, dtype=torch.float32)
            start_heat = time.time()
            cropped_batch.div_(255.0)
            with torch.no_grad() :
                heatmap_batch = face_model.face_alignment_net(cropped_batch) # (BS,68,64,64)
            # print(f'landmark batch shape : {heatmap_batch.shape}')
            # print(f'landmark batch is on device {heatmap_batch.device}')
            end_heat = time.time()
            print(f'Heatmap takes {end_heat-start_heat}')
            start_land = time.time()
            pts, pts_img, scores = get_preds_fromhm(heatmap_batch, device, centers, scales)
            end_land = time.time()
            print(f'Landmark takes {end_land-start_land}')
            pts_img = pts_img.view(batch.shape[0], 68, 2)
            # scores = scores.squeeze(0)
            landmarks = pts_img
            # landmarks_scores=[scores]
            # define boundingboxes
            kpt = landmarks.squeeze()
            left = torch.min(kpt[:,:,0],dim=1).values
            right = torch.max(kpt[:,:,0],dim=1).values
            top = torch.min(kpt[:,:,1],dim=1).values
            bottom = torch.max(kpt[:,:,1],dim=1).values

            # bounding_boxes_batch = [left, right, top, bottom]
            # crop image
            batch = batch / 255.
            old_size, center = bbox2point(left, right, top, bottom)
            size = (old_size*1.25).int()
            # size = old_size
            dst_image = bbpoint_warp_resize(batch, center, size, 224, landmarks=kpt)
            
            end = time.time()
            b_size = dst_image.shape[0]
            print(f'It takes {end-start}')
            if len(dst_image.shape) == 3:
                dst_image = dst_image.view(1,3,224,224)
            batch = {"image": dst_image}
            vals = emoca.encode(batch, training=False)
            vals, visdict = decode(emoca, vals, training=False)
            
            for i in range(b_size):
                # name = f"{(j*batch_size + i):05d}"
                # name =  batch["image_name"][i]

                # sample_output_folder = Path(outfolder) /name
                # sample_output_folder.mkdir(parents=True, exist_ok=True)

                # if args.save_mesh:
                #     save_obj(emoca, str(sample_output_folder / "mesh_coarse.obj"), vals, i)
                # if args.save_images:
                #     save_images(outfolder, name, visdict, i)
                if args.save_codes:
                    # flame_param = np.concatenate([vals["expcode"][i].detach().cpu().numpy(), vals["posecode"][i].detach().cpu().numpy(), vals["shapecode"][i].detach().cpu().numpy(), vals["detailcode"][i].detach().cpu().numpy()])
                    flame_param = np.concatenate([vals["expcode"][i].detach().cpu().numpy(), vals["posecode"][i][3:].detach().cpu().numpy(), vals["shapecode"][i].detach().cpu().numpy()])
                    # print(f'expression : {len(vals["expcode"][i].detach().cpu().numpy())}')
                    # print(f'pose : {len(vals["posecode"][i].detach().cpu().numpy())}')
                    # print(f'shape : {len(vals["posecode"][i].detach().cpu().numpy())}')
                    # print(f'details : {len(vals["detailcode"][i].detach().cpu().numpy())}')
                    # 1/0
                    # save_codes(Path(outfolder), name, vals, i)
                    flame_params.append(flame_param)
                print(f'flame_params length : {len(flame_params)}')
            flame_params_final.extend(flame_params)
        flame_params = np.array(flame_params)
        # video_name = os.path.basename(input_video).split('.')[0]
        output_folder = f'{args.output_folder}/{output_file_name.split("_")[0]}'
        np.save(f'{output_folder}/{output_file_name}.npy', flame_params) # /mnt/storage/Celebv_text/flame_param/sp_0003/uid.npy
        # shutil.rmtree(f'{args.output_folder}/{model_name}')
        print(f'{output_folder}/{output_file_name}.npy saved')
        end_total = time.time()
        print(f'Total takes {end_total-start_total}')
        
        return cropped_batch

def decode(emoca, values, training=False):
    with torch.no_grad():
        values = emoca.decode(values, training=training)
        # losses = deca.compute_loss(values, training=False)
        # batch_size = values["expcode"].shape[0]
        uv_detail_normals = None
        if 'uv_detail_normals' in values.keys():
            uv_detail_normals = values['uv_detail_normals']
        visualizations, grid_image = emoca._visualization_checkpoint(
            values['verts'],
            values['trans_verts'],
            values['ops'],
            uv_detail_normals,
            values, 
            0,
            "",
            "",
            save=False
        )

    return values, visualizations

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', type=str, default=str(Path(gdl.__file__).parents[1] / "/assets/data/EMOCA_test_example_data/videos/82-25-854x480_affwild2.mp4"), 
        help="Filename of the video for reconstruction.")
    parser.add_argument('--output_folder', type=str, default="/mnt/storage/MEAD/flame_param", help="Output folder to save the results to.")
    parser.add_argument('--model_name', type=str, default='EMOCA_v2_lr_mse_20', help='Name of the model to use. Currently EMOCA or DECA are available.')
    parser.add_argument('--path_to_models', type=str, default=str(Path(gdl.__file__).parents[1] / "assets/EMOCA/models"))
    parser.add_argument('--mode', type=str, default="detail", choices=["detail", "coarse"], help="Which model to use for the reconstruction.")
    parser.add_argument('--save_images', type=str2bool, default=True, help="If true, output images will be saved")
    parser.add_argument('--save_codes', type=str2bool, default=False, help="If true, output FLAME values for shape, expression, jaw pose will be saved")
    parser.add_argument('--save_mesh', type=str2bool, default=False, help="If true, output meshes will be saved")
    # add a string argument with several options for image type
    parser.add_argument('--image_type', type=str, default='geometry_detail', 
        choices=["geometry_detail", "geometry_coarse", "out_im_detail", "out_im_coarse"], 
        help="Which image to use for the reconstruction video.")
    parser.add_argument('--processed_subfolder', type=str, default=None, 
        help="If you want to resume previously interrupted computation over a video, make sure you specify" \
            "the subfolder where the got unpacked. It will be in format 'processed_%Y_%b_%d_%H-%M-%S'")
    parser.add_argument('--cat_dim', type=int, default=0, 
        help="The result video will be concatenated vertically if 0 and horizontally if 1")
    parser.add_argument('--include_rec', type=str2bool, default=True, 
        help="The reconstruction (non-transparent) will be in the video if True")
    parser.add_argument('--include_transparent', type=str2bool, default=True, 
        help="Apart from the reconstruction video, also a video with the transparent mesh will be added")
    parser.add_argument('--include_original', type=str2bool, default=True, 
        help="Apart from the reconstruction video, also a video with the transparent mesh will be added")
    parser.add_argument('--black_background', type=str2bool, default=False, help="If true, the background of the reconstruction video will be black")
    parser.add_argument('--use_mask', type=str2bool, default=True, help="If true, the background of the reconstruction video will be black")
    parser.add_argument('--logger', type=str, default="", choices=["", "wandb"], help="Specify how to log the results if at all.")
    parser.add_argument('--batch_size', type=str, default=16)
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device {device} is used')
    ## args
    path_to_models = args.path_to_models
    input_video = args.input_video
    model_name = args.model_name
    output_folder = args.output_folder + "/" + model_name
    image_type = args.image_type
    black_background = args.black_background
    include_original = args.include_original
    include_rec = args.include_rec
    cat_dim = args.cat_dim
    use_mask = args.use_mask
    include_transparent = bool(args.include_transparent)
    processed_subfolder = args.processed_subfolder

    mode = args.mode
    ##
    start_model = time.time()
    face_alignment_model = face_alignment.FaceAlignment_(face_alignment.LandmarksType.TWO_D, device=str(device), 
                                                flip_input=False, face_detector='sfd',
                                                face_detector_kwargs={"filter_threshold":0.9})
    
    emoca, conf = load_model(path_to_models, model_name, mode)
    emoca.cuda()
    emoca.eval()
    end_model = time.time()
    print(f'Model loading takes {end_model-start_model}')
    
    
    dataset = load_dataset(args,args.input_video)
    face_detected = face_detect(args, emoca, face_alignment_model, dataset, device, args.batch_size)
    print('DONE')


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--image_folder', type=str, default='/mnt/storage/MEAD/frames/M005/angry/level_1/001')
    # parser.add_argument('--batch_size', type=str, default=16)
    # args = parser.parse_args()
    
    main()


    
    
