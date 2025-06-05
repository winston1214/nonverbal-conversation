import glob
import os
import torch
import sys
from torch.utils.data.dataloader import DataLoader
import argparse
from scipy.io import wavfile
import python_speech_features
from tqdm import tqdm
from SyncNetInstance import *
import torchvision.transforms as t 
import torchvision
import json
import noisereduce as nr

sys.path.append('../')
from face_alignment_ import face_alignment_1 as face_alignment


# (12-25) JB - following EMOTE 
training_ids = ['M003', 'M005', 'M007', 'M009', 'M011', 'M012', 'M013', 'M019', 
                'M022', 'M023', 'M024', 'M025', 'M026', 'M027', 'M028', 'M029', 
                'M030', 'M031', 'W009', 'W011', 'W014', 'W015', 'W016', 'W018', 
                'W019', 'W021', 'W023', 'W024', 'W025', 'W026', 'W028', 'W029'
                ] # 32 ids
val_ids = ['M032', 'M033', 'M034', 'M035', 'W033', 'W035', 'W036']  # 7 ids
# val_ids = ['M032']

test_ids = ['M037', 'M039', 'M040', 'M041', 'M042', 'W037', 'W038', 'W040'] # 8 ids

EMOTION_DICT = {'neutral': 1, 'happy': 2, 'sad': 3, 'surprised': 4, 'fear': 5, 'disgusted': 6, 'angry': 7, 'contempt': 8, 'calm' : 9}

modify_DICT = {1:1, 3:2, 4:3, 5:7, 6:5, 7:6, 8:4, 9:8}

def main(syncnet_model, face_model, image_paths, audio_path, args, transform, device) :
    # for faster face detection
    transforms_1 = transform
    # for matching input of syncnet
    transforms_2 = t.Compose([t.Resize((224,224))])

    # process cropping image for batch
    image_paths = DataLoader(image_paths, batch_size = args.batch_size, shuffle=False)
    input_images = []
    for i, img_paths in enumerate(tqdm(image_paths,desc='Face detect')) :
        tensor_image = []
        cropped_images = []
        for img_path in img_paths :
            try :
                image = torchvision.io.read_image(img_path)
                image = image[:3,:,:]
            except :
                with open(f'{args.save_path}/image_load_error.txt', 'a') as log_file:
                    log_file.write(f"{os.path.dirname(img_path)} already exists! skipping...\n")
                return 'load_image_error'
            tensor_image+=[image]
        tensor_image = torch.stack(tensor_image, dim=0).to(device, dtype=torch.float32)
        tensor_image = transforms_1(tensor_image)
        # face detect
        sfd_bboxlist = face_model.face_detector.detect_from_batch(tensor_image)

        for i, bbox in enumerate(sfd_bboxlist) :
            # try : 
            bbox = bbox[0]
            cropped_image = tensor_image[i][:, int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            # to match input size of syncnet
            cropped_image = transforms_2(cropped_image)
            cropped_images+=[cropped_image]
            # except:
            #     with open(f'{args.save_path}/face_detector_error.txt', 'a') as log_file:
            #         log_file.write(f"{os.path.dirname(img_path)} face not detected\n")
            #     print(f'face detector error!! {os.path.dirname(img_paths[0])}')
            #     return 'face_detector_error'
        cropped_images = torch.stack(cropped_images, dim=0)
        # cropped_images = transforms_2(cropped_images)
        input_images+=[cropped_images.cpu()]
    input_images = torch.cat(input_images, dim=0)
    # print(f'input_images : {input_images.shape}')
    #################### fps : 25
    # if input_images.shape[0] <= 50 :
    #     pass
    # else :
    #     # crop 0.8s , 0.2s
    #     input_images = input_images[20:len(input_images)-5]
    ###################### fps : 30
    if input_images.shape[0] <= 60 :
        pass
    else :
        # crop 0.8s , 0.2s
        input_images = input_images[24:len(input_images)-6]
    

    tensor_image = input_images.permute(1,0,2,3).unsqueeze(0)
    # print(f'input image shape : {tensor_image.shape}')

    # image_loader = DataLoader(img_paths, batch_size=batch_size, shuffle=False)

    # wav mfcc
    # try :
    sample_rate, audio = wavfile.read(audio_path)
    # erase background noise
    audio = nr.reduce_noise(y=audio, sr=sample_rate)
    if len(audio) <= 32000 :
        pass
    else :
        # crop 0.8s, 0.2s
        audio = audio[12800:len(audio)-3200]
    # except :
    #     print(f"There's no audio file : {audio_path}")
    #     with open(f'{args.save_path}/no_audio_file.txt', 'a') as log_file:
    #         log_file.write(f"{audio_path} doensn't exist! skipping...\n")
    #     return 'audio not existing'
    mfcc = zip(*python_speech_features.mfcc(audio,sample_rate))
    mfcc = torch.stack([torch.tensor(i) for i in mfcc])
    cc = mfcc.unsqueeze(0).unsqueeze(0).to(torch.float32)

    # check audio and video input lengths
    if (float(len(audio))/16000) != (float(input_images.shape[0])/25) :
        print("WARNING: Audio (%.4fs) and video (%.4fs) lengths are different."%(float(len(audio))/16000,float(input_images.shape[0])/25))

    min_length = min(input_images.shape[0],math.floor(len(audio)/640))

    with torch.no_grad() :
        offset, conf, fconfm = syncnet_model.inference(face_model, args, tensor_image, cc, device, min_length)
    return offset, conf, fconfm

if __name__=='__main__' :
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_frame_folder', type=str, help='input_folder of frames')
    # parser.add_argument('--input_audio_folder', type=str, help='input_folder of audio')
    parser.add_argument('--audio_path', type=str, help='input processed audio, frame paired path')
    parser.add_argument('--frame_path', type=str, help='input processed audio, frame paired path')
    parser.add_argument('--save_path', type=str, help='save_folder for syncneet')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--dataset', type=str, help='dataset type')
    parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='')
    parser.add_argument('--save_name', type=str, help='model name')
    parser.add_argument('--vshift', type=int, default='15', help='')
    parser.add_argument('--fps', type=int, default='25', help='')
    args = parser.parse_args()


    print(f'MODEL : {args.dataset}')
    # load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    face_alignment_model = face_alignment.FaceAlignment_(face_alignment.LandmarksType.TWO_D, device=str(device), 
                                        flip_input=False, face_detector='sfd',
                                        face_detector_kwargs={"filter_threshold":0.9})
    s = SyncNetInstance()
    s.loadParameters(args.initial_model)
    print("Model %s loaded."%args.initial_model)

    # define transform from dataset
    if args.dataset == 'VOXCELEB': # 224X224
        transform = None
    elif args.dataset == 'MEAD': # 1240 X 1080
        transform = t.Compose([t.CenterCrop(900),t.Resize((300,300))])
    elif args.dataset == 'CELEBV-TEXT': # 598 X 598
        transform = t.Compose([t.Resize((224,224))])
    elif args.dataset == 'CELEBV-HQ': # 
        transform = t.Compose([t.Resize((224,224))])
    elif args.dataset == 'CREMA-D': # 480 X 360
        print('Start CREMA-D')
        # transform = t.Compose([t.CenterCrop(360),t.Resize((300,300))])
        transform = t.Compose([t.Resize((300,300))])
        val_ids = ['1001', '1002', '1003', '1004', '1005', '1006', '1007', '1008', '1009', '1010',
                '1011', '1012', '1013', '1014', '1015', '1016', '1017', '1018', '1019', '1020']
    else:
        transform = t.Compose([t.Resize((300,300))])
        if args.dataset == 'RAVDESS' :
            val_ids = ['Actor_01', 'Actor_02', 'Actor_03', 'Actor_04', 'Actor_05', 'Actor_06',
            'Actor_07', 'Actor_08', 'Actor_09', 'Actor_10', 'Actor_11', 'Actor_12']

    # audio_frame_pair = load_data(args.input_frame_folder, args.input_audio_folder)

    # syncnet_score_emotion = {}
    # syncnet_score_actor = {}
    # syncnet_score_actor_emotion = {}

    total_conf = 0
    total_min_distance = 0
    total_frames = 0

    # for val_id in val_ids :
    #     syncnet_score_actor[val_id] = {'conf' :0, 'min_distance' :0, 'videos' :0}
    #     # syncnet_score_actor.append({val_id : {'conf' :0, 'min_distance' :0, 'videos' :0}})
    #     syncnet_score_actor_emotion[val_id] = {}
    #     for i in range(1,9) :
    #         syncnet_score_actor_emotion[val_id][i] = {'conf' : 0, 'min_distance' : 0, 'videos' : 0}
    #         # syncnet_score_actor_emotion.append({val_id : {i : {'conf' : 0, 'min_distance' : 0, 'videos' : 0}}})

    # print(syncnet_score_actor_emotion)
    

    # for i in range(1,9) :
    #     # syncnet_score_emotion[i] = {}
    #     syncnet_score_emotion[i] = {'conf' : 0, 'min_distance' : 0, 'videos' : 0}
    #     # syncnet_score_emotion.append({i : {'conf' : 0, 'min_distance' : 0, 'videos' : 0}})
    for val_id in val_ids :
        file_ids = glob.glob(f'{args.frame_path}/{val_id}/**')
        for file_id in file_ids :
            file_name = os.path.basename(file_id)
            audio_path = f'{args.audio_path}/{val_id}/{file_name}.wav'
            frame_paths = sorted(glob.glob(f'{file_id}/*.png'), key=lambda x: int(os.path.basename(x).split(".")[0]))
            print(f'frame paths : {frame_paths}')
            if len(frame_paths)/25 >=20 :
                print(f'{file_name} longer than 20s : {len(frame_paths)/25}')
                continue
            if frame_paths == [] :
                continue

            results = main(s, face_alignment_model, frame_paths, audio_path, args, transform, device)
            if len(results) == 3 :
                offset, conf, min_distance = results
                total_conf += conf
                total_min_distance += min_distance
                total_frames+=1
            else :
                continue
        
    print(f'Total LSE-C : {total_conf/total_frames}')
    print(f'Total LSE-D : {total_min_distance/total_frames}')
    print(f'Model : {args.save_name}')
            



    # emotions = glob.glob(f'{args.audio_path}/**')
    # for emotion in emotions :
    #     emotion_name = os.path.basename(emotion)
    #     audio_paths = glob.glob(f'{args.audio_path}/{emotion_name}/*.wav')
    #     for audio_path in audio_paths :
    #         file_name = os.path.basename(audio_path).split('.')[0]
    #         frame_paths = sorted(glob.glob(f'{args.frame_path}/{emotion_name}/{file_name}_idM003__0/*.png'))
    #         # frame_paths = sorted(glob.glob(f'{args.frame_path}/{emotion_name}/{file_name}/*.png'))
    #         if len(frame_paths)/25 >=20 :
    #             print(f'{file_name} longer than 20s : {len(frame_paths)/25}')
    #             continue
    #         if frame_paths == [] :
    #             continue

    #         results = main(s, face_alignment_model, frame_paths, audio_path, args, transform, device)
    #         if len(results) == 3 :
    #             offset, conf, min_distance = results
    #             # per actor
    #             # syncnet_score_actor[val_id]['conf']+=conf
    #             # syncnet_score_actor[val_id]['min_distance']+=min_distance
    #             # syncnet_score_actor[val_id]['videos']+=1
    #             # # per emotion
    #             # syncnet_score_emotion[emotion]['conf']+=conf
    #             # syncnet_score_emotion[emotion]['min_distance']+=min_distance
    #             # syncnet_score_emotion[emotion]['videos']+=1
    #             # # per actor per emotion
    #             # syncnet_score_actor_emotion[val_id][emotion]['conf']+=conf
    #             # syncnet_score_actor_emotion[val_id][emotion]['min_distance']+=min_distance
    #             # syncnet_score_actor_emotion[val_id][emotion]['videos']+=1

    #             total_conf += conf
    #             total_min_distance += min_distance
    #             total_frames+=1
    #         else :
    #             continue
        
    # print(f'Total LSE-C : {total_conf/total_frames}')
    # print(f'Total LSE-D : {total_min_distance/total_frames}')
    # print(f'Model : {args.save_name}')


    # for val_id in val_ids :
    #     syncnet_score_actor[val_id]['LSE-C'] = syncnet_score_actor[val_id]['conf']/syncnet_score_actor[val_id]['videos']
    #     syncnet_score_actor[val_id]['LSE-D'] = syncnet_score_actor[val_id]['min_distance']/syncnet_score_actor[val_id]['videos']

    #     for i in range(1,9) :
    #         syncnet_score_actor_emotion[val_id][i]['LSE-C'] = syncnet_score_actor_emotion[val_id][i]['conf']/syncnet_score_actor_emotion[val_id][i]['videos']
    #         syncnet_score_actor_emotion[val_id][i]['LSE-D'] = syncnet_score_actor_emotion[val_id][i]['min_distance']/syncnet_score_actor_emotion[val_id][i]['videos']


    # for i in range(1,9) :
    #     syncnet_score_emotion[i]['LSE-C'] = syncnet_score_emotion[i]['conf']/syncnet_score_emotion[i]['videos']
    #     syncnet_score_emotion[i]['LSE-D'] = syncnet_score_emotion[i]['min_distance']/syncnet_score_emotion[i]['videos']
    
    # with open(f'{args.save_path}/{args.dataset}_actor_syncnet_score.json', 'w') as f :
    #     json.dump(syncnet_score_actor, f)
    
    # with open(f'{args.save_path}/{args.dataset}_actor_emotion_syncnet_score.json', 'w') as f :
    #     json.dump(syncnet_score_actor_emotion, f)

    # with open(f'{args.save_path}/{args.dataset}_emotion_syncnet_score.json', 'w') as f :
    #     json.dump(syncnet_score_emotion, f)