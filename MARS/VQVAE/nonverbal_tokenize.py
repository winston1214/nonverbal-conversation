import torch
import numpy as np
from options.vq_option import arg_parse
from utils.fixseed import fixseed
from models.vq.model import RVQVAE
import pickle
from os.path import join as pjoin
from venus_dataset_huggingface import VENUSDataset, custom_collate_fn
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import os

def extract_frame_id(s: str) -> str:
    if s == '':
        return s
    rest = s[12:]
    parts = rest.split('_')
    
    if len(parts) >= 4:
        frame_id = parts[-2]
    else:
        frame_id = parts[-1]
    
    return int(frame_id)

def extract_id_without_frame(s: str) -> str:
    # 1) video_id: First 11 characters
    video_id = s[:11]
    
    remainder = s[11:]
    # Remove the last frame_id: Remove the last '_' and take the front part
    #    rsplit('_', 1) → ['_<segment>_<utterance>', '<frame_id>']
    base_suffix = remainder.rsplit('_', 1)[0]
    return video_id + base_suffix


def load_model(model_path):  
    if opt.mode == "face":
        dim_pose = 53  # Expression(50) + Jaw(3)
    elif opt.mode == "body":
        dim_pose = 117  # Upper body + right hand + left hand
    elif opt.mode == "full":
        dim_pose = 170 

    model = RVQVAE(opt,
                 dim_pose,
                 opt.nb_code,
                 opt.code_dim,
                 opt.code_dim,
                 opt.down_t,
                 opt.stride_t,
                 opt.width,
                 opt.depth,
                 opt.dilation_growth_rate,
                 opt.vq_act,
                 opt.vq_norm)
    model.load_state_dict(torch.load(model_path)['vq_model'])
    model = model.to(opt.device)
    model.eval()
    return model

def encode_with_mask(model, inputs, masks):
    # Convert mask to encoder mask
    x_in = model.preprocess(inputs)
    x_encoder = model.encoder(x_in)
    encoder_mask = model.create_encoder_mask(masks, x_encoder.shape)
    
    # Pass mask when quantize
    code_idx, all_codes = model.quantizer.quantize(x_encoder, mask=encoder_mask, return_latent=True)
    return code_idx, all_codes

def convert_to_special_token(token_idx, token_type="FACE"):
    return f"<{token_type}_{token_idx}>"

def get_frame_numbers_from_dataset(test_dataset, test_dataset_key):
    """
    Extract all frame numbers corresponding to a specific utterance from the test dataset
    
    Args:
        test_dataset: Test dataset
        test_dataset_key: Utterance key (video_id_segment_id_utterance_id)
    
    Returns:
        frame_numbers: List of frame numbers for the specific utterance
    """

    # Find all data indices corresponding to the utterance key
    indices = [i for i, x in enumerate(test_dataset) if '_'.join(x[1][0].split('_')[:-1]) == test_dataset_key]
    
    # Extract the frame numbers from the names of the data entries
    frame_numbers = []
    for idx in indices:
        # Name format: {video_id}_{segment_id}_{utterance_id}_{frame_num}
        name = test_dataset[idx][1][0]
        frame_num = int(name.split('_')[-1])
        frame_numbers.append(frame_num)
    
    frame_numbers.sort()
    return frame_numbers


def interleave_tokens_with_multimodal(face_tokens, body_tokens, words_info, frame_numbers, start_fps, end_fps, downsampling_factor=8):
    """
    Interleave Face and Body motion tokens with words based on frame number information (no length limit)
    
    Args:
        face_tokens: List of face motion tokens
        body_tokens: List of body motion tokens
        words_info: List containing start/end time information for each word
        frame_numbers: List of actual frame numbers for the utterance
        start_fps: Start frame index of the utterance
        end_fps: End frame index of the utterance
        downsampling_factor: Downsampling factor
    
    Returns:
        interleaved_sequence: Token sequence interleaved with text and motion
    """
    # Ready to event by time order (word start, word end)
    word_events = []
    
    for i, word_info in enumerate(words_info):
        # Time -> Frame
        word_start_fps = int(word_info['start_time'] * 25)  # 25fps
        word_end_fps = int(word_info['end_time'] * 25)
        
        # Check and adjust if it is within the utterance range
        if word_start_fps < start_fps:
            word_start_fps = start_fps
        if word_end_fps > end_fps:
            word_end_fps = end_fps
            
        if word_start_fps <= word_end_fps:  # Only add valid range
            word_events.append({
                'fps': word_start_fps,
                'type': 'word_start',
                'word': word_info['word'],
                'index': i
            })
            word_events.append({
                'fps': word_end_fps,
                'type': 'word_end',
                'index': i
            })
    
    # Create actual frame mapping for motion tokens
    motion_events = []
    
    # Calculate the relationship between frame numbers and speech time range
    if len(frame_numbers) > 0:
        # Check how well the frame numbers match the speech time range
        # Find the closest frame numbers
        closest_start_idx = min(range(len(frame_numbers)), key=lambda i: abs(frame_numbers[i] - start_fps))
        closest_end_idx = min(range(len(frame_numbers)), key=lambda i: abs(frame_numbers[i] - end_fps))
        
        # Determine the effective frame range to use
        effective_frame_numbers = frame_numbers[closest_start_idx:closest_end_idx+1]
        
        # Map both Face and Body tokens for each frame
        min_tokens = min(len(face_tokens), len(body_tokens))
        for i in range(min_tokens):
            if i >= len(effective_frame_numbers) // downsampling_factor:
                # If there are no more frames to map
                break
                
            # Calculate the frame index considering downsampling
            frame_idx = i * downsampling_factor
            if frame_idx < len(effective_frame_numbers):
                real_frame_num = effective_frame_numbers[frame_idx]
                
                # Check if it is within the speech range
                if start_fps <= real_frame_num <= end_fps:
                    motion_events.append({
                        'fps': real_frame_num,
                        'type': 'multimodal_motion',
                        'face_token': face_tokens[i],
                        'body_token': body_tokens[i],
                        'token_idx': i
                    })
    
    # Merge all events and sort in chronological order
    all_events = word_events + motion_events
    all_events.sort(key=lambda x: x['fps'])
    
    # Initialize result sequence and current state
    # interleaved_sequence = ["<SOT>"]  # Start of sequence token
    interleaved_sequence = []
    last_token_idx = -1
    processed_words = set()
    current_word = None
    
    # Process each event
    for event in all_events:
        # Process based on event type
        if event['type'] == 'word_start':
            word = event['word']
            word_idx = event['index']
            
            # Check for duplicate words
            if word_idx not in processed_words:
                interleaved_sequence.append(word)
                processed_words.add(word_idx)
                current_word = word_idx
                
        elif event['type'] == 'word_end':
            if event['index'] == current_word:
                current_word = None
                
        elif event['type'] == 'multimodal_motion':
            token_idx = event['token_idx']
            
            # Check if token has already been processed (avoid duplicates)
            if token_idx > last_token_idx:
                # Add Face token
                face_special_token = convert_to_special_token(int(event['face_token']), "FACE")
                interleaved_sequence.append(face_special_token)
                
                # Add Body token
                body_special_token = convert_to_special_token(int(event['body_token']), "BODY")
                interleaved_sequence.append(body_special_token)
                
                last_token_idx = token_idx
    
    # Check and add missing motion tokens - remove max_length limit
    if last_token_idx + 1 < min(len(face_tokens), len(body_tokens)):
        remaining_tokens = []
        for i in range(last_token_idx + 1, min(len(face_tokens), len(body_tokens))):
            remaining_tokens.append(convert_to_special_token(int(face_tokens[i]), "FACE"))
            remaining_tokens.append(convert_to_special_token(int(body_tokens[i]), "BODY"))
        
        if remaining_tokens:
            interleaved_sequence.extend(remaining_tokens)
    
    return interleaved_sequence


if __name__ == "__main__":
    opt = arg_parse()
    fixseed(opt.seed)

    MODE = 'train'
    PART = opt.part

    opt.save_root = pjoin(opt.checkpoints_dir, opt.mode, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    print(f"Using Device: {opt.device}")
    face_model = load_model(pjoin(opt.model_dir, 'best.tar'))
    print("face_model loaded!")

    if MODE == 'train':
        with open(f'medium_{opt.mode}_train_dataset.pkl', 'rb') as f:
            face_dataset = pickle.load(f)
        print("face_dataset loaded!")
    else:
        with open(f'medium_{opt.mode}_test_dataset.pkl', 'rb') as f:
            face_dataset = pickle.load(f)
        print("face_dataset loaded!")

    opt.mode = "body"
    opt.name = "final_l1_recon_dim16"
    opt.save_root = pjoin(opt.checkpoints_dir, opt.mode, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.code_dim = 16
    body_model = load_model(pjoin(opt.model_dir, 'best.tar'))
    print("body_model loaded!")

    if MODE == 'train':
        with open(f'medium_{opt.mode}_train_dataset.pkl', 'rb') as f:
            body_dataset = pickle.load(f)
        print("body_dataset loaded!")
    else:
        with open(f'medium_{opt.mode}_test_dataset.pkl', 'rb') as f:
            body_dataset = pickle.load(f)
        print("body_dataset loaded!")
    print("face dataset: ", len(face_dataset))
    print("body _dataset: ", len(body_dataset))

    if MODE == 'train':
        venus_test = load_dataset("winston1214/VENUS-5K", split="train")
    else:
        venus_test = load_dataset("winston1214/VENUS-5K", split="test")
  


    


    token_sequence_list = []
    for i in tqdm(range(len(venus_test))):
        video_id = venus_test[i]['video_id']
        segment_id = venus_test[i]['segment_id']

        if os.path.exists(f'{MODE}_token/{video_id}_{segment_id}.pkl'):
            continue

        data = venus_test[i]['conversation']
        utterances = [
            {'utterance_id': uid, 'speaker': spk, 'text': txt, "start_time": start, "end_time": end, 'words': words, }
            for uid, spk, txt, start, end, words in zip(data['utterance_id'], data['speaker'], data['text'], data['start_time'], data['end_time'], data['words'])
        ]
        
        
        name_list = [(test_idx, extract_id_without_frame(x[1][0])) for test_idx, x in enumerate(face_dataset)]
        for utt in utterances:

            # Filtering Harmful dataset
            if utt['utterance_id'] in venus_test[i]['harmful_utterance_id']:
                continue

            # Extract face features for the utterance
            test_dataset_key = f'{video_id}_{segment_id}_{utt["utterance_id"]}'
            try: # If not exists utterance, skip (i.e. without features)
                test_dataset_idx = list(filter(lambda x: x[1] == test_dataset_key, name_list))[0][0]
            except:
                continue
            face_features_for_utt = face_dataset[test_dataset_idx]
            body_features_for_utt = body_dataset[test_dataset_idx]

            face_features = custom_collate_fn([face_features_for_utt])
            body_features = custom_collate_fn([body_features_for_utt])
            with torch.no_grad(): # Model inference
                inputs = face_features['inputs'].to(opt.device)
                masks = face_features['masks'].to(opt.device)
                names = face_features['names'][0]
                frame_numbers = list(map(extract_frame_id, names))
                frame_numbers = [x for x in frame_numbers if x]
                code_idx, all_codes = encode_with_mask(face_model, inputs, masks)
                
                # Check the encoder mask to determine token validity
                x_in = face_model.preprocess(inputs)
                x_encoder = face_model.encoder(x_in)
                encoder_mask = face_model.create_encoder_mask(masks, x_encoder.shape)
                encoder_mask = encoder_mask.squeeze().cpu().numpy()
                
                # Extract the processed codebook indices and mask information
                code_idx = code_idx.squeeze().squeeze().cpu().numpy()
                
                # Check for valid token indices (use tokens only where the mask is True)
                valid_token_indices = np.where(encoder_mask)[0]
                
                # Select only valid tokens by slicing the array
                valid_face_code_idx = code_idx[valid_token_indices]
                
                # Also store the original indices of each token (for frame mapping)
                valid_face_token_original_indices = valid_token_indices
            with torch.no_grad():
                inputs = body_features['inputs'].to(opt.device)
                masks = body_features['masks'].to(opt.device)
                names = body_features['names'][0]
                frame_numbers = list(map(extract_frame_id, names))
                frame_numbers = [x for x in frame_numbers if x]

                code_idx, all_codes = encode_with_mask(body_model, inputs, masks)
                # Judge validity of tokens by checking the encoder mask
                x_in = body_model.preprocess(inputs)
                x_encoder = body_model.encoder(x_in)
                encoder_mask = body_model.create_encoder_mask(masks, x_encoder.shape)
                encoder_mask = encoder_mask.squeeze().cpu().numpy()
                
                # Extract the processed codebook indices and mask information
                code_idx = code_idx.squeeze().squeeze().cpu().numpy()
                
                # Check for valid token indices (use tokens only where the mask is True)
                valid_token_indices = np.where(encoder_mask)[0]
                
                # Select only valid tokens by slicing the array
                valid_body_code_idx = code_idx[valid_token_indices]
                
                # Save valid token indices for body tokens
                valid_body_token_original_indices = valid_token_indices


            start_fps = int(utt['start_time'] * 25)
            end_fps = int(utt['end_time'] * 25)
            words_info = [{'word': w, 'start_time': s, 'end_time': e} for w, s, e in zip(utt['words']['word'], utt['words']['start_time'], utt['words']['end_time'])]
            try:
                interleaved_tokens = interleave_tokens_with_multimodal(
                    valid_face_code_idx, 
                    valid_body_code_idx, 
                    words_info, 
                    frame_numbers,
                    start_fps, 
                    end_fps, 
                    downsampling_factor=opt.stride_t ** opt.down_t, 
                )
            except:
                print(f'{video_id}_{segment_id}_{utt["utterance_id"]} failed')
                continue

            # Save token sequence
            token_sequence = {
                'video_id': video_id + '_' + str(segment_id).zfill(3),
                'utterance_id': utt['utterance_id'],
                'token_sequence': interleaved_tokens
            }

            token_sequence_list.append(token_sequence)
            save_video_id = video_id + '_' + str(segment_id).zfill(3) + '_' + str(utt['utterance_id'])
            with open(f'{MODE}_token/{save_video_id}.pkl', 'wb') as f:
                pickle.dump(token_sequence, f)

