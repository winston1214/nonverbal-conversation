import torch
import time
from bbox_utils import transform
from skimage.transform import estimate_transform, warp
from torchvision.transforms.functional import affine
import torchvision
def get_preds_fromhm(hm, device, center=None, scale=None) :
    """Obtain (x,y) coordinates given a set of N heatmaps. If the center
    and the scale is provided the function will return the points also in
    the original coordinate frame.

    Arguments:
        hm {torch.tensor} -- the predicted heatmaps, of shape [B, N, W, H]

    Keyword Arguments:
        center {torch.tensor} -- the center of the bounding box (default: {None})
        scale {float} -- face scale (default: {None})
    """
    B, C, H, W = hm.shape
    hm_reshape = hm.view(B, C, H * W)
    idx = torch.argmax(hm_reshape, axis=-1) # (10,68)
    # scores = np.take_along_axis(hm_reshape, np.expand_dims(idx, axis=-1), axis=-1).squeeze(-1) # (10,68)
    scores = torch.take_along_dim(hm_reshape, idx.unsqueeze(-1), dim=-1).squeeze(-1)

    # # Use torch.gather to gather values along the last axis
    # scores = torch.gather(hm_reshape, -1, idx).squeeze(-1)
    # print(scores)
    preds, preds_orig = _get_preds_fromhm(hm, idx, device, center, scale)

    return preds, preds_orig, scores

def _get_preds_fromhm(hm, idx, device, center=None, scale=None):
    """Obtain (x,y) coordinates given a set of N heatmaps and the
    coresponding locations of the maximums. If the center
    and the scale is provided the function will return the points also in
    the original coordinate frame.

    Arguments:
        hm {torch.tensor} -- the predicted heatmaps, of shape [B, N, W, H]

    Keyword Arguments:
        center {torch.tensor} -- the center of the bounding box (default: {None})
        scale {float} -- face scale (default: {None})
    """
    B, C, H, W = hm.shape
    idx += 1
    # preds = idx.repeat(2).view(B, C, 2).to(dtype=torch.float32)
    preds = idx.unsqueeze(-1).repeat(1,1,2).view(B, C, 2).to(dtype=torch.float32) # (10,68,2)
    preds[:, :, 0] = (preds[:, :, 0] - 1) % W + 1
    # preds[:, :, 1] = np.floor((preds[:, :, 1] - 1) / H) + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / H) + 1
                
                
                
    pX, pY = preds[:, :, 0].long() - 1, preds[:, :, 1].long() - 1

    # Check boundaries
    valid_mask = (pX > 0) & (pX < 63) & (pY > 0) & (pY < 63)
    
    pX_ = pX * valid_mask
    pY_ = pY * valid_mask

    # Extract hm_ values using advanced indexing
    hm_ = hm[torch.arange(B)[:, None, None], torch.arange(C)[None, :, None], :]

    # Calculate diff using advanced indexing
    diff = torch.stack([
        hm[torch.arange(B)[:, None], torch.arange(C)[None, :], pY_, pX_ + 1] - hm[torch.arange(B)[:, None], torch.arange(C)[None, :], pY_, pX_ - 1],
        hm[torch.arange(B)[:, None], torch.arange(C)[None, :], pY_ + 1, pX_] - hm[torch.arange(B)[:, None], torch.arange(C)[None, :], pY_ - 1, pX_]
    ], dim=-1)

    # Apply the mask to only update valid values
    preds[valid_mask] += torch.sign(diff[valid_mask]) * 0.25


    preds -= 0.5

    # preds_orig = np.zeros_like(preds)
    preds_orig = torch.zeros_like(preds)

    if center is not None and scale is not None:
        preds_orig = transform_tensor(preds, center, scale, H, device, True)
        # for i in range(B):
        #     center_ = center[i]
        #     scale_ = scale[i]
        #     for j in range(C):
        #         preds_orig[i, j] = transform(
        #             preds[i, j], center_, scale_, H, True)


    return preds, preds_orig

def transform_tensor(point, center, scale, resolution, device, invert=False):
    """Generate and affine transformation matrix.

    Given a set of points, a center, a scale and a targer resolution, the
    function generates and affine transformation matrix. If invert is ``True``
    it will produce the inverse transformation.

    Arguments:
        point {torch.tensor} -- the input 2D point
        center {torch.tensor or numpy.array} -- the center around which to perform the transformations
        scale {float} -- the scale of the face/object
        resolution {float} -- the output resolution

    Keyword Arguments:
        invert {bool} -- define wherever the function should produce the direct or the
        inverse transformation matrix (default: {False})
    """
    B, C = point.shape[0], point.shape[1]
    _pt = torch.ones(B,C, 3)
    _pt[:,:, 0] = point[:,:, 0]
    _pt[:,:, 1] = point[:,:, 1]

    # h = torch.tensor(200.0 * scale).unsqueeze(0).unsqueeze(0).repeat(B,C,1,1).squeeze(0).to(device)
    # h = (torch.full((B, C, 1, 1), 200.0) * scale.unsqueeze(2).unsqueeze(3)).to(device)
    # h = torch.full((B, C, 1, 1), 200.0) * scale.unsqueeze(2).unsqueeze(3)
    h = torch.full((B, C), 200.0).to(device) * scale
    # h = (200.0*scale).squeeze(1).to(device)

    # resolution = torch.tensor(resolution).unsqueeze(0).repeat(B,1).squeeze(1).to(device)
    # resolution = torch.full((B,C,1),resolution).to(device)

    resolution = torch.tensor(resolution).to(device)
    t = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B,C,1,1).to(device)
    half = torch.tensor(0.5).to(device)
    # center = center.repeat(1,68).to(device)
    center = center.unsqueeze(1).repeat(1, 68, 1).to(device)
    t[:,:, 0, 0] = resolution / h
    t[:,:, 1, 1] = resolution / h
    t[:,:, 0, 2] = resolution * (-center[:,:,0] / h + half)
    t[:,:, 1, 2] = resolution * (-center[:,:,1] / h + half)

    if invert:
        t = torch.pinverse(t)

    _pt = _pt.unsqueeze(-1).to(device)
    new_point = (torch.matmul(t, _pt))[:,:,0:2]

    return new_point.int()

def bbox2point(left, right, top, bottom) :
    old_size = (right-left+bottom-top)/2*1.1
    center_x = right-(right-left)/2.0
    center_y = bottom-(bottom-top)/2.0
    center = torch.stack([center_x, center_y], axis=1)
    
    return old_size, center



def bbpoint_warp(image, center, size, target_size_height, target_size_width=None, output_shape=None, inv=True, landmarks=None, 
        order=3 # order of interpolation, bicubic by default
        ):
    target_size_width = target_size_width or target_size_height

    # print(f'image : {image.shape}')
    # print(f'center : {center.shape}')
    # print(f'size : {size.shape}')
    # print(f'target_size_height : {target_size_height}')
    # print(f'target_size_width : {target_size_width}')
    tform = point2transform(center, size, target_size_height, target_size_width) # ()
    tf = torch.inverse(tform) if inv else tform
    output_shape = output_shape or (target_size_height, target_size_width)
    dst_image = warp_image(image, tform, output_shape, order=order)
    # dst_image = warp(image, tf, output_shape=output_shape, order=order)
    if landmarks is None:
        return dst_image
    # points need the matrix
    if isinstance(landmarks, np.ndarray):
        assert isinstance(landmarks, np.ndarray)
        tf_lmk = tform if inv else tform.inverse
        dst_landmarks = tf_lmk(landmarks[:, :2])
    elif isinstance(landmarks, list): 
        tf_lmk = tform if inv else tform.inverse
        dst_landmarks = [] 
        for i in range(len(landmarks)):
            dst_landmarks += [tf_lmk(landmarks[i][:, :2])]
    elif isinstance(landmarks, dict): 
        tf_lmk = tform if inv else tform.inverse
        dst_landmarks = {}
        for key, value in landmarks.items():
            dst_landmarks[key] = tf_lmk(landmarks[key][:, :2])
    else: 
        raise ValueError("landmarks must be np.ndarray, list or dict")
    return dst_image, dst_landmarks

def point2transform(center, size, target_size_height, target_size_width):
    target_size_width = target_size_width or target_size_height
    src_pts = point2bbox(center, size)
    dst_pts = torch.tensor([[0, 0], [0, target_size_width - 1], [target_size_height - 1, 0]])
    tform = estimate_transform('similarity', src_pts, dst_pts)


    tform = torch.linalg.lstsq(src_pts, dst_pts).solution.t()


    return tform

def point2bbox(center, size):
    # size2 = size.unsqueeze(-1).repeat(1,2) / 2
    size2 = size/2

    src_pts = torch.stack( # return (top left, bottom left, top right  )
        [torch.stack([center[:,0] - size2, center[:,1] - size2],dim=1), torch.stack([center[:,0] - size2, center[:,1] + size2],dim=1),
         torch.stack([center[:,0] + size2, center[:,1] - size2],dim=1)], dim=1)

    return src_pts

def bbpoint_warp_resize(batch_images, center, size, target_size_height, target_size_width=None, output_shape=None, inv=True, landmarks=None, 
        order=3 # order of interpolation, bicubic by default
        ):
    target_size_width = target_size_width or target_size_height

    src_pts = point2bbox(center, size) #(top left, bottom left, top right  )
    left = src_pts[:,0,0]
    top = src_pts[:,0,1]
    height = size # bbox height
    width = size # bbox width
    scale_ratios = target_size_height / size # bbox에서 dst image로의 scale factor

    dst_batch_images = []
    for i,image in enumerate(batch_images) :
        scale_ratio = scale_ratios[i]
        resized_image = torchvision.transforms.functional.resize(image, (int(image.shape[1]*scale_ratio), int(image.shape[2]*scale_ratio)))

        resized_top = top[i] * scale_ratio
        resized_left = left[i] * scale_ratio

        center = [resized_top + target_size_height/2, resized_left + target_size_width/2]
        dst_top = center[0] - target_size_height/2
        dst_left = center[1] - target_size_width/2
 
        cropped_image = torchvision.transforms.functional.crop(resized_image, int(dst_top), int(dst_left), int(target_size_height), int(target_size_width))
        dst_batch_images.append(cropped_image)  

    dst_batch_images = torch.stack(dst_batch_images, dim=0).to(batch_images.device)
    return dst_batch_images
    # print(src_pts)
    # 아래와 같이 crop하고 resize하면 index들이 음수가 나오는 경우들이 있어서 불가능
    # cropped_images = torchvision.transforms.functional.crop(image, top, left, height, width)
    # resized_images = torchvision.transforms.functional.resize(cropped_images, (target_size_height, target_size_width))

