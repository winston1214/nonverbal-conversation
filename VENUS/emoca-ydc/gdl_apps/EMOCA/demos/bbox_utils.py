import torch
from torchvision import transforms
import os
import numpy as np

def transform(point, center, scale, resolution, invert=False):
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
    _pt = torch.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = 200.0 * scale
    t = torch.eye(3)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if invert:
        t = torch.inverse(t)

    new_point = (torch.matmul(t, _pt))[0:2]

    return new_point.int()

def sfd_bbox_to_fan_bbox(sfd_bbox, reference_scale=195. , resolution=256.):
    """
    Convert a bounding box from the SFD format to the FAN format.
    WARNING : this function assumes that the sfd BBOX is perfect when transformed to FAN BBOX (no out of bound values)
    Parameters:
    - sfd_bbox: A bounding box in the SFD format [start_x, start_y, end_x, end_y].

    Returns:
    - fan_bbox: A bounding box in the for FAN in original_image pixel space
    """
    start_x, start_y, end_x, end_y = sfd_bbox

    # find center and scale for affine transformation
    center = torch.tensor( 
        [end_x - (end_x - start_x) / 2.0,  # x_center
         end_y - (end_y - start_y) / 2.0]) # y_center
    # not sure why but the center is shifted by 12% of the hieght of bbox
    center[1] = center[1] - (end_y - start_y) * 0.12 
    scale  = (end_x - start_x + end_y - start_y) / reference_scale 

    upper_left = transform([1, 1], center, scale, resolution, True) # start_x + 1, start_y + 1
    bottom_right = transform([resolution, resolution], center, scale, resolution, True) # end_x, end_y

    fan_bbox = torch.tensor([upper_left[0] - 1, upper_left[1] - 1, bottom_right[0], bottom_right[1]])
    scale = torch.tensor(scale)
    center = torch.tensor(center)
    
    # WARNING : this function assumes that the sfd BBOX is perfect when transformed to FAN BBOX (no out of bound values)
    # We have to ensure indices are within bounds
    # (JB 11-24) for now, skip and assume perfect bbox

    return fan_bbox, center, scale

def crop_batch(batch_tensors, bounding_boxes):
    """
    Crop a batch of tensors with different bounding boxes.

    Parameters:
    - batch_tensors: A batch of input tensors (B x C x H x W).
    - bounding_boxes: A tensor of bounding boxes, (B X 4) 
    where each bounding box is specified as a tensor  [start_x, start_y, end_x, end_y] 

    Returns:
    - cropped_batch: A batch of cropped tensors.(B x C x 256 X 256)
    """
    device = batch_tensors.device  # Get the device of the input tensors

    cropped_tensors = []
    B,C,height, width = batch_tensors.shape # this is the shape of original images

    for tensor, bbox in zip(batch_tensors, bounding_boxes):
        start_x, start_y, end_x, end_y = bbox

        # Ensure indices are within bounds
        # (JB 11-24) for now, skip and assume perfect bbox
        # (JB 11-25) there were events where bboxes are bigger than original image -> add padding
        pad_y= [0,0]
        pad_x=[0,0]
        if start_y < 0:
            pad_up = 0 - start_y # how much should we pad
            pad_y[0] = pad_up
            start_y = 0
        if end_y > height:
            pad_down = end_y - height
            pad_y[1] = pad_down
            end_y = height
        if start_x < 0:
            pad_left = 0 - start_x # how much should we pad
            pad_x[0] = pad_left
            start_x = 0
        if end_x > height:
            pad_right = end_x - width
            pad_x[1] = pad_right
            end_x = width
        pad_xy = tuple(pad_x + pad_y) # how much should we pad ex) (1,0,0,0)
        # Crop using narrow or index_select
        cropped_tensor = tensor.narrow(1, start_y, end_y - start_y).narrow(2, start_x, end_x - start_x)
        cropped_tensor = torch.nn.functional.pad(cropped_tensor,pad_xy)
        cropped_tensor = transforms.functional.resize(cropped_tensor,(256,256))
        cropped_tensors.append(cropped_tensor)

    # Stack the cropped tensors into a new batch
    cropped_batch = torch.stack(cropped_tensors)

    return cropped_batch.to(device)  # Ensure the cropped batch stays on the same device as the input batch

def save_tensor_as_image(image, dir, file_name):
    """
    Save a tensor as an image.

    Parameters:
    - image: A tensor of shape (C x H x W).
    - dir: The directory to save the image to.
    - file_name: The name of the file to save the image to.
    """
    # Create the directory if it does not exist
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Save the image
    transforms.ToPILImage()(image).save(os.path.join(dir, file_name))