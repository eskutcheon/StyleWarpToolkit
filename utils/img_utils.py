import os
import torch
import torchvision.transforms as T
import torchvision.io as IO
import torch.nn.functional as F
from torchvision.utils import draw_segmentation_masks
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Tuple, Dict, List
# project files
from functools import wraps

# dictionary to replace the use of CAVS_Occlusion_SSM.config.config with minimal refactoring
cfg = {
    "CLASS_NUM": 4,
    "COLOR_LABELS" : {
        "clean": (0,0,0), # black
        "transparent": (0,255,0), # green
        "semi_transparent": (0,0,255), # blue
        "opaque": (255,0,0) # red
    },
    "COLOR_INT_LABELS" : {
        "clean": 0, # black
        "transparent": 1, # green
        "semi_transparent": 2, # blue
        "opaque": 3 # red
    },
}


# NOTE: used to be in utils (in the Segmentation_Improvement repo) before I started eliminating circular dependencies
def enforce_type(target_type):
    def decorator(func):
        def wrapper(img, *args, **kwargs):
            if target_type == "tensor":
                if isinstance(img, np.ndarray):
                    img = ndarray_to_tensor(img)
            elif target_type == "ndarray":
                if torch.is_tensor(img):
                    img = tensor_to_ndarray(img.cpu())
            else:
                raise ValueError("Unsupported target type, must be either 'tensor' or 'ndarray'")
            return func(img, *args, **kwargs)
        return wrapper
    return decorator

def ensure_image_settings(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract existing positional arguments based on the function signature
        arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
        args_dict = dict(zip(arg_names, args))
        # Update kwargs with any positional arguments provided
        kwargs.update(args_dict)
        # Set color_opts and num_classes if not provided
        if 'color_opts' not in kwargs or kwargs['color_opts'] is None:
            kwargs['color_opts'] = get_mask_colors()
        if 'num_classes' not in kwargs or kwargs['num_classes'] is None:
            kwargs['num_classes'] = len(kwargs['color_opts']) if kwargs['color_opts'] is not None else cfg["CLASS_NUM"]
        return func(**kwargs)
    return wrapper

# TODO: may want another decorator to enforce all tensor inputs are on the same device or the device specified


def is_batch_tensor(tensor: torch.Tensor):
    ''' test if shape is (N,C,H,W) or (C,H,W) '''
    return int(len(tensor.shape) == 4) # 0 if shape is (C,H,W), 1 if shape is (N,C,H,W)

def is_label_tensor(tensor: torch.Tensor, num_classes: int):
    ''' test if the only values in the tensor are in range(num_classes), i.e., if uniques is a subset of range(num_classes)'''
    uniques = torch.unique(tensor).to(dtype=tensor.dtype)
    return torch.all(torch.any(uniques[:, None] == torch.arange(num_classes+1, dtype=uniques.dtype, device=tensor.device), dim=1))

def is_C_channel(tensor: torch.Tensor, C: int):
    ''' test if C categorical labels are in C dimensions '''
    return (tensor.shape[is_batch_tensor(tensor)] == C)

def is_flat_label_tensor(tensor: torch.Tensor, num_classes: int):
    ''' test if values are class labels and if labels are all in one dim'''
    return is_C_channel(tensor, 1) and is_label_tensor(tensor, num_classes)

def is_onehot_encoded_tensor(tensor: torch.Tensor, num_classes: int):
    ''' test if the tensor is encoded in C channels and whether values are in {0,1} '''
    # torch.equal not implemented for bool, so I had to cast it to uint8
    return is_C_channel(tensor, num_classes) and is_label_tensor(tensor.to(torch.uint8), 2)

def get_mask_colors() -> List[Tuple]:
    ''' primary job of this one is to align the colors used in the dataset masks with our output
        Returns:
            list of RGB tuples given in the config (in range 0-255)'''
    color_label_dict = cfg["COLOR_LABELS"]
    color_int_dict = cfg["COLOR_INT_LABELS"]
    return [color for _, color in sorted(color_label_dict.items(), key=lambda items: color_int_dict[items[0]])]

# TODO: should really consider rewriting these to use torchvision.transforms.v2.ToTensor 
def tensor_to_ndarray(tensor: torch.Tensor) -> np.ndarray:
    ''' convert pytorch tensor in shape (N,C,H,W) or (C,H,W) to ndarray of shape (N,H,W,C) or (H,W,C) '''
    assert isinstance(tensor, torch.Tensor), f"input must be a torch.Tensor object; got {type(tensor)}"
    if tensor.dim() not in [3,4]:
        return tensor.cpu().numpy()
    # TODO: check if tensor is already permuted to shape below
    # TODO: need to account for the case where a batch tensor of 2D arrays are passed (of shape (N,H,W))
    is_batch = is_batch_tensor(tensor)
    new_dims = (0,2,3,1) if is_batch else (1,2,0)
    np_array = np.transpose(tensor.cpu().numpy(), new_dims)
    return np_array

def ndarray_to_tensor(arr: np.ndarray) -> torch.Tensor:
    ''' convert np.ndarray with shape (N,H,W,C) or (H,W,C) to torch.tensor of shape (N,C,H,W) or (C,H,W) '''
    assert isinstance(arr, np.ndarray), f"input must be a numpy.ndarray object; got {type(arr)}"
    if arr.ndim not in [3,4]:
        return torch.from_numpy(arr)
    is_batch = (arr.ndim == 4)
    new_dims = (0,3,1,2) if is_batch else (2,0,1)
    tensor = torch.from_numpy(np.transpose(arr, new_dims))
    return tensor

def is_int_dtype(img):
    return img.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool)

def is_float_dtype(img):
    return img.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16)


def is_valid_tensor_values(tensor: torch.Tensor, valid_set: set) -> bool:
    # tensor should usually be passed as some_tensor.unique() for efficiency, but should also work with the original tensor
        # TODO: should confirm this works as expected later, since I remember some issue with nested lists as input to set()
    tensor_set = set(tensor.tolist())
    return tensor_set.issubset(valid_set)

def is_valid_shape(tensor):
    def check_channels(num_dims, channel_idx):
        if num_dims in [2,3,4]:
            if num_dims != 2:
                if tensor.shape[channel_idx[num_dims]] not in [1,3]:
                    raise ValueError(f"expected channel dimension to be in {{1,3}} in index 0 for 3D tensor or index 1 for batch tensor. Got shape {tensor.shape}")
        else:
            raise ValueError(f"image tensor must be 2D, 3D, or 4D")
        return True
    num_dims = len(tensor.shape)
    if isinstance(tensor, torch.Tensor):
        channel_idx = {4: 1, 3: 0}
        return check_channels(num_dims, channel_idx)
    elif isinstance(tensor, np.ndarray):
        channel_idx = {4: -1, 3: -1}
        return check_channels(num_dims, channel_idx)
    else:
        raise NotImplementedError(f"method only implemented for torch.Tensor, np.ndarray, got {type(tensor)}")

def test_if_valid_img(img: torch.Tensor, is_int: bool=None, is_float: bool=None):
    # TODO: not sure if I want to add a test to see if the first dimension is 3
    is_int = is_int_dtype(img) if is_int is None else is_int
    is_float = is_float_dtype(img) if is_float is None else is_float
    if torch.min(img) < 0 or torch.max(img) > 255:
        raise ValueError(f"The given tensor is not an image; detected (min,max)=({torch.min(img)},{torch.max(img)})")
    if not (is_int or is_float):
        raise ValueError(f"The given tensor is not an image with valid dtype of int or float; got dtype={img.dtype}")

def is_boolean_mask(img: torch.Tensor) -> bool:
    if img.dtype == torch.bool:
        return True
    uniques = torch.unique(img)
    is_binary = (len(uniques) < 3)
    is_valid = is_valid_tensor_values(uniques, {0, 1}) or is_valid_tensor_values(uniques, {0, 255})
    return is_binary and is_valid


def get_normalized_image(img: torch.Tensor) -> torch.FloatTensor:
    TOL = 10e-8
    if is_int_dtype(img) and torch.max(img) > 1:
        img = img.to(dtype=torch.float32)/255
    elif is_float_dtype(img) and torch.max(img) > 1+TOL:
        img /= 255
    return img


def get_uint_image(img: torch.Tensor) -> torch.ByteTensor:
    TOL = 10e-8
    is_int = is_int_dtype(img)
    is_float = is_float_dtype(img)
    test_if_valid_img(img, is_int, is_float)
    new_img = img
    # doing this is the off chance that an in img with only black and/or white values in {0,1} is given
    if is_int:
        uniques = torch.unique(img)
        if is_valid_tensor_values(uniques, {0, 1}):
            new_img = 255*(new_img.to(dtype=torch.float32))
        elif is_valid_tensor_values(uniques, {0, 255}) and uniques.dtype == torch.uint8:
            return new_img
    elif is_float:
        if torch.max(new_img) < 1+TOL:
            new_img = new_img*255
    return new_img.clamp(0,255).to(dtype=torch.uint8)


def get_flat_grayscale_image(img: torch.Tensor) -> torch.Tensor:
    img_normed = get_normalized_image(img)
    grayscale_transform = T.Grayscale(num_output_channels=1)
    is_batch = is_batch_tensor(img)
    # with flat grayscale transformation and squeeze below, dim=(H,W) or (N,H,W) if is_batch
    img_map_gray = grayscale_transform(img_normed).squeeze(dim=is_batch)
    return img_map_gray

def get_reshaped_tensor(tensor, dims, interp_mode='nearest'):
    # TODO: need to also check that dim's size is correct later
    # should also ensure mode stays the same as what interpolated it from HxW first
    return F.interpolate(tensor, size=tuple(dims), mode=interp_mode)


def logits_to_labels(preds: torch.Tensor, num_classes: int, dims: Union[torch.Size, tuple, list]=None) -> torch.Tensor:
    ''' accepts logits as multichannel input, returns long tensor of class indices
        Args:
            preds: multichannel logit predictions in shape (N,C,H,W) or (C,H,W)
            num_classes: number of channels C
            dims: The resize dimensions if applicable
        Returns
            long tensor of class labels of shape (1,H,W) or (N,1,H,W)
    '''
    assert num_classes > 1, f"Number of classes must be greater than 1"
    if not is_C_channel(preds, num_classes):
        raise ValueError(f"expected {num_classes}-channel input in shape (N,C,H,W) or (C,H,W), got shape {tuple(preds.shape)}")
    is_batch = is_batch_tensor(preds)
    if dims is not None:
        # reshape logits tensor back into original size given by dims
        if is_batch:
            preds = get_reshaped_tensor(preds, dims)
        else: # if not a batch, turn it into one then reshape
            is_batch = int(not is_batch)
            preds = get_reshaped_tensor(preds.unsqueeze(0), dims)
    #util.get_all_debug(softmaxed, "softmaxed")
    pred_indices = torch.argmax(F.softmax(preds, dim=is_batch), dim=is_batch, keepdim=True)
    return pred_indices.to(dtype=torch.uint8)


def logits_to_onehot(tensor: torch.Tensor, num_classes: int) -> torch.Tensor:
    ''' take tensor of logit values and convert to boolean onehot encoding
        Args:
            tensor: tensor of logit values that satisfies input requirements of logits_to_labels
            num_classes: number of classes C in tensor input
        Returns:
            one-hot encoded boolean tensor in shape (C,H,W) or (N,C,H,W)
    '''
    # call to pred_indices means it has the same input shape restrictions
    pred_indices = logits_to_labels(tensor, num_classes)
    return labels_to_onehot(pred_indices, num_classes)


def labels_to_onehot(idx_tensor: torch.LongTensor, num_classes: int) -> torch.Tensor:
    ''' mostly intended to be used by prediction functions above - may change later
        Args:
            idx_tensor: tensor of C different categorical labels of shape (1,H,W) or (N,1,H,W)
            num_classes: number of classes C in tensor input
        Returns:
            one-hot encoded boolean tensor in shape (C,H,W) or (N,C,H,W)
    '''
    is_batch = is_batch_tensor(idx_tensor)
    out_shape = list(idx_tensor.shape)
    out_shape[is_batch] = num_classes
    # because F.one_hot() returns an annoying shape
    one_hot = torch.zeros(size=out_shape).to(idx_tensor.device, dtype=torch.bool)
    # this method only accepts long tensors for some reason
    return one_hot.scatter_(is_batch, idx_tensor.to(dtype=torch.int64), 1)

# TODO: reshape_mask does the same thing as the above. run tests on speed later
def reshape_mask(masks, pred_shape, num_classes):
    is_batch = is_batch_tensor(masks)
    return torch.reshape(F.one_hot(masks.squeeze(is_batch).to(dtype=torch.int64), num_classes), pred_shape)

###########################################################################################################

def onehot_to_labels(tensor: torch.Tensor, num_classes: int):
    ''' converts a one-hot encoded boolean tensor to a class label encoding
        Args:
            tensor: one-hot encoded tensor of shape (N,C,H,W) or (C,H,W) and dtype torch.bool
            num_classes: number of classes C in tensor input
        Returns:
            label_tensor: an unsigned int tensor of categorical labels of shape (N,1,H,W) or (1,H,W)
    '''
    if not is_C_channel(tensor, num_classes):
        raise ValueError(f'Expected C-channel tensor of shape (N,C,H,W) or (C,H,W), got {tuple(tensor.shape)}')
    if not is_onehot_encoded_tensor(tensor, num_classes):
        raise ValueError(f'Expected one-hot encoded binary tensor, got elements {list(torch.unique(tensor))}')
    is_batch = is_batch_tensor(tensor)
    output_shape = list(tensor.shape)
    output_shape[is_batch] = 1
    label_tensor = torch.zeros(size=output_shape, dtype=torch.uint8)
    for i in range(1,num_classes):
        tensor_accessed = tensor[:, i, ...] if is_batch else tensor[i, ...]
        tensor_accessed = tensor_accessed.unsqueeze(is_batch)
        label_tensor[tensor_accessed] += i
    return label_tensor


# might rewrite this later to use a method different from onehot_to_image
@ensure_image_settings
def labels_to_image(pred: torch.Tensor, color_opts: list = None, num_classes: int = None) -> torch.Tensor:
    ''' takes a flattened tensor of class labels and returns a 3-channel colored image tensor
        Args:
            pred: one-hot segmentation mask with either shape (N,1,H,W) or (1,H,W)
            color_opts: list of RGB tuples numbered according to the order in the dataset
            num_classes: The number of class labels C
        Returns:
            an RGB tensor in shape (N,3,H,W) or (3,H,W) that may be saved as a PIL image
    '''
    if pred.dtype in [bool, torch.bool, np.bool]:
        pred = pred.to(dtype=torch.uint8)
    if not is_flat_label_tensor(pred, num_classes):
        raise Exception(f'pred argument must be a single-channel tensor of labels 0-{num_classes-1}')
    onehot_mask = labels_to_onehot(pred, num_classes)
    return onehot_to_image(onehot_mask, color_opts, num_classes)

@ensure_image_settings
def onehot_to_image(pred: torch.Tensor, color_opts: list = None, num_classes: int = None) -> torch.Tensor:
    ''' takes one-hot segmentation mask and returns a 3-channel, colored image tensor
        Args:
            pred: one-hot segmentation mask with either shape (N,C,H,W) or (C,H,W)
            color_opts: list of RGB tuples numbered according to the order in the dataset
            num_classes: The number of class labels C
        Returns:
            an RGB tensor in shape (N,3,H,W) or (3,H,W) that may be saved as a PIL image
    '''
    if not is_onehot_encoded_tensor(pred, num_classes):
        raise Exception(f'pred argument must be one-hot encoded binary tensor with {num_classes} channels')
    # TODO: allow for batch inputs?
    is_batch = is_batch_tensor(pred)
    # treat it as a batch either way:
    if not is_batch:
        pred = pred[None, ...]
    N, C, H, W = pred.shape
    img = torch.zeros((N,3,H,W), dtype=torch.uint8) # preallocate img tensor
    palette = torch.tensor(color_opts, dtype=torch.uint8, device=pred.device)
    for class_i in range(num_classes):
        # Get the indices where the prediction tensor is not zero for each class
        indices = torch.nonzero(pred[:, class_i], as_tuple=True)
        # Set color of pixels at the 1 indices to class_i's color
        img[indices[0], :, indices[1], indices[2]] = palette[class_i].view(1, -1)
    # change back to a single image if it wasn't a batch tensor to start with
    return img.squeeze(0) if not is_batch else img

# TODO: TWEAK, TEST, AND DOCUMENT THE REMAINING FUNCTIONS:

def read_and_resize_image(img_path: str, out_size: Union[Tuple[int], List[int]], is_labels: bool = True) -> torch.Tensor:
    ''' 3-channel image will be read from file as (3,*out_size) tensor with values 0-255
        a flat tensor of mask values will be read as (1, *out_size) tensor with values 0 to C-1
    '''
    read_mode = IO.ImageReadMode.UNCHANGED if is_labels else IO.ImageReadMode.RGB
    interp_mode = T.InterpolationMode.NEAREST if is_labels else T.InterpolationMode.BILINEAR
    # TODO: really need to split this into two functions and add checks if out_size < img shape
    # docs are inaccurate, antialias=True is NOT simply ignored if interpolation mode isn't bicubic or bilinear
    img_resize: T.Resize
    if interp_mode in [T.InterpolationMode.BILINEAR, T.InterpolationMode.BICUBIC]:
        img_resize = T.Resize(out_size, interp_mode, antialias=True)
    else:
        img_resize = T.Resize(out_size, interp_mode)
    return img_resize(IO.read_image(img_path, read_mode))

def get_normed_batch_tensor(tensor:torch.Tensor):
    return torch.unsqueeze(torch.div(tensor, 255), 0)


# may add another function to accept label tensors similar to how labels_to_image is used
@ensure_image_settings
def get_mask_overlay(img: torch.Tensor, onehot_mask: torch.Tensor, color_opts:list = None, num_classes: int = None, transparency:float=0.1) -> torch.Tensor:
    ''' performs a segmentation mask overlay of a corresponding image tensor
        Args:
            img: The base image tensor on which the mask will be overlaid; in shape (3,H,W)
            onehot_mask: The one-hot encoded segmentation mask in shape (C,H,W)
            color_opts: the ordered color options that the mask will be filled in with
            num_classes: the number of class labels in the segmentation mask
            transparency: The index of transparency for the masks in range 0-1 (fully transparent to opaque)
        Returns
            an overlaid image mask in shape (3,H,W)
    '''
    # TODO: need to make some custom Exceptions at some point
    if not is_C_channel(img, 3):
        raise Exception('img must be a 3 color channel image tensor')
    if not is_onehot_encoded_tensor(onehot_mask, num_classes):
        raise Exception(f'pred_mask must be a one-hot encoded tensor with {num_classes} channels')
    if is_batch_tensor(img) or is_batch_tensor(onehot_mask):
        raise Exception('batches not supported')
    # TODO: add assertions for color_opts and transparency types and values
    overlay = torch.clone(img.to(dtype=torch.uint8))
    for class_i in range(num_classes):
        # NOTE: one of the most annoying functions with the worst error messages
        # draw_segmentation_masks requires pred_mask be Boolean and that img and mask be on the same device
        # additionally, some default tensor settings require that both be on the GPU
        overlay = draw_segmentation_masks(overlay, onehot_mask[class_i], alpha=transparency, colors=color_opts[class_i])
    return overlay


# TODO: still need an image to labels function

def get_pred_error_mask(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # TODO: should be able to allow for image batches later
    ''' takes a segmentation mask prediction and ground truth mask and plots true and false positives
        Args:
            pred: the prediction segmentation mask - label tensor of shape (1,H,W) or one-hot tensor of shape (C,H,W)
            mask: segmentation mask as a label tensor of shape (1,H,W) or onehot tensor of shape (C,H,W)
        Returns:
            an image tensor of shape (3,H,W) with correct preds in black, misclassifications in white
    '''
    # pred and mask both need to be label tensors of the same size - add test later
    if pred.shape != mask.shape:
        raise ValueError(f'pred and mask arguments must be the same shape, got {pred.shape} and {mask.shape}')
    if is_batch_tensor(pred):
        raise NotImplementedError("batch pred or mask input not supported")
    C = pred.shape[0] # C should be 1 for labels, but it should be able to take one-hot tensors as well
    '''if not is_onehot_encoded_tensor(pred, C) or not is_onehot_encoded_tensor(mask, C):
        raise ValueError('pred and mask arguments must be one-hot encoded tensors')'''
    misclassifications = (pred != mask)
    # Using all white so that it works with onehot_to_image to simply color the pixel with 1's values
    palette = [(255,255,255) for _ in range(C)]
    return onehot_to_image(misclassifications, palette, C)


def get_pred_batch_dict(sample_ptr):
    N, _, H, W = sample_ptr["preds"].shape
    batch_to_log = {
        "names": list(map(os.path.basename, sample_ptr["img_paths"])),
        "preds": sample_ptr["preds"],
        "images": torch.zeros((N,3,H,W)),
        "masks": torch.zeros((N,1,H,W))}
    for i, (img_path, mask_path) in enumerate(zip(sample_ptr['img_paths'], sample_ptr['mask_paths'])):
        batch_to_log["images"][i] = read_and_resize_image(img_path, (H,W), is_labels=False)
        batch_to_log["masks"][i] = read_and_resize_image(mask_path, (H,W), is_labels=True)
    #util.get_all_debug(batch_to_log["preds"][0], "batch_to_log")
    return batch_to_log

def get_pred_outputs(sample: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]], transparency=0.1) -> Dict[str, torch.Tensor]:
    ''' what i think the state is when called from train (after reading it in below):
            preds as logits in shape (N,4,512,512), dtype=torch.float32, device=cuda:0
            images: image batch in shape (N,3,512,512), dtype=torch.float32, device=cuda:0
            masks: mask batch in shape (N,1,512,512), dtype=torch.int64, device=cuda:0
    '''
    # NOTE: Rethink the interpolation methods' effects on the hallucinations later
    preds = sample['preds']
    # ensure these are all in integer ranges later - removed 255*
    images = sample['images'].to(preds.device) # values 0-255
    masks = sample['masks'].to(preds.device, dtype=torch.long) # values {0,1,2,3}
    # TODO: the indexing below assumes input to this function is always a batch - add check
    assert is_batch_tensor(preds), "'preds' within the passed sample must be a batch tensor"
    assert is_batch_tensor(images), "'images' within the passed sample must be a batch tensor"
    assert is_batch_tensor(masks), "'masks' within the passed sample must be a batch tensor"
    num_classes = preds.shape[1]
    img_dict = {}
    for i, img_name in enumerate(sample["names"]):
        with torch.no_grad():
            # TODO: need to add a check to see if pred_indices is already a byte tensor
            # pred_indices in range {0,1,2,3}, shape=(1,512,512), torch.int64
            pred_indices = logits_to_labels(preds[i], num_classes)
            onehot_preds = labels_to_onehot(pred_indices, num_classes)
            img_dict[img_name] = {
                'pred mask': labels_to_image(pred_indices, num_classes=num_classes),
                'pred overlay': get_mask_overlay(images[i], onehot_preds, num_classes=num_classes, transparency=transparency),
                'truth mask': labels_to_image(masks[i], num_classes=num_classes),
                'misclassifications': get_pred_error_mask(pred_indices, masks[i])
            }
    return img_dict
