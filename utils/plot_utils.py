import os, sys
import torch
import matplotlib.pyplot as plt
from matplotlib import get_backend
import numpy as np
from typing import Union, Tuple, Dict, List
# project files - renamed from the module import used in the main code
import utils.img_utils as img_util

def maximize_window():
    manager = plt.get_current_fig_manager()
    backend = get_backend()
    if backend == 'TkAgg':
        if sys.platform.startswith('win'):  # For windows
            manager.window.state('zoomed')
        else:  # For Linux
            manager.window.wm_attributes('-zoomed', '1')
    elif backend == 'Qt5Agg':
        manager.window.showMaximized()
    elif backend == 'WXAgg':
        manager.window.Maximize()
    else:
        print(f"WARNING: Unsupported backend {backend} for maximize operation")
    return manager

def convert_batch_to_list(imgs):
    assert(img_util.is_batch_tensor(imgs)), f"argument 'imgs' must be a 4D batch tensor"
    return [img for img in imgs]

def image_plot_preprocessing(img: Union[torch.Tensor, np.ndarray]):
    ''' TODO: test input image for the following:
        * check if np.ndarray or torch.Tensor
        * ensure valid dtype - convert if needed
        * check shape and account for (N,C,H,W), (C,H,W), (H,W), (H,W,C), and (N,H,W,C)
            * if (H,W), check if binary image or label tensor
    '''
    if isinstance(img, torch.Tensor):
        pass
    elif isinstance(img, np.ndarray):
        pass
    else:
        raise NotImplementedError(f"method only implemented for torch.Tensor, np.ndarray, got {type(img)}")

def show_image(imgs: torch.Tensor, title=None, show_all=None):
    # TODO: above all, need to make this robust to both torch.tensor and np.ndarray - had issues with both recently
    # TODO: add checks for the following:
        # if tensor is 3-channel
        # check to make sure tensor is the right dtype
        # if in shape (3,H,W), permute to (H,W,3)
        # whether pixel values are in range [0,1]
    # TODO: really want to extend this for grids of images and to take the same arguments as save_multiple_images
    plt.rcParams["figure.autolayout"] = True
    if img_util.is_batch_tensor(imgs):
        imgs = convert_batch_to_list(imgs)
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    if title is not None:
        plt.title(title)
    for i, img in enumerate(imgs):
        if isinstance(img, torch.Tensor) and img.device != torch.device('cpu'):
            img = img.cpu()
        axs[0, i].imshow(img_util.tensor_to_ndarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    manager = maximize_window()
    plt.show(block=show_all)


# FIXME: mostly written by ChatGPT and needs testing - edited heavily by me
def plot_images(*images, title=None, cols=2):
    """ Plots a variable number of images.
        Args:
            *images: Variable-length list of images to plot.
            title (str): Title for the entire plot.
            cols (int): Number of columns for the subplot grid.
    """
    if (len(images) == 1) and img_util.is_batch_tensor(images[0]):
        images = convert_batch_to_list(images[0])
    # ~FIXME: circle back to finish this logic later - the above doesn't handle batch tensors when given more than 1 positional argument
    # need to ensure batch dimensions of all inputs are consistent -> save unpacked to a 2D list -> iterate over batch dim w/ plotting for each image in the batch
    '''if (len(images) > 1) and any(img_util.is_batch_tensor(img) for img in images):
        temp = []
        temp.extend(convert_batch_to_list(img) for img in images if img_util.is_batch_tensor(img))
        images = temp
    else:
        images = convert_batch_to_list(images[0])'''
    num_images = len(images)
    rows = num_images // cols + num_images % cols
    cols = num_images if num_images < cols else cols
    position = range(1, num_images + 1)
    fig = plt.figure(figsize=(20, 30), facecolor = 'lightgray')
    for k, img in zip(position, images):
        if not img_util.is_valid_shape(img):
            raise ValueError(f"Cannot accept img of type {type(img)} with shape {img.shape}")
        ax = fig.add_subplot(rows, cols, k)
        if isinstance(img, torch.Tensor):
            img = img_util.tensor_to_ndarray(img_util.get_uint_image(img).cpu())
        elif isinstance(img, np.ndarray):
            img = img.astype(int)
        if img.ndim == 2:  # Grayscale
            plt.imshow(img, cmap='gray', aspect='auto')
        else:  # Color
            plt.imshow(img, aspect='auto')
        plt.axis('off')
    if title:
        fig.suptitle(title, fontsize='xx-large')
    manager = maximize_window()
    plt.show()


def show_binary_image(img: torch.tensor, title: str=None):
    # TODO: handle batch case in a new function eventually
    img_expanded = img_util.labels_to_image(img, [(0,0,0),(255,255,255)], 2)
    show_image(img_expanded.permute(1,2,0).to(device='cpu'), title=title)


def show_grayscale_image(img: torch.tensor, title: str=None):
    # TODO: allow for numpy types later then remove this assertion
    assert isinstance(img, torch.Tensor), f"input must be a torch.Tensor object; got {type(img)}"
    if not (len(img.shape) == 3):
        img = img.unsqueeze(0).repeat(3,1,1)
    elif img.shape[0] == 1:
        img = img.repeat(3,1,1)
    show_image(img_util.get_uint_image(img).permute(1,2,0), title=title)

def labels_to_plottable_mask(labels):
    # FIXME: only written to work in a narrow case for now - remember to flesh out later
    if len(labels.shape) > 3:
        labels = labels.squeeze(0)
    return img_util.tensor_to_ndarray(img_util.labels_to_image(labels))

def plot_image_diff_superpixels(input_image, target_image, title, pool_size=10):
    # Ensentially plotting mean absolute error along the channels
    difference = torch.mean(torch.abs(input_image - target_image).squeeze(0).to(dtype=torch.float32), dim=0)
    difference = torch.nn.functional.avg_pool2d(difference.unsqueeze(0), kernel_size=pool_size).squeeze(0)
    #util.get_all_debug(difference, 'diff')
    images_to_plot = {
        "input" : {"tensor": img_util.tensor_to_ndarray(input_image.squeeze().cpu()), "map": "gray"},
        "target": {"tensor": img_util.tensor_to_ndarray(target_image.squeeze().cpu()), "map": "gray"},
        "difference": {"tensor": img_util.tensor_to_ndarray(difference.cpu()), "map": "hot"}
    }
    plt.figure(figsize=(10, 4))
    plt.suptitle(title)
    for idx, (key, plotting_dict) in enumerate(images_to_plot.items()):
        plt.subplot(1, 3, idx+1)
        plt.imshow(plotting_dict["tensor"], cmap = plotting_dict["map"])
        plt.title(key)
        plt.axis('off')
    manager = maximize_window()
    plt.show()


def plot_image_diff_heatmap(diff, pool_size: tuple = None):
    """
    Plots a 2D heatmap of pixel-wise relative differences for each channel in the difference tensor as a heat map
    Args:
        diff: torch.Tensor of shape (C, H, W) or (1, H, W) representing the absolute differences. C can be 1 or 3.
        pool_size: size of the neighborhood over which to average for superpixel visualization (if provided)
    """
    num_channels = diff.shape[0]
    fig, axs = plt.subplots(1, num_channels, figsize=(num_channels*5, 5), squeeze=False)
    # Apply average pooling for superpixel effect if pool_size is specified
    if pool_size is not None:
        diff = torch.nn.functional.avg_pool2d(diff.float(), kernel_size=pool_size, stride=pool_size, padding=0)
    # Normalize the difference tensor to have values between 0 and 1
    diff -= diff.min()
    if diff.max() > 0:
        diff /= diff.max()
    for i in range(num_channels):
        channel_diff = diff[i].cpu().numpy()
        im = axs[0, i].imshow(channel_diff, cmap='coolwarm', aspect='auto')
        axs[0, i].axis('off')  # Turn off axis
        ax_title = f'Channel {i+1} Differences in Images' if num_channels == 3 else 'Single-Channel Differences in Mask'
        axs[0, i].set_title(ax_title)
        fig.colorbar(im, ax=axs[0, i], fraction=0.046, pad=0.04)  # Add a colorbar to a plot
    plt.tight_layout()
    plt.show()


def plot_difference_histogram(img_diff, num_bins=10):
    # FIXME: only written to work in a narrow case for now - remember to flesh out later
    # assume that img_diff is a 3D tensor of absolute differences in the images of shape (3,H,W); likewise for masks of shape (1,H,W)
    num_channels = img_diff.shape[0]
    fig, axs = plt.subplots(1, num_channels, figsize=(int(5*num_channels), 5), sharex=True, sharey=True, tight_layout=True)
    # when num_channels = 1, put it in a list since matplotlib doesn't keep that consistent on their end
    if num_channels == 1:
        axs = [axs]
    for channel in range(num_channels):
        #channel_diff = img_util.tensor_to_ndarray(img_diff[channel].clone().cpu())
        channel_diff = img_diff[channel].view(-1).cpu().numpy()
        axs[channel].hist(channel_diff, bins=num_bins, color=['r', 'g', 'b'][channel], log=True)
        axs[channel].set_title(f'Channel {channel+1} Difference Distribution (Log Scale)')
    plt.show()
        # increase the number of bins on each axis equally
        #axs[0].hist2d(dist1, dist2, bins=num_bins)
    #mean_dist = img_util.tensor_to_ndarray(torch.mean(img_diff, dim=0).cpu())

def create_grid(*images, tiling_dim):
    if (len(images) == 1) and img_util.is_batch_tensor(images[0]):
        images = convert_batch_to_list(images[0])
    # ~FIXME: circle back to finish this logic later - the above doesn't handle batch tensors when given more than 1 positional argument
    # need to ensure batch dimensions of all inputs are consistent -> save unpacked to a 2D list -> iterate over batch dim w/ plotting for each image in the batch
    num_images = len(images)
    # do stuff later