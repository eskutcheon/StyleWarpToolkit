import torch
import torchvision.io as IO
from torchvision.transforms.v2 import Lambda
import numpy as np
import os
import json
from tqdm import tqdm
from typing import Union, Tuple, Dict, List

# dictionary to replace the use of CAVS_Occlusion_SSM.config.config with minimal refactoring
cfg = {
    "DATA_SOURCE" : "https://drive.google.com/uc?export=download&id=1Id-K7SjwCqWkLwtIGJUj5Q0Dw0E_TP_9",
    "CLASS_DISTRIBUTION" : {
        "clean": 0.3736, # black
        "transparent": 0.1223, # green
        "semi_transparent": 0.2040, # blue
        "opaque": 0.2999 # red
    },
    "DATA_DIRECTORY" : r"C:\Users\Jacob\Documents\MSU HPC2 Work\Camera Occlusion\TransUNET_Segmentation\data\soiling_dataset"
}

# NOTE: generally trying to put the most general-use functions near the top and the most task-specific and least-used at the bottom

class MultiArgLambda(Lambda):
    def __init__(self, lambd, *args, **kwargs):
        super().__init__(lambd)
        self.args = args
        self.kwargs = kwargs

    # overriding TT.Lambda.__call__ to take multiple arguments
    def __call__(self, img):
        return self.lambd(img, *self.args, **self.kwargs)


def tensor_to_cuda(tensor):
    # should only accept torch.Tensor or torchvision.tv_tensors.TVTensor - pretty sure issubclass works recursively if tensor is a TVTensor subclass
    if not (isinstance(tensor, torch.Tensor) or issubclass(tensor, torch.Tensor)):
        raise TypeError(f"Expected torch.Tensor or a subclass of torch.Tensor; got {type(tensor)}")
    if "cuda" in str(tensor.device):
        return tensor
    if not torch.cuda.is_available():
        print("WARNING: CUDA not found; check results of running 'nvcc --version'")
        return tensor
    return tensor.to(device="cuda")


def test_path(dir_path):
    if not os.path.exists(dir_path):
        raise Exception(f'Directory {dir_path} given is invalid')

def check_output_path(out_dir):
    # check if the new directories exist, create them if not
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

def get_matching_filenames(input_dir, substr):
    return list(filter(lambda x: (substr in x) and (not os.path.isdir(os.path.join(input_dir, x))), os.listdir(input_dir)))


def test_folder_lengths(img_paths, mask_paths, img_dir_name, mask_dir_name):
    assert len(img_paths) == len(mask_paths), "folders of images and mask are not the same length (after processing) with lengths:\n" + \
            f"{img_dir_name} - length {len(img_paths)} \n" + \
            f"{mask_dir_name} - length {len(mask_paths)}"

# mostly writing this just to be more explicit with calls for now
class Debugger(object):
    @staticmethod
    def get_min_max(arr, is_torch):
        min_func, max_func = (torch.min, torch.max) if is_torch else (np.min, np.max)
        print(f'\tmin and max: {min_func(arr), max_func(arr)}')

    @staticmethod
    def get_array_values(arr, is_torch):
        set_func = torch.unique if is_torch else np.unique
        print(f'\tvalues: {set_func(arr)}')

    @staticmethod
    def get_arr_shape(arr, is_torch):
        print(f'\tshape: {arr.shape}')

    @staticmethod
    def get_arr_type(arr, is_torch):
        print(f'\tdata type: {type(arr)}')

    @staticmethod
    def get_arr_device(arr, is_torch):
        if is_torch:
            print(f'\tdevice: {arr.device}')

    @staticmethod
    def get_element_type(arr, is_torch):
        print(f'\telement types: {arr.dtype}') # same for torch and numpy


def get_all_debug(arr, msg='unnamed tensor'):
    print(f'attributes of {msg}:')
    for func in filter(lambda x: callable(x), Debugger.__dict__.values()):
        func(arr, isinstance(arr, torch.Tensor))
    print()

# NOTE: pretty much writing this to drop those files with contentions first - may want to generalize further
    # going to read from the metadata json for future consistency
def get_files_to_drop(json_path: str, mode: str) -> List[str]:
    with open(json_path, 'r') as fptr:
        meta_dict = dict(json.load(fptr))
    return [name for name, file_dict in meta_dict.items()
                if file_dict["contention"] and file_dict["image_subset"] == mode]

def get_inverse_frequency():
    # mean of class label proportion of pixels across all images in the dataset
    dist_dict = cfg["CLASS_DISTRIBUTION"]
    mean_dists = np.array(list(dist_dict.values()), dtype=np.float32)
    # inverse class frequency based on mean proportion of pixels
    class_weights = np.sum(mean_dists)/mean_dists
    # return normalized class weights
    return class_weights/np.sum(class_weights)

def get_user_confirmation(prompt):
    answers = {'y': True, 'n': False}
    # ! WARNING: only works in Python 3.8+
    while (response := input(f"[Y/n] {prompt} ").lower()) not in answers:
        print("Invalid input. Please enter 'y' or 'n' (not case sensitive).")
    return answers[response]

def get_user_rating(prompt, choices):
    choices = list(map(str, choices))
    while (response := input(f"{prompt} ").lower()) not in choices:
        print(f"Invalid input. Please enter a value from \n\t{choices}.")
    return response

def download_dataset(data_source, data_destination):
    from gdown import download
    from zipfile import ZipFile
    check_output_path(data_destination)
    output_path = os.path.join(data_destination, 'soiling_dataset.zip')
    print("\nDownloading Valeo Woodscape dataset from Google Drive:")
    try:
        file_path = download(data_source, output_path, quiet=False, use_cookies=False)
        print(f"Success: Downloaded soiling_dataset.zip. Now unzipping into {data_destination}")
        with ZipFile(file_path, 'r') as zipped_dataset:
            zipped_dataset.extractall(data_destination)
    except:
        raise Exception("Something went wrong in downloading the dataset")





# TODO: separate this into more functions - keep test_images and compose them from there
def test_images(dir_path, output_size, read_mode=IO.ImageReadMode.UNCHANGED, folder_name='input images'):
    # TODO: add npy and maybe npz to list below if handled in __getitem__
    allowed_ext = ['.png','.jpg','.jpeg','.pbm','.pgm','.ppm']
    file_list = os.listdir(dir_path)
    test_path = os.path.join(dir_path, file_list[0])
    file_shape = (IO.read_image(test_path, read_mode)).size()
    # ensures that both image dimensions are greater than intended resizing to avoid added noise
    if file_shape[1] < output_size or file_shape[2] < output_size:
        raise Exception(f'One or more image dimensions below threshold of {output_size}')
    print(f'\nTesting {folder_name} for correctness')
    for img in tqdm(file_list):
        # ensures that all images are an acceptable image file type
        img_path = os.path.join(dir_path, img)
        if not list(filter(img.endswith, allowed_ext)):
            ext = os.path.splitext(img)[1]
            raise Exception(f'Encountered in {img} >\n\t.{ext} is not an accepted file type')
        # ensures that all image sizes are the same - again, to avoid added noise from inconsistent interpolation
        if (IO.read_image(img_path, read_mode)).size() != file_shape:
            raise Exception(f'Encountered inconsistent image dimensions >\n\t in {img}')
    return file_shape

def test_image_quality(img_path, mask_path, output_size):
    # TODO: remove hardcoding when I'm lucid enough to deal with tracking down breaking changes
    img_shape = test_images(img_path, output_size, IO.ImageReadMode.RGB, folder_name='rgbImages')
    mask_shape = test_images(mask_path, output_size, folder_name='gtLabels')
    print()
    # ensure that img and mask share core dimensions
    if len(img_shape) != 3:
        raise Exception('img should have 3 channels and not be encoded!')
    if img_shape[1:] != mask_shape[1:]:
        raise Exception('Image and mask shapes should be equivalent!')
    if mask_shape[0] != 1:
        print('WARNING: Masks are not yet encoded. Real time encoding will slow training.')
        return False
    return True


