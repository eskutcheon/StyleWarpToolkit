import torch
import torchvision.transforms.v2 as TT
from torchvision import tv_tensors
import torchvision.io as IO
#import torch.nn.functional as F
from torchvision.utils import make_grid
import json, os, sys
from typing import Dict, List, Tuple, Union, Callable
import numpy as np
from tqdm import tqdm
# to allow the relative import from the parent directory
sys.path.append(os.path.join(os.getcwd(), ".."))
from utils.utils import MultiArgLambda, tensor_to_cuda
#import utils.plot_utils as plot_util


def remove_batch_dim(img):
    if len(img.shape) == 4:
        img = img.squeeze(0)


class AugmentationTester(object):
    def __init__(self, input_dir, using_ipynb=False, device="cpu"):
        self.input_dir = input_dir
        self.device = device
        self.input_paths = [os.path.join(input_dir, p) for p in os.listdir(input_dir)]
        self.read_pipeline = TT.Compose([
            MultiArgLambda(IO.read_image, mode=IO.ImageReadMode.RGB),
            tv_tensors.Image,
            TT.Resize((512, 512), TT.InterpolationMode.BILINEAR, antialias=True),
            TT.ToDtype(torch.float32, scale=True)]
        )
        if self.device == "cuda":
            self.read_pipeline.transforms.insert(1, TT.Lambda(tensor_to_cuda))
        self.grid_postprocessing = TT.Compose([
            TT.Lambda(lambda x: make_grid(x, padding=8, normalize=True, scale_each=False).squeeze(0))]
        )
        # convert image to PIL at the end if the object is used in a Jupyter notebook so it can be displayed
        if using_ipynb:
            self.grid_postprocessing.transforms.append(TT.ToPILImage())
        # ? NOTE: categorical variables must have a "range" key with values of all possible variable choices
        self.sampling_funcs = {"float": np.random.uniform, "int": np.random.randint, "categorical": np.random.choice}

    def read_image_generator(self):
        # want to write this as a generator to call next() in the test files and interate over self.input_paths
        for p in self.input_paths:
            yield self.read_pipeline(p)

    # * Would prefer to move the two functions below to a new class or something for extensibility in using the dataclasses or not
    def _sample_params(self, var_params, fixed_params):
        # TODO: write a function to resample the parameters of an augmentation when called
            # added complexity: resampling the parameters of only one augmentation within a composition 
        params = {}
        for key, attr in var_params.items():
            if attr["dtype"] == "categorical":
                attr["range"] = [attr["range"]]
            # TODO: if I'm keeping the dataclass representation, I need to add something that indicates whether their parameters accept a range or not
                # if so, give it the range directly, else, sample a scalar in that range with numpy
            params[key] = self.sampling_funcs[attr["dtype"]](*attr["range"])
        params.update(fixed_params)
        return params

    def get_augmentation(self, aug_wrapper):
        """ Apply an augmentation to an image with parameters given by params """
        params = self._sample_params(aug_wrapper.tuning_params, aug_wrapper.non_tuning_params)
        return aug_wrapper.handle(**params)

    def set_composition(self, aug_wrapper_list: list):
        # TODO: write some logic here to handle the case when a single augmentation wrapper is given
        to_compose = []
        for wrapper in aug_wrapper_list:
            to_compose.append(self.get_augmentation(wrapper))
        self.augmentations = TT.Compose(to_compose)

    def get_comparison_grid(self, img, img_aug):
        remove_batch_dim(img)
        remove_batch_dim(img_aug)
        img_batch = torch.stack((img, img_aug), dim=0)
        return self.grid_postprocessing(img_batch)
    
