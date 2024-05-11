from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Callable
import torchvision.transforms.v2 as TT
#import torchvision # keeping it like this for better compatibility with the other files when using `handle`
import utils.transforms as augmentations
from torch.nn import Module


''' TO COPY-PASTE:
    tuning_params = {
        "param": {"range": (0,1), "dtype": "float"},
    }
    non_tuning_params = {}
'''

''' # ? NOTE: augmentations not tested here (may be necessary later when testing sequences):
    # TODO: implement these anyway if I'm going to continue using this file
    RandomHorizontalFlip - literally just transposing the array with the center column as axis of symmetry and I know it won't affect much
    RandomAutoContrast - only takes a probability parameter - no tuning needed
    RandomGrayscale - only takes a probability parameter - no tuning needed
    RandomEqualize - only takes a probability parameter - no tuning needed
    ScaleJitter - unless I'm understanding it wrong (probably the case), it's not accomplishing anything except shrinking or blurring the image
'''


class AugmentationFunctionalWrapper(Module):
    def __init__(self, func_handle, **kwargs):
        super().__init__()
        self.forward_func = func_handle
        self.new_params = kwargs

    def forward(self, img):
        return self.forward_func(img, **self.new_params)

'''
# ? NOTE: straight up just started repeating the same thing as AugmentationFunctionalWrapper but with the functional options baked in
class RandomAdjustWrapper(TT.Transform):
    def __init__(self, type, factor, p=0.5):
        
    def __call__(self, img):
        pass

    def __repr__(self):
        return self.__class__.__name__ + '()'
'''


# ? NOTE: not sure about these acting right for the loss functions that I've created
@dataclass(frozen=True)
class RandomRotation:
    name = "random_rotation"
    handle = TT.RandomRotation # torchvision.transforms.v2.RandomRotation
    # only the starting values for the optimization study - may be the max allowed by the object or just what I'm setting manually
    tuning_params = {
        "degrees": {"range": (-20, 20), "dtype": "float"}
    }
    non_tuning_params = {
        "interpolation": TT.InterpolationMode.BILINEAR,
        "fill": 0
    }
    #postprocessor = None

@dataclass(frozen=True)
class RandomCrop:
    name = "random_crop"
    handle = TT.RandomResizedCrop
    tuning_params = {
        "size": {"range": (224, 512), "dtype": "int"}, # indicating that these should be ints so that the image is always square
        "scale": {"range": (0.1, 1.0), "dtype": "float"},
        "ratio": {"range": (0.75, 4/3), "dtype": "float"}
    }
    non_tuning_params = {
        "interpolation": TT.InterpolationMode.BILINEAR,
        "antialias": True
    }
    #postprocesser = TT.Resize((512, 512), TT.InterpolationMode.BILINEAR, antialias=True)


@dataclass(frozen=True)
class AdjustSharpness:
    name = "adjust_sharpness"
    handle = TT.RandomAdjustSharpness
    tuning_params = {
        "sharpness_factor": {"range": (0,2), "dtype": "float"}
    }
    non_tuning_params = {"p": 1}

@dataclass(frozen=True)
class AdjustContrast:
    name = "adjust_contrast"
    handle = AugmentationFunctionalWrapper
    tuning_params = {
        "contrast_factor": {"range": (0,2), "dtype": "float"},
    }
    non_tuning_params = {"func_handle": TT.functional.adjust_contrast}


@dataclass(frozen=True)
class AdjustHue:
    name = "adjust_hue"
    handle = AugmentationFunctionalWrapper
    tuning_params = {
        "hue_factor": {"range": (0,0.5), "dtype": "float"},
    }
    non_tuning_params = {"func_handle": TT.functional.adjust_hue}

@dataclass(frozen=True)
class AdjustSaturation:
    name = "adjust_saturation"
    handle = AugmentationFunctionalWrapper
    tuning_params = {
        "saturation_factor": {"range": (0,2), "dtype": "float"},
    }
    non_tuning_params = {"func_handle": TT.functional.adjust_saturation}

@dataclass(frozen=True)
class AdjustBrightness:
    name = "adjust_brightness"
    handle = AugmentationFunctionalWrapper
    tuning_params = {
        "saturation_factor": {"range": (0,2), "dtype": "float"},
    }
    non_tuning_params = {"func_handle": TT.functional.adjust_saturation}
    # is_functional = True # ~ IDEA for later - if is_functional, more augmentation sampling is encouraged

# Power Law Transform
@dataclass(frozen=True)
class AdjustGamma:
    name = "adjust_gamma"
    handle = AugmentationFunctionalWrapper
    tuning_params = {
        "gamma": {"range": (0.25,2), "dtype": "float"},
        "gain": {"range": (0.8,1.2), "dtype": "float"}
    }
    non_tuning_params = {"func_handle": TT.functional.adjust_gamma}

@dataclass(frozen=True)
class ColorJitter:
    name = "color_jitter"
    handle = TT.ColorJitter
    tuning_params = {
        "brightness": {"range": (0,2), "dtype": "float"},
        "contrast": {"range": (0,2), "dtype": "float"},
        "saturation": {"range": (0,2), "dtype": "float"},
        "hue": {"range": (0,0.5), "dtype": "float"},
    }
    non_tuning_params = {}


@dataclass(frozen=True)
class RandomBlur:
    name = "random_blur"
    handle = TT.GaussianBlur
    tuning_params = {
        "kernel_size": {"range": (3,7), "dtype": "int"},
        "sigma": {"range": (0.1, 2.0), "dtype": "float"}
    }
    non_tuning_params = {}


@dataclass(frozen=True)
class ElasticTransform:
    name = "elastic_transform"
    handle = TT.ElasticTransform
    tuning_params = {
        "alpha": {"range": (25,75), "dtype": "float"},
        "sigma": {"range": (2.5,7.5), "dtype": "float"}
    }
    non_tuning_params = {
        "interpolation": TT.InterpolationMode.BILINEAR,
        "fill": 0
    }

'''# ? NOTE: questionable levels of augmentation - seems too aggressive
@dataclass(frozen=True)
class RandomAffine:
    name = "random_affine"
    handle = TT.RandomAffine
    tuning_params = {
        "degrees": {"range": (-10,10), "dtype": "float"},       # set to 0 to deactivate rotations
        #"translate": {"range": (0.01, 0.1), "dtype": "float"},    # remaining 3 params are optional - won't be performed by default
        #"scale": {"range": (2.5,7.5), "dtype": "float"},
        "shear": {"range": (-10,10), "dtype": "float"}
    }
    non_tuning_params = {
        "interpolation": TT.InterpolationMode.BILINEAR,
        "fill": 0
    }'''

# ? NOTE: basically seems to do the same thing as RandomAffine but without setting each parameter
@dataclass(frozen=True)
class RandomPerspective:
    name = "perspective_transform"
    handle = TT.RandomPerspective
    tuning_params = {
        "distortion": {"range": (0.05,0.3), "dtype": "float"}
    }
    non_tuning_params = {
        "p": 1,
        "interpolation": TT.InterpolationMode.BILINEAR,
        "fill": 0
    }

@dataclass(frozen=True)
class RandomPosterize:
    name = "random_posterize"
    handle = TT.RandomPosterize
    tuning_params = {
        "bits": {"range": (3,7), "dtype": "int"}
    }
    non_tuning_params = {"p": 1}

@dataclass(frozen=True)
class RandomSolarize:
    name = "random_solarize"
    handle = TT.RandomSolarize
    tuning_params = {
        "threshold": {"range": (200,255), "dtype": "int"}
    }
    non_tuning_params = {"p": 1}

# ! this threw an import error for no reason that I can tell so I'm taking it out for now - TODO: replace with the kornia implementation later
"""# ? NOTE: possibly not GPU-friendly: "If the input is a torch.Tensor, it is expected to be of dtype uint8, on CPU, and have […, 3 or 1, H, W] shape"
@dataclass(frozen=True)
class RandomJPEG:
    name = "random_jpeg"
    handle = TT.JPEG
    tuning_params = {
        "quality": {"range": (50,90), "dtype": int}
    }"""


'''
https://pytorch.org/vision/main/transforms.html
To look into:
- RandomAffine
- RandomPerspective
- RandomInvert - seems like too much of a change
- RandomPosterize
- RandomSolarize
- LinearTransformation
    - might skip this one for now - intended to do ZCA whitening transformations, but it seems more like the transformation matrix is computed on sample-wise basis
    - may look into it again if I wanted to get some sort of mean across the training dataset
- Normalize - NOTE: use Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- RandomErasing - maybe skip for now: https://pytorch.org/vision/main/generated/torchvision.transforms.v2.RandomErasing.html#torchvision.transforms.v2.RandomErasing
- JPEG -
'''


# TODO: define dataclass that wraps transforms with their classification of augmentation type, allowed parameter range, etc - making a nested dataclass CLS
# TODO: define transform manager class that aggregates transforms, possibly adopting one of the auto-augmentation policies I wrote about
    # also look into integrating kornia's Augmentation Dispatchers if possible: https://kornia.readthedocs.io/en/latest/augmentation.container.html