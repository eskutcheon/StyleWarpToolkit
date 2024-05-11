import sys
import time
import os
from typing import Union, Type, List, Tuple
import torch
import numpy as np
import torchvision.transforms as TTv1
import torchvision.transforms.v2 as TTv2
import torchvision.io as IO
#import torchvision.transforms.functional as F
from torch.nn.functional import mse_loss
from torchvision import tv_tensors
from torch.utils.data import Dataset, DataLoader
# local util files
import utils.utils as util
import utils.img_utils as img_util
import utils.plot_utils as plot_util
from utils.transforms import AugmentationFunctions



def print_tensor_details(img, mask, msg=""):
    util.get_all_debug(img, f"Image {msg}")
    util.get_all_debug(mask, f"Mask {msg}")


class SegmentationDataset(Dataset):
    def __init__(self, path, transforms, out_size, TT_version: object):
        super().__init__()
        self.TT_version = TT_version
        self.OUTPUT_SIZE = (out_size, out_size)
        self.transforms = transforms
        # TODO: add the totvtensor idea to the beginning of read_funcs if TT_version == TTv2 - check if it could still output a dictionary with the same structure
            # AND whether it could infer the proper interpolation mode from whether it was tv_tensor.Image or tv_tensor.Mask
        self.read_funcs = {
            'img' : [
                util.MultiArgLambda(IO.read_image, mode=IO.ImageReadMode.RGB),
                TT_version.Resize(self.OUTPUT_SIZE, TTv1.InterpolationMode.BILINEAR, antialias=True)],
            'mask' : [
                util.MultiArgLambda(IO.read_image, mode=IO.ImageReadMode.UNCHANGED),
                TT_version.Resize(self.OUTPUT_SIZE, TTv1.InterpolationMode.NEAREST)],
        }
        if TT_version == TTv2:
            self.read_funcs["img"].insert(1, tv_tensors.Image)
            self.read_funcs["mask"].insert(1, tv_tensors.Mask)
        for key, compositions in self.read_funcs.items():
            self.read_funcs[key] = TT_version.Compose(compositions)
        self.preprocessing = TT_version.Compose([
            # alternatively: util.MultiArgLambda(TTv1.functional.convert_image_dtype, dtype=torch.float32)
            # ? NOTE: will work for tv_tensor.Image (converting to float and dividing by 255) but not tv_tensor.Mask - preserves uint8 dtype
            # ? NOTE: pretty sure scale is an inherited parameter of the Transform base class, not ToDtype; ConvertDtype does the same thing but scales implicitly
            TTv2.ToDtype(torch.float32, scale=True),
        ])
        # ? NOTE: Should add ToPureTensor at the end of any post-processing pipes to change from TVTensor to pure tensor
        img_folder = os.path.join(path, 'rgbImages')
        mask_folder = os.path.join(path, 'gtLabels')
        # NOTE: for future datasets, note that img and mask path names must be identical to assure proper sorting
        filenames = sorted(os.listdir(img_folder))[37:45]
        self.img_paths = [os.path.join(img_folder, name) for name in filenames]
        self.mask_paths = [os.path.join(mask_folder, name) for name in filenames]
        self.num_images = len(self.img_paths)
        assert len(self.img_paths) == len(self.mask_paths), "folder of images is not the same length as folder of masks with lengths:\n" + \
            f"{img_folder} - length {len(self.img_paths)} \n{mask_folder} - length {len(self.mask_paths)}"

    # following 2 functions have to be overridden for the DataLoader
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        sample = {
            'img' : self.read_funcs['img'](img_path),
            'mask' :  self.read_funcs['mask'](mask_path),
            #'img_path': img_path,
            #'mask_path': mask_path
        }
        img, mask = sample["img"].clone(), sample["mask"].clone()
        print_tensor_details(img, mask, "before preprocessing")
        # need to be rewritten for v2 - shouldn't need 2 commands
        #sample['img'] = self.preprocessing(sample['img'])
        #sample['mask'] = self.preprocessing(sample['mask'])
        sample = self.preprocessing(sample)
        print_tensor_details(sample["img"], sample["mask"], "after preprocessing; before transforms")
        torch.manual_seed(123)
        sample = self.transforms(sample)
        print_tensor_details(sample["img"], sample["mask"], "after transforms")
        sample.update({"img_original": img, "mask_original": mask})
        return sample

    def __len__(self):
        return self.num_images


# NOTE: eventually just want to start passing deep copies of the global_settings object rather than calling from all over
def get_dataloader(path: str, device: torch.device, batch_size: int, out_size: int, transforms: Union[TTv1.Compose, TTv2.Compose], TT_version):
    # TODO: update for both this and the main get_dataloader function - pass a seed to set the generator with
    shuffle = False
    dataset = SegmentationDataset(path, transforms, out_size, TT_version)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=torch.Generator(device=device)) #, num_workers=6)
    return loader


class AugmentationWrapper:
    def __init__(self, func_names):
        # add to the list below as more augmentations are implemented above
        self.augmentations = []
        for func in dir(AugmentationFunctions):
            if callable(getattr(AugmentationFunctions, func)) and not func.startswith("__") and func in func_names:
                self.augmentations.append(getattr(AugmentationFunctions, func))

    def __call__(self, sample):
        for augmentation in self.augmentations:
            sample['img'], sample['mask'] = augmentation(sample['img'], sample['mask'])
        return sample


def psnr_loss(mse, max_val=1.0):
    if mse == 0:
        return float('inf')
    return float(20*torch.log10(torch.Tensor([max_val])) - 10*torch.log10(torch.Tensor([mse])))

class TransformCompPairs():
    @staticmethod
    def random_rotation():
        return (
            TTv1.Compose([AugmentationWrapper(["random_rotation"])]),
            # even using InterpolationMode.BILINEAR for making the Compose object, it defaults to InterpolationMode.NEAREST for tv_tensor.Mask and works just like the old one
            TTv2.Compose([TTv2.RandomRotation((4,4), interpolation=TTv1.InterpolationMode.BILINEAR, fill=0)])
        )

    @staticmethod
    def random_crop():
        # FIXME: finish later - can't think right now, but I need to make some more tests on the masks running through these transforms that do interpolation - was just going by mask equivalence before
        # ? NOTE: will need that later: https://pytorch.org/vision/main/generated/torchvision.transforms.v2.RandomResizedCrop.html#torchvision.transforms.v2.RandomResizedCrop
        return (
            TTv1.Compose([AugmentationWrapper(["random_crop"])]),
            TTv2.Compose([TTv2.RandomResizedCrop((512,512), scale = (0.6,0.6), ratio = (0.5,0.5), interpolation=TTv1.InterpolationMode.BILINEAR, antialias=True)])
        )

    @staticmethod
    def random_blur():
        return (
            TTv1.Compose([AugmentationWrapper(["random_blur"])]),
            TTv2.Compose([TTv2.GaussianBlur(kernel_size=(5,5), sigma=(1,1))])
        )

    @staticmethod
    def random_grayscale():
        return (
            TTv1.Compose([AugmentationWrapper(["random_grayscale"])]),
            TTv2.Compose([TTv2.RandomGrayscale(p=1)])
        )

    @staticmethod
    def color_jitter():
        return (
            TTv1.Compose([AugmentationWrapper(["color_jitter"])]),
            # ? NOTE: v2 changed default parameters to None to do nothing - None is equivalent to what 0 used to do
                # both versions: first 3 parameters use [max(0, 1 - x), 1 + x] when given a float x and range (min,max) when given a tuple
                # values chosen uniformly from these intervals - x=0 equates to interval [1,1]
                # exception in both versions is hue - given float x, range is [-x,x]; given tuple, range is explicit (min,max) with below restriction
                    # (use mathover VS code extension to view rendered)
                    # Math: hue \in [-0.5, min] \cup [min, max] \cup [max, 0.5]
            TTv2.Compose([TTv2.ColorJitter(brightness=(0.5, 0.5), contrast=None, saturation=(0.6,0.6), hue=None)])
        )

class NewTransformFuncs(object):
    # wrap the transform callables in a new function prob with hardcoded parameters - may include extra processing
    @staticmethod
    def scale_jitter():
        # ? NOTE: may be too volatile (and too similar to random_blur) to keep
        # refer to this paper to figure out how to use ScaleJitter properly: https://arxiv.org/pdf/2012.07177.pdf
        return TTv2.Compose([TTv2.ScaleJitter(target_size=(224,224), scale_range=(1.9,1.99), interpolation=TTv2.InterpolationMode.BILINEAR, antialias=True),
                            TTv2.Resize(size=(512,512), interpolation=TTv2.InterpolationMode.BILINEAR, antialias=True),
                            TTv2.Lambda(lambda x: 255*x),
                            TTv2.ToDtype(torch.uint8)])

    @staticmethod
    def elastic_transform():
        # ? NOTE: haven't played with the parameters much - using the default and it looks decent
        return TTv2.Compose([TTv2.ElasticTransform(alpha=50, sigma=5, interpolation=TTv2.InterpolationMode.BILINEAR)])

    @staticmethod
    def perspective_transform():
        # possibly useful since it smushes the input more in the center region
        # distortion worsens as distortion_scale is increased
        return TTv2.Compose([TTv2.RandomPerspective(distortion_scale=0.1, p=1, interpolation=TTv2.InterpolationMode.BILINEAR)])

    @staticmethod
    def random_equalize():
        return TTv2.Compose([TTv2.RandomEqualize(p=1)])

    @staticmethod
    def adjust_sharpness():
        # if included in the pipeline, it may need a wrapper function just to select a range to select sharpness factor
        return TTv2.Compose([TTv2.RandomAdjustSharpness(sharpness_factor = np.random.uniform(0.5,2), p=1)])

    @staticmethod
    def random_solarize():
        # possibly antithetical to learning in some cases - some images look like they have new soiled regions when the light sky turns dark
        # threshold - all pixels equal or above this value are inverted - will probably need some tuning
        return TTv2.Compose([TTv2.RandomSolarize(threshold=0.5, p=1)])

    @staticmethod
    def random_posterize():
        # bit - number of bits to keep for each channel (0-8)
        return TTv2.Compose([TTv2.RandomPosterize(bits=5, p=1)])

    @staticmethod
    def random_inversion():
        # May be too extreme of a transformation to keep - update: yeah, almost definitely toss it
        return TTv2.Compose([TTv2.RandomInvert(p=1)])

    @staticmethod
    def scale_shift():
        # also affects masks unfortunately - will need to be wrapped in a function to apply it to images only
        # given the nature of the Transforms base classes, I might just need to add decorators to some overridden __call__ functions
            # so i could do something like @skipif(lambda x: isinstance(x, torchvision.tv_sensors.Mask)) where it skips the transform and jumps to returning the input
        return TTv2.Compose([TTv2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False),
                            TTv2.Lambda(torch.nn.functional.sigmoid)])


    # also look into using linear transformation for different transforms like their suggested whitening: https://pytorch.org/vision/2.0/generated/torchvision.transforms.v2.LinearTransformation.html#torchvision.transforms.v2.LinearTransformation

# Maybe move things around into this to document and atomize it all better
class TransformTestingFramework(object):
    def __init__(self):
        pass


# TODO: considering wrapping this in something else to call it within a Jupyter notebook to publish output in a single document
    # may want more variance in the image samples, which could probably be done by setting the seed and setting and shuffle=True
def test_transform_consistency(get_transform_func: callable, test_label: str, constants: dict):
    test_tranforms_v1, test_tranforms_v2 = get_transform_func()
    # creating the dataloaders here rather than passing them since timing should be a separate test
    v1_loader = get_dataloader(**constants, transforms = test_tranforms_v1, TT_version=TTv1)
    v2_loader = get_dataloader(**constants, transforms = test_tranforms_v2, TT_version=TTv2)
    print(f"TEST '{test_label}':")
    for idx, (l1_batch, l2_batch) in enumerate(zip(v1_loader, v2_loader)):
        plot_util.plot_images(l1_batch['img'].squeeze(0), l2_batch['img'].squeeze(0))
        plot_util.plot_images(l1_batch['mask'].squeeze(0), l2_batch['mask'].squeeze(0))
        print(f"image {idx+1}")
        print(f"img {idx} same across loaders: {torch.equal(l1_batch['img'], l2_batch['img'])}")
        print(f"mask {idx} same across loaders: {torch.equal(l1_batch['mask'], l2_batch['mask'])}")
        # TODO: maybe should do some sort of perceptual hashing instead, since these differences aren't scaling well enough to see the issue
        # really need to return to the heatmap idea to see why
        #plot_util.plot_image_diff_superpixels(l1_batch['img'], l2_batch['img'], f"absolute difference of img {idx + 1}")
        #plot_util.plot_image_diff_superpixels(l1_batch['mask'], l2_batch['mask'], f"absolute difference of mask {idx + 1}")
        img_diff = torch.abs(l1_batch['img'] - l2_batch['img']).squeeze(0)
        mask_diff = torch.abs(l1_batch['mask'] - l2_batch['mask']).squeeze(0)
        plot_util.plot_image_diff_heatmap(img_diff, pool_size=(5,5))
        plot_util.plot_image_diff_heatmap(mask_diff, pool_size=(5,5))
        diff_img_mse = mse_loss(l1_batch['img'].to(dtype=torch.float32), l2_batch['img'].to(dtype=torch.float32))
        diff_mask_mse = mse_loss(l1_batch['mask'].to(dtype=torch.float32), l2_batch['mask'].to(dtype=torch.float32))
        print("MSE Losses:")
        print(f"img: {diff_img_mse},\nmask: {diff_mask_mse}")
        print("PSNR values:")
        print(f"img: {psnr_loss(diff_img_mse)},\nmask: {psnr_loss(diff_mask_mse)}")
        plot_util.plot_difference_histogram(img_diff, num_bins=20)
        plot_util.plot_difference_histogram(mask_diff, num_bins=20)
        #time.sleep(1)
        break


def test_new_transforms(get_transform_func: callable, test_label: str, constants: dict):
    test_tranforms = get_transform_func()
    # creating the dataloaders here rather than passing them since timing should be a separate test
    loader = get_dataloader(**constants, transforms = test_tranforms, TT_version=TTv2)
    print(f"TEST '{test_label}':")
    trial_name = test_label.split(" - ")[1]
    for idx, batch in enumerate(loader):
        plot_util.plot_images(batch['img_original'].squeeze(0), batch['img'].squeeze(0), title=f"{trial_name}\nbase image: original (left), transformed (right)")
        plot_util.plot_images(batch['mask_original'].squeeze(0), batch['mask'].squeeze(0), title=f"{trial_name}\nmask: original (left), transformed (right)")
        img_diff = torch.abs(batch['img'] - batch['img_original']).squeeze(0)
        mask_diff = torch.abs(batch['mask'] - batch['mask_original']).squeeze(0)
        plot_util.plot_image_diff_heatmap(img_diff, pool_size=(5,5))
        plot_util.plot_image_diff_heatmap(mask_diff, pool_size=(5,5))
        break


if __name__ == "__main__":
    reusable_args = {
        "path" : r"C:\Users\Jacob\Documents\MSU HPC2 Work\Camera Occlusion\TransUNET_Segmentation\data\soiling_dataset\train",
        "device" : "cpu",
        "batch_size" : 1,
        "out_size" : 512,
    }
    # Unit testing and (sort of) regression testing of replacing old functions with new ones
    # ? NOTE: timing will probably just be a separate script once I separate out the code above a bit more
    #test_transform_consistency(TransformCompPairs.random_rotation, "random rotation consistency test")
    #test_transform_consistency(TransformCompPairs.random_crop, "random resized crop consistency test")
    #test_transform_consistency(TransformCompPairs.random_blur, "random Gaussian blur consistency test")
    #test_transform_consistency(TransformCompPairs.random_grayscale, "random grayscaling consistency test")
    #test_transform_consistency(TransformCompPairs.color_jitter, "random color jittering consistency test (parameters tested 1 by 1)")
    #######################################################################################################
    test_new_transforms(NewTransformFuncs.scale_jitter, "new transform test - scale jitter")
    #test_new_transforms(NewTransformFuncs.elastic_transform, "new transform test - elastic transform")
    #test_new_transforms(NewTransformFuncs.perspective_transform, "new transform test - random perspective transform")
    #test_new_transforms(NewTransformFuncs.random_equalize, "new transform test - random global histogram equalization")
    #test_new_transforms(NewTransformFuncs.adjust_sharpness, "new transform test - random sharpness adjustment")
    #test_new_transforms(NewTransformFuncs.random_solarize, "new transform test - random solarization")
    #test_new_transforms(NewTransformFuncs.random_posterize, "new transform test - random posterization")
    #test_new_transforms(NewTransformFuncs.random_inversion, "new transform test - random inversion")
    #test_new_transforms(NewTransformFuncs.scale_shift, "new transform test - using normalization for shift-scale transform")
