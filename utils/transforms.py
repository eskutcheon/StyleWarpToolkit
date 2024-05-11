# This whole file can be replaced with an explicit Compose call in train_network once torchvision.transforms.v2 leaves Beta
import numpy as np
import torch
# NOTE: use only scriptable transformations, i.e. that work with torch.Tensor
import torchvision.transforms.v2 as TT
#import torchvision.transforms.functional as F
import torch.nn.functional as F

# might want the class below to subclass torchvision.transforms.Transform


class AugmentationFunctions:
    @staticmethod
    def random_rotation(img, mask, hardcoded_test=None):
        #torch.manual_seed(0)
        #angle = np.random.uniform(-5, 5) # random angle in degrees in range [-20,20]
        angle = 4
        img = TT.functional.rotate(img, angle, TT.InterpolationMode.BILINEAR, fill=0)
        mask = TT.functional.rotate(mask, angle, TT.InterpolationMode.NEAREST, fill=0)
        return img, mask

    @staticmethod
    def random_hflip(img, mask):
        img_hflip = TT.RandomHorizontalFlip(p=0.5)
        return img_hflip(img), img_hflip(mask)

    @staticmethod
    # TODO: If keeping a single function for jittering all 4 ways, maybe randomly create a Compose of up to 2-3 of them since you can nest Composes infinitely as far as I know
    def color_jitter(img, mask):
        # may need to keep a function like this even in v2 so that it doesn't always affect all parameters or just have multiple custom parameters using https://pytorch.org/vision/2.0/transforms.html#functional-transforms
        # they have an adjust function for brightness, contrast, gamma, hue, saturation, sharpness, and more
        '''param = [0. for _ in range(4)]
        for _ in range(2):
            # It's fine if randint selects the same index twice. Only one thing will be changed then
            # beta distribution in range [0,1], but a=2 and b=3 gives a unimodal shape with peak around 0.4
            param[np.random.randint(0,4)] = np.random.beta(a=2, b=2) # peak ~= (a-1)/(a+b-2)
        # all parameters except hue are chosen uniformly from [max(0,1-val), 1+val] while hue is chosen
            # uniformly from [-hue, hue] with restriction 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5
        if(param[3] != 0):
            param[3] *= 0.5'''
        # easier to debug than unpacking a list of positional arguments - just going to comment out tuples and add a different one for different tests
        kwargs = {
            "brightness": (0.5, 0.5),
            "contrast": 0, #(0.75, 0.75),
            "saturation": (0.6,0.6),
            "hue": 0
        }
        # randomly changes brightness, contrast, saturation, or hue. Doing 2 at most
        img_recolor = TT.ColorJitter(**kwargs)
        img = img_recolor(img)
        return img, mask


    @staticmethod
    def random_crop(img, mask):
        #scale=(0.25, 1.0)
        scale = (0.6,0.6)
        #ratio=(3./4., 4./3.)
        ratio = (0.5,0.5)
        # (81, 34, 326, 426)
        torch.manual_seed(123)
        i, j, h, w = TT.RandomResizedCrop.get_params(img, scale=scale, ratio=ratio)
        print(f"i,j,h,w = {i,j,h,w}")
        return TT.functional.resized_crop(img, i, j, h, w, img.shape[-2:], TT.InterpolationMode.BILINEAR, antialias=True), \
            TT.functional.resized_crop(mask, i, j, h, w, mask.shape[-2:], TT.InterpolationMode.NEAREST)

    @staticmethod
    def random_blur(img, mask):
        # TODO: randomize kernel size from a small selection of sizes later
        # NOTE: sigma chosen from (0.1, 2.0) - I swear I have some details written on default kernels somewhere on the SegmentationImprovement repo
        img_blur = TT.GaussianBlur(kernel_size=(5,5), sigma=(1,1))
        img = img_blur(img)
        return img, mask

    @staticmethod
    def random_grayscale(img, mask):
        # NOTE: number of output channels (1 or 3) is the same as the input
        img_grayscale = TT.RandomGrayscale(p=1)
        return img_grayscale(img), mask




# will combine this class with the above when I'm at a point where they can be extensively tested
    # i.e., regression testing with methods in RandomAugmentation and individual unit testing
    # intent is to actually move all of this to RandomAugmentation and have everything in Compose objects
class AugmentationFunctions_v2(object):
    def __init__(self):
        pass






# Have to write the augmentations below as a class w/ __call__ method to be used with
    # torchvision.transforms.Compose - mostly needed for cases where args must be different for img and mask
class RandomAugmentation:
    def __init__(self, max_augment_count):
        # add to the list below as more augmentations are implemented above
        self.augmentations = []
        for func in dir(AugmentationFunctions):
            if callable(getattr(AugmentationFunctions, func)) and not func.startswith("__"):
                self.augmentations.append(getattr(AugmentationFunctions, func))
        self.max_augment_count = min(max_augment_count, len(self.augmentations))
        # NOTE: idea - sort the the list first; use func as dict keys where values are probabilities

    def __call__(self, sample):
        # can't remember if sample contains lists of images and masks or a single img-mask pair
        augmentation_count = np.random.randint(0, self.max_augment_count, dtype=int)
        selected_augmentations = np.random.choice(self.augmentations, size=augmentation_count, replace=False)
        # TODO: specify probabilities used as arguments in the line above to use some transforms more sparingly
            # NOTE: consider that some probabilities could probably be conditional on others
                # so that some combinations don't happen often (ones with high information loss)
        # print(selected_augmentations)
        for augmentation in selected_augmentations:
            # essentially giving these functions a 40% chance of running at all
            if augmentation in [AugmentationFunctions.random_crop, AugmentationFunctions.color_jitter,
                                AugmentationFunctions.random_rotation, AugmentationFunctions.random_blur] and np.random.uniform(0, 1) <= 0.6:
                continue
            sample['img'], sample['mask'] = augmentation(sample['img'], sample['mask'])
