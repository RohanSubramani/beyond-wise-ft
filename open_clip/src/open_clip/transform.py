from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from PIL import Image


from torchvision.transforms import Normalize, Compose, RandomResizedCrop, ToTensor, Resize, \
    CenterCrop, RandomHorizontalFlip, RandomRotation, ColorJitter

    # InterpolationMode


class ResizeMaxSize(nn.Module):

    def __init__(self, max_size, interpolation=Image.BICUBIC, fn='max', fill=0): # InterpolationMode.BICUBIC
        super().__init__()
        if not isinstance(max_size, int):
            raise TypeError(f"Size should be int. Got {type(max_size)}")
        self.max_size = max_size
        self.interpolation = interpolation
        self.fn = min if fn == 'min' else min
        self.fill = fill

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[:2]
        else:
            width, height = img.size
        scale = self.max_size / float(max(height, width))
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = F.resize(img, new_size, self.interpolation)
            pad_h = self.max_size - new_size[0]
            pad_w = self.max_size - new_size[1]
            img = F.pad(img, padding=[pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2], fill=self.fill)
        return img


def _convert_to_rgb(image):
    return image.convert('RGB')


def image_transform(
        image_size: int,
        is_train: bool,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        resize_longest_max: bool = False,
        fill_color: int = 0,
        data_augmentation_number: int = 1,
):
    mean = mean or (0.48145466, 0.4578275, 0.40821073)  # OpenAI dataset mean
    std = std or (0.26862954, 0.26130258, 0.27577711)  # OpenAI dataset std
    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    normalize = Normalize(mean=mean, std=std)
    if is_train:
        transform = dataAugment(data_augmentation_number,image_size,normalize)
        return transform
    else:
        if resize_longest_max:
            transforms = [
                ResizeMaxSize(image_size, fill=fill_color)
            ]
        else:
            transforms = [
                Resize(image_size, interpolation=Image.BICUBIC), # InterpolationMode.BICUBIC
                CenterCrop(image_size),
            ]
        transforms.extend([
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
        return Compose(transforms)

def dataAugment(num,image_size,normalize):
    if num == 1:
        return Compose([
            RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=Image.BICUBIC), # InterpolationMode.BICUBIC
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
    elif num == 2:
        return Compose([
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
    elif num == 3:
        return Compose([
            RandomHorizontalFlip(),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
    elif num == 4:
        return Compose([
            RandomRotation(180),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
    elif num == 5:
        return Compose([
            RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=Image.BICUBIC), # InterpolationMode.BICUBIC
            RandomHorizontalFlip(),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
    elif num == 6:
        return Compose([
            RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=Image.BICUBIC), # InterpolationMode.BICUBIC
            RandomHorizontalFlip(),
            RandomRotation(180),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
    elif num == 7: 
        return Compose([
            RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=Image.BICUBIC), # InterpolationMode.BICUBIC
            RandomHorizontalFlip(0.1),
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
    elif num == 8: 
        return Compose([
            RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=Image.BICUBIC), # InterpolationMode.BICUBIC
            RandomHorizontalFlip(0.3),
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
    elif num == 9: 
        return Compose([
            RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=Image.BICUBIC), # InterpolationMode.BICUBIC
            RandomHorizontalFlip(),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
    elif num == 10: 
        return Compose([
            RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=Image.BICUBIC), # InterpolationMode.BICUBIC
            RandomHorizontalFlip(0.7),
            ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.7),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
    elif num == 11: 
        return Compose([
            RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=Image.BICUBIC), # InterpolationMode.BICUBIC
            RandomHorizontalFlip(0.9),
            ColorJitter(brightness=0.9, contrast=0.9, saturation=0.9, hue=0.9),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])