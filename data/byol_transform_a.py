#-*- coding:utf-8 -*-
"""
To use this file, you have to install the latest version of albumentations (>0.4.6) from GitHub:
    pip install -U git+https://github.com/albumentations-team/albumentations
"""
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MultiViewDataInjectorA():
    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, sample):
        image = np.array(sample)
        output = [transform(image=image)['image'].unsqueeze(0) for transform in self.transform_list]
        output_cat = torch.cat(output, dim=0)
        return output_cat

class DataInjectorA():
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        image = np.array(sample)
        output = self.transform(image=image)['image']
        return output

def get_transform(stage, gb_prob=1.0, solarize_prob=0.):
    t_list = []
    normalize = A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    if stage in ('train', 'val'):
        t_list = [
            A.RandomResizedCrop(224, 224),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
            A.HueSaturationValue(hue_shift_limit=int(0.1 * 180),
                                 sat_shift_limit=int(0.2 * 255),
                                 val_shift_limit=0, p=0.8),
            A.ToGray(p=0.2),
            A.GaussianBlur(blur_limit=[23, 23], sigma_limit=[0.1, 0.2], p=gb_prob),
            A.Solarize(p=solarize_prob),
            normalize,
            ToTensorV2()
        ]
    elif stage == 'ft':
        t_list = [
            A.RandomResizedCrop(224, 224),
            A.HorizontalFlip(p=0.5),
            normalize,
            ToTensorV2()
        ]
    elif stage == 'test':
        t_list = [
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            normalize,
            ToTensorV2()
        ]
    transform = A.Compose(t_list)
    return transform
