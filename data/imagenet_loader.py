#-*- coding:utf-8 -*-
import torch
from torchvision import datasets
from .byol_transform import MultiViewDataInjector, get_transform


class ImageNetLoader():
    def __init__(self, config):
        self.image_dir = config['data']['image_dir']
        self.num_replicas = config['world_size']
        self.rank = config['rank']
        self.distributed = config['distributed']
        self.resize_size = config['data']['resize_size']
        self.data_workers = config['data']['data_workers']
        self.dual_views = config['data']['dual_views']

    def get_loader(self, stage, batch_size):
        dataset = self.get_dataset(stage)
        if self.distributed and stage in ('train', 'ft'):
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=self.num_replicas, rank=self.rank)
        else:
            self.train_sampler = None

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=(self.train_sampler is None and stage not in ('val', 'test')),
            num_workers=self.data_workers,
            pin_memory=True,
            sampler=self.train_sampler,
            drop_last=True
        )
        return data_loader

    def get_dataset(self, stage):
        image_dir = self.image_dir + f"imagenet_{'train' if stage in ('train', 'ft') else 'val'}"
        transform1 = get_transform(stage)
        if self.dual_views:
            transform2 = get_transform(stage, gb_prob=0.1, solarize_prob=0.2)
            transform = MultiViewDataInjector([transform1, transform2])
        else:
            transform = transform1
        dataset = datasets.ImageFolder(image_dir, transform=transform)
        return dataset

    def set_epoch(self, epoch):
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
