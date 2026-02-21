import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from monai.transforms import (
    Compose, LoadImaged, AddChanneld, Orientationd,
    Spacingd, ScaleIntensityRanged, CropForegroundd,
    RandCropByPosNegLabeld, ToTensord
)


class MedicalDataset(Dataset):
    """Dataset for medical image segmentation"""
    
    def __init__(self, data_dir, split='train', img_size=(96, 96, 96)):
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        
        # TODO: Implement actual data loading based on dataset structure
        # This is a placeholder
        self.image_files = []
        self.label_files = []
        
        self.transforms = self._get_transforms()
        
    def _get_transforms(self):
        if self.split == 'train':
            return Compose([
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=self.img_size,
                    pos=1,
                    neg=1,
                    num_samples=2
                ),
                ToTensord(keys=["image", "label"])
            ])
        else:
            return Compose([
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
                ToTensord(keys=["image", "label"])
            ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        data = {
            "image": self.image_files[idx],
            "label": self.label_files[idx]
        }
        
        data = self.transforms(data)
        
        return data["image"], data["label"]


def get_dataloader(config, split='train'):
    """Create dataloader from config"""
    dataset = MedicalDataset(
        data_dir=config['dataset']['data_dir'],
        split=split,
        img_size=config['model']['img_size']
    )
    
    batch_size = config['training']['batch_size'] if split == 'train' else 1
    shuffle = split == 'train'
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )
    
    return loader