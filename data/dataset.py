import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from scipy import ndimage
from monai.transforms import (
    Compose, LoadImaged, AddChanneld, Orientationd,
    Spacingd, ScaleIntensityRanged, CropForegroundd,
    RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
    RandScaleIntensityd, RandShiftIntensityd, ToTensord
)


class SynapseDataset(Dataset):
    """Synapse Multi-Organ CT Dataset"""
    
    def __init__(self, data_dir, split='train', img_size=(96, 96, 96)):
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        
        # Load file lists
        self.data_list = self._load_file_list()
        
        if len(self.data_list) == 0:
            raise ValueError(f"No data found in {data_dir} for split {split}")
        
        print(f"Loaded {len(self.data_list)} cases for {split}")
        
        # Setup transforms
        self.transforms = self._get_transforms()
        
    def _load_file_list(self):
        """Load image and label file pairs"""
        img_dir = os.path.join(self.data_dir, 'imagesTr' if self.split in ['train', 'val'] else 'imagesTs')
        label_dir = os.path.join(self.data_dir, 'labelsTr' if self.split in ['train', 'val'] else 'labelsTs')
        
        if not os.path.exists(img_dir):
            print(f"Warning: {img_dir} does not exist")
            return []
        
        # Get all image files
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.nii.gz')])
        
        data_list = []
        for img_file in img_files:
            img_path = os.path.join(img_dir, img_file)
            
            # Corresponding label file
            label_file = img_file.replace('img', 'label') if 'img' in img_file else img_file
            label_path = os.path.join(label_dir, label_file)
            
            # Check if label exists (for test set, labels might not exist)
            if os.path.exists(label_path) or self.split == 'test':
                data_list.append({
                    'image': img_path,
                    'label': label_path if os.path.exists(label_path) else None,
                    'name': img_file.replace('.nii.gz', '')
                })
        
        # Split data (70% train, 15% val, 15% test for Synapse)
        if self.split == 'train':
            data_list = data_list[:int(0.7 * len(data_list))]
        elif self.split == 'val':
            data_list = data_list[int(0.7 * len(data_list)):int(0.85 * len(data_list))]
        elif self.split == 'test':
            data_list = data_list[int(0.85 * len(data_list)):]
        
        return data_list
    
    def _get_transforms(self):
        """Get data transforms"""
        if self.split == 'train':
            return Compose([
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest")
                ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-175,
                    a_max=250,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=self.img_size,
                    pos=1,
                    neg=1,
                    num_samples=2
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
                RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
                RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
                RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
                ToTensord(keys=["image", "label"])
            ])
        else:
            return Compose([
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest")
                ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-175,
                    a_max=250,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True
                ),
                ToTensord(keys=["image", "label"])
            ])
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data_dict = self.data_list[idx]
        
        # Apply transforms
        try:
            transformed = self.transforms(data_dict)
            image = transformed["image"]
            label = transformed["label"] if "label" in transformed else None
            
            # Handle case where label might not exist (test set)
            if label is None:
                label = torch.zeros_like(image[0]).long()
            else:
                label = label.long()
            
            return image, label.squeeze(0)  # Remove channel dim from label
            
        except Exception as e:
            print(f"Error loading {data_dict['name']}: {e}")
            # Return dummy data on error
            return torch.zeros(1, *self.img_size), torch.zeros(self.img_size).long()


class ACDCDataset(Dataset):
    """ACDC Cardiac MRI Dataset"""
    
    def __init__(self, data_dir, split='train', img_size=(96, 96, 96)):
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        
        self.data_list = self._load_file_list()
        
        if len(self.data_list) == 0:
            raise ValueError(f"No data found in {data_dir} for split {split}")
        
        print(f"Loaded {len(self.data_list)} cases for {split}")
        self.transforms = self._get_transforms()
    
    def _load_file_list(self):
        """Load ACDC file pairs"""
        data_dir = os.path.join(self.data_dir, 'training' if self.split != 'test' else 'testing')
        
        if not os.path.exists(data_dir):
            print(f"Warning: {data_dir} does not exist")
            return []
        
        data_list = []
        
        # ACDC structure: patient folders with frame files
        for patient_dir in sorted(os.listdir(data_dir)):
            patient_path = os.path.join(data_dir, patient_dir)
            if not os.path.isdir(patient_path):
                continue
            
            # Get ED and ES frames
            for frame_type in ['ED', 'ES']:
                img_file = f"{patient_dir}_frame{frame_type[0]}{frame_type[1]}.nii.gz"
                label_file = f"{patient_dir}_frame{frame_type[0]}{frame_type[1]}_gt.nii.gz"
                
                img_path = os.path.join(patient_path, img_file)
                label_path = os.path.join(patient_path, label_file)
                
                if os.path.exists(img_path):
                    data_list.append({
                        'image': img_path,
                        'label': label_path if os.path.exists(label_path) else None,
                        'name': f"{patient_dir}_{frame_type}"
                    })
        
        # 80-10-10 split
        if self.split == 'train':
            data_list = data_list[:int(0.8 * len(data_list))]
        elif self.split == 'val':
            data_list = data_list[int(0.8 * len(data_list)):int(0.9 * len(data_list))]
        else:
            data_list = data_list[int(0.9 * len(data_list)):]
        
        return data_list
    
    def _get_transforms(self):
        """ACDC-specific transforms (MRI, different intensity range)"""
        if self.split == 'train':
            return Compose([
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest")
                ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=0,
                    a_max=1000,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True
                ),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=self.img_size,
                    pos=1,
                    neg=1,
                    num_samples=2
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1]),
                RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
                ToTensord(keys=["image", "label"])
            ])
        else:
            return Compose([
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=0,
                    a_max=1000,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True
                ),
                ToTensord(keys=["image", "label"])
            ])
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data_dict = self.data_list[idx]
        
        try:
            transformed = self.transforms(data_dict)
            image = transformed["image"]
            label = transformed["label"] if "label" in transformed else torch.zeros_like(image[0])
            
            return image, label.long().squeeze(0)
            
        except Exception as e:
            print(f"Error loading {data_dict['name']}: {e}")
            return torch.zeros(1, *self.img_size), torch.zeros(self.img_size).long()


def get_dataset(config, split='train'):
    """Factory function to get dataset based on config"""
    dataset_name = config['dataset']['name'].lower()
    data_dir = config['dataset']['data_dir']
    img_size = tuple(config['model']['img_size'])
    
    if dataset_name == 'synapse':
        return SynapseDataset(data_dir, split, img_size)
    elif dataset_name == 'acdc':
        return ACDCDataset(data_dir, split, img_size)
    elif dataset_name == 'btcv':
        # BTCV uses same structure as Synapse
        return SynapseDataset(data_dir, split, img_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_dataloader(config, split='train'):
    """Create dataloader from config"""
    dataset = get_dataset(config, split)
    
    batch_size = config['training']['batch_size'] if split == 'train' else 1
    shuffle = split == 'train'
    num_workers = 4 if split == 'train' else 2
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=split == 'train'
    )
    
    return loader