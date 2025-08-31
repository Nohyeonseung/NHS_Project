# dataloader
import os
import glob
import numpy as np
import torch
import nibabel as nib
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.ndimage import rotate
from natsort import natsorted
from scipy.ndimage import rotate, map_coordinates, gaussian_filter
from skimage import exposure

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

class RandomCrop:
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        d, h, w = image.shape
        new_d, new_h, new_w = self.output_size
        pad_d = max(0, new_d - d)
        pad_h = max(0, new_h - h)
        pad_w = max(0, new_w - w)
        pad_d_before, pad_h_before, pad_w_before = pad_d // 2, pad_h // 2, pad_w // 2
        pad_width = ((pad_d_before, pad_d - pad_d_before), (pad_h_before, pad_h - pad_h_before), (pad_w_before, pad_w - pad_w_before))
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            image = np.pad(image, pad_width, mode='constant', constant_values=image.min())
            mask = np.pad(mask, pad_width, mode='constant', constant_values=0)
        d, h, w = image.shape
        top = np.random.randint(0, d - new_d + 1)
        left = np.random.randint(0, h - new_h + 1)
        front = np.random.randint(0, w - new_w + 1)
        sample['image'] = image[top: top + new_d, left: left + new_h, front: front + new_w]
        sample['mask'] = mask[top: top + new_d, left: left + new_h, front: front + new_w]
        return sample

class CTNormalize:
    def __init__(self, min_bound=-1024, max_bound=3071, **kwargs):
        self.min_bound = min_bound
        self.max_bound = max_bound
    def __call__(self, sample):
        image = sample['image']
        image = np.clip(image, self.min_bound, self.max_bound)
        image = (image - self.min_bound) / (self.max_bound - self.min_bound)
        sample['image'] = image.astype(np.float32)
        return sample

class ToTensor:
    def __init__(self, **kwargs):
        pass
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        image = np.expand_dims(image, axis=0)
        sample['image'] = torch.from_numpy(image.astype(np.float32))
        sample['mask'] = torch.from_numpy(mask.astype(np.int64))
        return sample

class MedicalImageDataset3D(Dataset):
    def __init__(self, image_paths, mask_paths, phase='train', transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.phase = phase
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = nib.load(img_path).get_fdata().astype(np.float32)
        mask_path = self.mask_paths[idx]
        mask = nib.load(mask_path).get_fdata().astype(np.uint8)
        sample = {'image': image, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)
        if self.phase == 'test':
            return sample['image'], sample['mask'], img_path
        else:
            return sample['image'], sample['mask']

def load_data_paths(image_dir, mask_dir, target_folders=None):
    all_image_paths = natsorted(glob.glob(os.path.join(image_dir, '**', '*.nii.gz'), recursive=True))
    if target_folders:
        print(f"--- Filtering for {target_folders} ---")
        filtered_image_paths = [p for p in all_image_paths if any(f in p for f in target_folders)]
    else:
        filtered_image_paths = all_image_paths

    paired_image_paths, paired_mask_paths = [], []
    for img_path in filtered_image_paths:
        img_filename = os.path.basename(img_path)
        base_name, ext = os.path.splitext(img_filename)
        if ext == '.gz':
            base_name, _ = os.path.splitext(base_name)
            ext = '.nii.gz'
        mask_filename = f"{base_name}_lbl{ext}"
        rel_dir = os.path.dirname(os.path.relpath(img_path, image_dir))
        mask_path = os.path.join(mask_dir, rel_dir, mask_filename)
        if os.path.exists(mask_path):
            paired_image_paths.append(img_path)
            paired_mask_paths.append(mask_path)
            
    print(f"Found {len(paired_image_paths)} valid image-mask pairs.")
    assert len(paired_image_paths) > 0, "No valid image-mask pairs were found."
    return paired_image_paths, paired_mask_paths

def get_dataloaders(image_paths, mask_paths, batch_size, train_val_split=0.8, patch_size=(64, 64, 64)):
    train_transform = Compose([CTNormalize(), RandomCrop(patch_size), ToTensor()])
    val_transform = Compose([CTNormalize(), RandomCrop(patch_size), ToTensor()])
    full_dataset = MedicalImageDataset3D(image_paths, mask_paths, transform=train_transform)
    train_size = int(train_val_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return {'train': train_loader, 'val': val_loader}
