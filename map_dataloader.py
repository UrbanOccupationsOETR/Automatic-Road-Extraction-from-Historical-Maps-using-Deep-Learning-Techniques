import os

import albumentations as albu
import numpy as np
import skimage.io
import torch
from torch.utils.data import Dataset as BaseDataset

from utils import convert_from_color, convert_to_color

class Map_Dataset(BaseDataset):
    """
    Dataset iterator. 
    """
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.mask_ids = os.listdir(masks_dir)

        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.mask_ids]
        
        self.class_values = np.arange(0,6,1) 
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        
    @staticmethod
    def _read_img(image_path):
        img = skimage.io.imread(image_path, plugin='tifffile')
        return img

    def __getitem__(self, i):
        
        # read data
        im_id = self.images_fps[i].split('\\')[-1]
        image = self._read_img(self.images_fps[i])
        image = image[:,:,0:3] #omit the fourth channel.
        mask =  self._read_img(self.masks_fps[i])
    
        if self.augmentation is not None:
            transformed = self.augmentation(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            
        if self.preprocessing:
            preprocessed  = self.preprocessing(image=image, mask=mask)
            image,mask = preprocessed['image'], preprocessed['mask']

        mask_raw = convert_from_color(mask) #encode the mask.
        masks = [(mask_raw == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1)      
       
        image = image.transpose(2, 0, 1).astype('float32') 
        mask = mask.transpose(2, 0, 1).astype('float32')
        
        image_ = image / image.max() 
        
        image = torch.as_tensor(image, dtype=torch.float32).cuda()
        mask = torch.as_tensor(mask, dtype=torch.float32).cuda() 

        return image, mask
        
    def __len__(self):
        return len(self.ids)

class Dataset_Inference(BaseDataset):
    
    def __init__(
            self, 
            images_dir, 
            augmentation=None, 
            preprocessing=None,
            sample_ids=False,
    ):
        self.ids = os.listdir(images_dir)
        self.sample_ids = sample_ids

        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        
        self.class_values = np.arange(0,6,1)
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        print("num of images: ",len(self.ids))
        
    @staticmethod
    def _read_img(image_path):
        img = skimage.io.imread(image_path, plugin='tifffile')
        return img

    def __getitem__(self, i):
        
        # read data
        im_id = self.images_fps[i].split('\\')[-1]
        image = self._read_img(self.images_fps[i])
        image = image[:,:,0:3]
    
        if self.augmentation is not None:
            transformed = self.augmentation(image=image)
            image = transformed["image"]
            
        if self.preprocessing:
            preprocessed  = self.preprocessing(image=image)
            image = preprocessed['image']

        image = image.transpose(2, 0, 1).astype('float32') 
        
        image_ = image / image.max()
        
        image = torch.as_tensor(image, dtype=torch.float32).cuda()
        
        if self.sample_ids is not False:
            return image,im_id, self.images_fps[i]
                       
        return image
        
    def __len__(self):
        return len(self.ids)

def get_training_augmentation():
    train_transform =     train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.RandomCrop(height=256, width=256, always_apply=True),
    ]
    return albu.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
    ]
    return albu.Compose(_transform)
