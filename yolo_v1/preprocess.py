import torch
import numpy as np
import os
import torchvision
import random

from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import ImageFilter, ImageOps


RESIZE_WIDTH = 448
RESIZE_HEIGHT = 448



CLASS_VOC = {
    'aeroplane': 0,
    'bicycle': 1,
    'bird': 2,
    'boat': 3,
    'bottle': 4,
    'bus': 5,
    'car': 6,
    'cat': 7,
    'chair': 8,
    'cow': 9,
    'diningtable': 10,
    'dog': 11,
    'horse': 12,
    'motorbike': 13,
    'person': 14,
    'pottedplant': 15,
    'sheep': 16,
    'sofa': 17,
    'train': 18,
    'tvmonitor': 19,
    'x_center_01' : 20,
    'y_center_01' : 21,
    'norm_width_01' : 22,
    'norm_height_0' : 23,
    'confidence_0' : 24,
    'x_center_02' : 25,
    'y_center_02' : 26,
    'norm_width_02' : 27,
    'norm_height_02' : 28,
    'confidence_02' : 29,
}




class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img
        
        
class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


def update_coords(c_start_row,
                c_start_col,
                c_end_row,
                c_end_col,
                center_row, center_col):
    
    with_update = False
    
    if center_row > c_start_row:
        c_start_row += 1
        with_update = True
        
    if center_col > c_start_col:
        c_start_col += 1
        with_update = True
        
    if center_row < c_end_row:
        c_end_row -= 1
        with_update = True
        
    if center_col < c_end_col:
        c_end_col -= 1
        with_update = True
        
    return c_start_row, c_start_col, c_end_row, c_end_col, with_update
    
    
    
def center_mask(target_mask, start_row, start_col, end_row, end_col, max_val = 1.0):
    
    assert end_row >= start_row, 'end_row >= start_row error'
    assert end_col >= start_col, 'end_col >= start_col error'
    
    center_row = (end_row + start_row) // 2
    center_col = (end_col + start_col) // 2
    
    with_update = True
    
    scaler = max(min(end_row - start_row, end_col - start_col), 1.0)
    
    while with_update:
        target_mask[start_row:end_row+1, start_col:end_col+1] += max_val / scaler
        start_row, start_col, end_row, end_col, with_update = update_coords(start_row,
                                                                                    start_col,
                                                                                    end_row,
                                                                                    end_col,
                                                                                    center_row,
                                                                                    center_col)
        
        
    return target_mask




def encode_feature_map(annotation, n_segments, n_bbox, n_classes):
    
    width = float(annotation['annotation']['size']['width'])
    height = float(annotation['annotation']['size']['height'])
    
    feature_map = np.zeros((n_classes + n_bbox * 5, n_segments, n_segments))
    
    for obj in annotation['annotation']['object']:
        
        
        class_idx = CLASS_VOC[obj['name']]
        
        bb = obj['bndbox']
        xmin, ymin, xmax, ymax = int(bb['xmin']), int(bb['ymin']), int(bb['xmax']), int(bb['ymax'])
   
        start_col = int(np.floor(xmin * n_segments / width))
        start_row = int(np.floor(ymin * n_segments / height))
        
        num_cols = int(np.ceil((xmax - xmin) * n_segments / width))
        num_rows = int(np.ceil((ymax - ymin) * n_segments / height))
        
        end_col = start_col + num_cols
        end_row = start_row + num_rows
        
        xcenter = (xmin + xmax) / 2.0
        ycenter = (ymin + ymax) / 2.0
        
        center_val_col = (xcenter / width) * n_segments
        center_val_row = (ycenter / height) * n_segments
        
        center_idx_col, center_idx_row = int(np.floor(center_val_col)), int(np.floor(center_val_row))
        
        x_offset = center_val_col - center_idx_col
        y_offset = center_val_row - center_idx_row
        
        norm_width = (xmax - xmin) / width
        norm_height = (ymax - ymin) / height
        
        for i in range(0, n_bbox):
            feature_map[n_classes + 0 + 5*i, center_idx_row, center_idx_col] = x_offset
            feature_map[n_classes + 1 + 5*i, center_idx_row, center_idx_col] = y_offset          
            feature_map[n_classes + 2 + 5*i, center_idx_row, center_idx_col] = norm_width
            feature_map[n_classes + 3 + 5*i, center_idx_row, center_idx_col] = norm_height
            
            feature_map[n_classes + 4 + 5*i, center_idx_row, center_idx_col] = 1.0
                
        feature_map[class_idx, center_idx_row, center_idx_col] = 1.0

    np.clip(feature_map, 0.0, 1.0, out=feature_map)
        
    return feature_map
        

def encode_target(y):
    n_segments = 7
    n_bbox = 2
    n_classes = 20
    
    return encode_feature_map(y, n_segments, n_bbox, n_classes)


def get_train_dataloader(dataset_dir, batch_size, image_set='train'):

    input_transform = transforms.Compose([
        transforms.RandomApply(
            [
                transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                    saturation=0.2, hue=0.1)],
            p=0.8
        ),
        transforms.Resize((RESIZE_WIDTH, RESIZE_HEIGHT)),
        transforms.ToTensor(),
    ])

    ds = torchvision.datasets.VOCDetection(
            root=dataset_dir,
            year='2007',
            image_set=image_set, 
            download=True, transform=input_transform, target_transform=encode_target)

    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=8)

    return dl


def get_val_dataloader(dataset_dir, batch_size):

    input_transform = transforms.Compose([
        transforms.Resize((RESIZE_WIDTH, RESIZE_HEIGHT)),
        transforms.ToTensor()
    ])

    ds = torchvision.datasets.VOCDetection(
            root=dataset_dir,
            year='2007',
            image_set='val', 
            download=True, transform=input_transform, target_transform=encode_target)

    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=8)

    return dl
