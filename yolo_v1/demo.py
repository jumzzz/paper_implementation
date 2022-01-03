import argparse
import os
from preprocess import get_val_dataloader
from postprocess import DecodeBB
import torch
from PIL import ImageFont, Image, ImageDraw
import numpy as plt
import numpy as np
from model import YOLO_V1


IDX_CLASSES = {
    0: 'aeroplane',
    1: 'bicycle',
    2: 'bird',
    3: 'boat',
    4: 'bottle',
    5: 'bus',
    6: 'car',
    7: 'cat',
    8: 'chair',
    9: 'cow',
    10: 'diningtable',
    11: 'dog',
    12: 'horse',
    13: 'motorbike',
    14: 'person',
    15: 'pottedplant',
    16: 'sheep',
    17: 'sofa',
    18: 'train',
    19: 'tvmonitor'
 }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='Path of the model')
    parser.add_argument('--dataset', help='Path to dataset.')
    parser.add_argument('--batch_size', help='Batch Size (Optional) for dataset to be sampled.', type=int, default=32)
    args = parser.parse_args()

    return args


def sample_image(X, bb_list):

    image_idx = 0
    max_prob = 0.0

    for idx, bb_sample in enumerate(bb_list):

        if len(bb_sample) == 0:
            continue

        if len(bb_sample) == 1:
            if bb_sample[0]['class_prob'] > max_prob:
                image_idx = idx
                max_prob = bb_sample[0]['class_prob']


    image_sample = (X[image_idx,:,:,:].cpu().permute(1,2,0).detach().numpy() * 255).astype(np.uint8)
    image_sample = Image.fromarray(image_sample)

    bb_sample = bb_list[image_idx]
    drawer = ImageDraw.Draw(image_sample)
    font = ImageFont.load_default()
    font


    for bb in bb_sample:
        class_idx = int(bb['class_idx'])
        class_str = IDX_CLASSES[class_idx]
        class_prob = '{:2.1f}'.format(bb['class_prob'])
        class_str = f' {class_str} : {class_prob}'.upper()
        drawer.rectangle([bb['xmin'], bb['ymin'], bb['xmax'], bb['ymax']], outline='white', width=2)
        drawer.text((bb['xmin'] + 5, bb['ymin']- 15),class_str,fill='#9acd32')


    image_sample.show()


def run_demo():
    
    args = get_args()
    val_dl = get_val_dataloader(args.dataset, args.batch_size)
    model_trained = torch.load(args.model_path).cpu()
 
    X,y = next(iter(val_dl))

    yp = model_trained(X)

    yp = yp.detach().numpy()
    y = y.detach().numpy()

    bb_decoder = DecodeBB(min_confidence=0.3)
    bb_list = bb_decoder.compile_bbox(yp)
    sample_image(X, bb_list)



if __name__ == '__main__':
    run_demo()