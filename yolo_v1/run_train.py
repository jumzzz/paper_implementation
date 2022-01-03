import os
import argparse
from pytorch_lightning.accelerators import accelerator

from torch.utils.data import DataLoader, dataset, random_split
import torchvision

from torchvision import transforms
from preprocess import encode_target
import torch


# I just want to measure execution time
import time

from model import YOLO_V1

# import loss_utils

from train import train_model
from preprocess import get_train_dataloader, get_val_dataloader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Directory of dataset')
    parser.add_argument('--batch_size', help='Batch Size (default = 32).', default=32, type=int)
    parser.add_argument('--epoch', help='Number of Epochs during training.', default=5, type=int)
    parser.add_argument('--model_dir', help='Directory for Model.', default='tb_logs')

    args = parser.parse_args()
    return args


def make_dir(target_dir):
    if os.path.isdir(target_dir):
        return
    os.mkdir(target_dir)

    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_yolo():
    args = get_args()

    make_dir(args.dataset)
    make_dir(args.model_dir)

    tr_dl = get_train_dataloader(args.dataset, args.batch_size)
    val_dl = get_val_dataloader(args.dataset, args.batch_size)

    start_time = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = YOLO_V1()

    print('Model Parameters: ', count_parameters(model))

    model_path = os.path.join(args.model_dir, 'model.pt')

    tr_hist, val_hist = train_model(model, tr_dl, val_dl, device, args.epoch, model_path, patience=5)

    end_time = time.time()

    elapsed_sec = end_time - start_time
    elapsed_min = elapsed_sec / 60.0

    print('Elapsed (sec): ', elapsed_sec)
    print('Elapsed (min): ', elapsed_min)



if __name__ == '__main__':
    run_yolo()