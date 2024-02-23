import imageio
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse
import random
import torch


def seed_everything(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True 

def saveImage(filename, image):
    imageTMP = np.clip(image * 255.0, 0, 255).astype('uint8')
    imageio.imwrite(filename, imageTMP)

def save_rgb (img, filename):
    
    img = np.clip(img, 0., 1.)
    if np.max(img) <= 1:
        img = img * 255
    
    img = img.astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)

    
def load_img (filename, norm=True,):
    img = np.array(Image.open(filename).convert("RGB"))
    if norm:   
        img = img / 255.
        img = img.astype(np.float32)
    return img

def plot_all (images, figsize=(20,10), axis='off', names=None):
    nplots = len(images)
    fig, axs = plt.subplots(1,nplots, figsize=figsize, dpi=80,constrained_layout=True)
    for i in range(nplots):
        axs[i].imshow(images[i])
        if names: axs[i].set_title(names[i])
        axs[i].axis(axis)
    plt.show()
    
def modcrop(img_in, scale=2):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img
    
def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


########## MODEL

def count_params(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params