#!/usr/bin/env python
"""
LDM evaluation script
Evaluates PredOccLatentDiffusion model with DDIM sampling on test/validation dataset
Computes IoU, prediction time, and saves visualizations
"""

import sys
import os
import argparse
import glob
import time
import yaml

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from PIL import Image
from predocc.util import instantiate_from_config

sys.path.append(os.path.join(os.getcwd(), 'predocc'))
sys.path.append(os.getcwd())

from models.diffusion.ddpm import PredOccLatentDiffusion
from models.diffusion.ddim import DDIMSampler
from util import instantiate_from_config
from data.dataloader import PredOccDataset
from predocc.occ_util import reprojection
import pandas as pd

# Constants
SEQ_LEN = 10
IMG_SIZE = 64
MAP_X_LIMIT = [0, 6.4]      # Map limits on the x-axis
MAP_Y_LIMIT = [-3.2, 3.2]   # Map limits on the y-axis
RESOLUTION = 0.1        # Grid resolution in [m]'
TRESHOLD_P_OCC = 0.8    # Occupancy threshold

config = OmegaConf.load("configs/autoencoder/ae_eval.yaml")
model = instantiate_from_config(config.model)
ckpt = torch.load("path/to/ae_ckpt.ckpt", map_location="cpu")
state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
model.load_state_dict(state_dict, strict=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

train_cfg = config.data.params.validation  # 또는 test
dataset = instantiate_from_config(train_cfg)
dataloader = DataLoader(
    dataset,
    batch_size=config.data.params.batch_size,
    num_workers=config.data.params.num_workers,
    shuffle=False,
)

iou_list = []

def compute_iou(pred, gt, occ_thr=0.3):
    pred_occ = (pred > occ_thr)
    gt_occ   = (gt > occ_thr)

    inter = (pred_occ & gt_occ).sum().float()
    union = (pred_occ | gt_occ).sum().float()

    iou = inter / (union + 1e-6)

    return iou

with torch.no_grad():
    for batch in tqdm(dataloader):
        # (B, T, C, H, W), (B, map_channels, H, W)
        x, x_map = model.get_input(batch, model.image_key)
        recon, _ = model(x, x_map)  # (B, T, C, H, W)
        x = x.to(device)
        recon = recon.to(device)

        # IoU
        batch_iou = []
        for t in range(recon.shape[1]):
            iou = compute_iou(recon[:, t], x[:, t])
            batch_iou.append(iou)
        iou_list.append(batch_iou)

iou_array = np.array(iou_list)  # shape: (num_batches, seq_len)
mean_iou_per_timestep = np.mean(iou_array, axis=0)
for t, miou in enumerate(mean_iou_per_timestep):
    print(f"timestep {t}: IoU={miou:.6f}")