#!/usr/bin/env python

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

output_dir = "ae_eval_results"
os.makedirs(output_dir, exist_ok=True)

config = OmegaConf.load("configs/autoencoder/ae_eval.yaml")
model = instantiate_from_config(config.model)
ckpt = torch.load("logs/2026-03-27T17-09-28_ae2.0.1/checkpoints/last.ckpt", map_location="cpu")
state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
model.load_state_dict(state_dict, strict=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

test_cfg = config.data.params.test
dataset = instantiate_from_config(test_cfg)
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
    for batch_idx, batch in enumerate(tqdm(dataloader)):
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
        iou_tensor = torch.stack(batch_iou)
        iou_list.append(iou_tensor.cpu().numpy())

        # Periodic CSV saving every 100 batches
        if (batch_idx + 1) % 100 == 0:
            iou_array_tmp = np.array(iou_list)
            per_timestep_df_tmp = pd.DataFrame(iou_array_tmp, columns=[f"timestep_{t}" for t in range(iou_array_tmp.shape[1])])
            per_timestep_df_tmp.to_csv(os.path.join(output_dir, "all_iou_per_timestep.csv"), index_label="batch_idx")

        fig = plt.figure(figsize=(8, 1))
        for m in range(SEQ_LEN):
            a = fig.add_subplot(1, SEQ_LEN, m + 1)
            mask = x[0, m]  # (C, H, W)
            input_grid = make_grid(mask.detach().cpu())
            input_image = input_grid.permute(1, 2, 0)
            plt.imshow(input_image)
            plt.xticks([])
            plt.yticks([])
            fontsize = 8
            a.set_title(f"n={m+1}", fontdict={'fontsize': fontsize})
        img_path = os.path.join(output_dir, f"gt_{batch_idx}.png")
        plt.savefig(img_path, dpi=300)
        plt.close(fig)

        # Predicted occupancy maps
        fig = plt.figure(figsize=(8, 1))
        for m in range(SEQ_LEN):
            a = fig.add_subplot(1, SEQ_LEN, m + 1)
            pred = recon[0, m]  # (C, H, W)
            input_grid = make_grid(pred.detach().cpu())
            input_image = input_grid.permute(1, 2, 0)
            plt.imshow(input_image)
            plt.xticks([])
            plt.yticks([])
            a.set_title(f"n={m+1}", fontdict={'fontsize': fontsize})
        img_path = os.path.join(output_dir, f"recon_{batch_idx}.png")
        plt.savefig(img_path, dpi=300)
        plt.close(fig)
