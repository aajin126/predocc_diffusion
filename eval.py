#!/usr/bin/env python
"""
LDM evaluation script
Evaluates PredOccLatentDiffusion model with DDIM sampling on test/validation dataset
Computes IoU, WMSE, prediction time, and saves visualizations
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
all_rows = []    
csv_path = os.path.join("output", "ldm2.1", "eval_table.csv")

def compute_iou(pred, gt, occ_thr=0.3):
    pred_occ = (pred > occ_thr)
    gt_occ   = (gt > occ_thr)

    inter = (pred_occ & gt_occ).sum().float()
    union = (pred_occ | gt_occ).sum().float()

    iou = inter / (union + 1e-6)

    return iou


def compute_wmse(pred, gt, occ_thr=0.3, occupied_weight=2.0, free_weight=1.0):
    gt_occ = (gt > occ_thr).float()
    weights = torch.where(
        gt_occ > 0,
        torch.full_like(gt, occupied_weight),
        torch.full_like(gt, free_weight)
    )
    weighted_sq_error = weights * (pred - gt).pow(2)
    return weighted_sq_error.sum() / (weights.sum() + 1e-6)

def load_ldm_model(ckpt_path, base_configs, unknown, device):

    # Load and merge all config files
    configs = [OmegaConf.load(cfg) for cfg in base_configs]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    # Instantiate model
    model = instantiate_from_config(config.model)
    model = model.to(device)
    
    # Load LDM checkpoint
    if ckpt_path is not None:
        print(f"Loading LDM checkpoint from {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        
        # Check what's in the checkpoint
        if "state_dict" in sd:
            state_dict = sd["state_dict"]
        else:
            state_dict = sd
        
        print(f"Checkpoint state_dict has {len(state_dict)} keys")
        
        # Fix key naming for backward compatibility
        fixed_state_dict = {}
        for k, v in state_dict.items():
            # encoder -> _encoder
            k = k.replace(".encoder.", "._encoder.")
            if k.startswith("encoder."):
                k = "_" + k
            fixed_state_dict[k] = v
        
        missing, unexpected = model.load_state_dict(fixed_state_dict, strict=False)
        print(f"[LDM] loaded from {ckpt_path}")
        print(f"[LDM] missing keys: {len(missing)}")
        print(f"[LDM] unexpected keys: {len(unexpected)}")
        
        if missing:
            print(f"\n  Missing keys (all):")
            for k in list(missing)[:100]:
                print(f"    {k}")
        if unexpected:
            print(f"\n  Unexpected keys (all):")
            for k in list(unexpected)[:100]:
                print(f"    {k}")
    
    model.eval()
    
    return model, config


@torch.no_grad()
def evaluate_ldm(model, dataloader, device, output_dir, ddim_steps=20, ddim_eta=1.0, 
                 occ_thr=0.3, save_images=True, num_batches=None, num_samples=1,
                 wmse_occ_weight=2.0, wmse_free_weight=1.0):
    """Evaluate LDM model on test/validation dataset"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize DDIM sampler
    sampler = DDIMSampler(model)
    
    results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", total=num_batches)):
            if num_batches is not None and batch_idx >= num_batches:
                break
            
            # Get model inputs z, c, x_gt, xrec, xc
            x_in, x_gt, x_occ_map, x_rel, y_rel, th_rel = model.get_input(
                                                            batch,
                                                            process = 'test'
                                                        )

            t0 = time.perf_counter()
            c, _ = model.get_encoding(x_in, x_gt, x_occ_map)
            c_exp = c.repeat_interleave(model.first_stage_model.seq_len, dim=0) # (B*T, 32, 16, 16)

            # Measure sampling time
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            # DDIM sampling from random noise
            with model.ema_scope("Evaluation"):
                z_samples, _ = model.sample_log(
                    cond=c_exp,
                    batch_size=num_samples * SEQ_LEN,
                    ddim=True,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta
                )
            
            # Decode latent to sequence
            pred_seq = model.decode_first_stage(z_samples)  # (num_samples, T, 1, H, W)

            # Reprojection for each time step
            prediction_maps = torch.zeros(SEQ_LEN, 1, IMG_SIZE, IMG_SIZE).to(device)

            for t in range(SEQ_LEN):
                pred_map_t, _ = reprojection(pred_seq[:, t], x_rel[:, t], y_rel[:, t], th_rel[:, t], MAP_X_LIMIT, MAP_Y_LIMIT) # pred_seq[:, t] -> (num_samples, C, H, W)
                pred_map_t = pred_map_t.reshape(-1, 1, 1, IMG_SIZE, IMG_SIZE)  # (num_samples, 1, 1, H, W)
                predictions = pred_map_t.squeeze(1)  # (num_samples, 1, H, W)
                pred_mean = torch.mean(predictions, dim=0, keepdim=True)  # (1, 1, H, W)
                prediction_maps[t, 0] = pred_mean.squeeze()
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            
            total_time_ms = (t1 - t0) * 1000
            
            # Compute metrics per frame - average across all samples
            row = {"i": batch_idx, "Inference_time": float(total_time_ms)}
            
            for t in range(SEQ_LEN):
                gt_map = x_gt[0, t]  # (H, W)
                pred_map = prediction_maps[t, 0]  # (H, W)
                row[f"n={t+1}"] = float(compute_iou(pred_map, gt_map, occ_thr=occ_thr).item())
                row[f"wmse_n={t+1}"] = float(
                    compute_wmse(
                        pred_map,
                        gt_map,
                        occ_thr=occ_thr,
                        occupied_weight=wmse_occ_weight,
                        free_weight=wmse_free_weight,
                    ).item()
                )
            results.append(row)

            if (batch_idx + 1) % 100 == 0:
                pd.DataFrame(results).to_csv(csv_path, index=False)
            
            # GT occupancy maps
            fig = plt.figure(figsize=(8, 1))
            for m in range(SEQ_LEN):   
                a = fig.add_subplot(1, SEQ_LEN, m + 1)
                mask = x_gt[0, m]
                input_grid = make_grid(mask.detach().cpu())
                input_image = input_grid.permute(1, 2, 0)
                plt.imshow(input_image)
                plt.xticks([])
                plt.yticks([])
                fontsize = 8
                a.set_title(f"n={m+1}", fontdict={'fontsize': fontsize})
            img_path = os.path.join(output_dir, f"mask_{batch_idx}.png")
            plt.savefig(img_path, dpi=300)
            plt.close(fig)

            # Predicted occupancy maps
            fig = plt.figure(figsize=(8, 1))
            for m in range(SEQ_LEN):   
                a = fig.add_subplot(1, SEQ_LEN, m + 1)
                pred = prediction_maps[m]
                input_grid = make_grid(pred.detach().cpu())
                input_image = input_grid.permute(1, 2, 0)
                plt.imshow(input_image)
                plt.xticks([])
                plt.yticks([])
                a.set_title(f"n={m+1}", fontdict={'fontsize': fontsize})
            img_path = os.path.join(output_dir, f"pred{batch_idx}.png")
            plt.savefig(img_path, dpi=300)
            plt.close(fig)
    
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
   
    
    return df

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to LDM checkpoint"
    )
    parser.add_argument(
        "-b",
        "--base",
        type=str,
        nargs="*",
        default=[],
        help="Path to config file(s) to load and merge (like main.py). Can specify multiple files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="Number of DDIM sampling steps"
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=1.0,
        help="DDIM eta (0.0=deterministic, 1.0=stochastic)"
    )
    parser.add_argument(
        "--occ_thr",
        type=float,
        default=0.3,
        help="Occupancy threshold for IoU and WMSE calculation"
    )
    parser.add_argument(
        "--wmse_occ_weight",
        "--mwse_occ_weight",
        dest="wmse_occ_weight",
        type=float,
        default=2.0,
        help="WMSE weight for occupied cells"
    )
    parser.add_argument(
        "--wmse_free_weight",
        "--mwse_free_weight",
        dest="wmse_free_weight",
        type=float,
        default=1.0,
        help="WMSE weight for free cells"
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=None,
        help="Number of batches to evaluate (None=all)"
    )
    parser.add_argument(
        "--no_images",
        action="store_true",
        help="Skip saving visualization images"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of diffusion samples per input"
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("\n=== Loading LDM Model ===")
    model, config = load_ldm_model(opt.ckpt, opt.base, unknown, device)
    print(f"LDM model type: {type(model)}")
    print(f"First stage model: {model.first_stage_model}")
    print(f"First stage model checkpoint path: {model.first_stage_ckpt_path}")
    
    # Load data (like main.py)
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    dataloader = data.test_dataloader()

    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    # Evaluate
    results_df = evaluate_ldm(
        model,
        dataloader,
        device,
        opt.output_dir,
        ddim_steps=opt.ddim_steps,
        ddim_eta=opt.ddim_eta,
        occ_thr=opt.occ_thr,
        save_images=not opt.no_images,
        num_batches=opt.num_batches,
        num_samples=opt.num_samples,
        wmse_occ_weight=opt.wmse_occ_weight,
        wmse_free_weight=opt.wmse_free_weight,
    )
    
    print(f"Evaluation complete! Results saved to {opt.output_dir}")
