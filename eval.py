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
from torchvision.utils import make_grid
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_mutual_information as nmi
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
IOU_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.07, 0.08, 0.09)
all_rows_by_thr = {occ_thr: [] for occ_thr in IOU_THRESHOLDS}
all_ssim_rows = []
all_psnr_rows = []
all_nmi_rows = []

def compute_iou(pred, gt, occ_thr=0.8):
    pred_occ = (pred > occ_thr)
    gt_occ   = (gt > occ_thr)

    inter = (pred_occ & gt_occ).sum().float()
    union = (pred_occ | gt_occ).sum().float()

    iou = inter / (union + 1e-6)

    return iou

def compute_miou(pred, gt, occ_thr=0.8):
    pred_occ = (pred > occ_thr)
    gt_occ   = (gt > occ_thr)

    pred_free = ~pred_occ
    gt_free   = ~gt_occ

    inter_occ = (pred_occ & gt_occ).sum().float()
    union_occ = (pred_occ | gt_occ).sum().float()
    iou_occ = inter_occ / (union_occ + 1e-6)

    inter_free = (pred_free & gt_free).sum().float()
    union_free = (pred_free | gt_free).sum().float()
    iou_free = inter_free / (union_free + 1e-6)

    miou = (iou_occ + iou_free) / 2.0

    return miou

def compute_ssim_metric(pred, gt):
    """Compute SSIM between prediction and ground truth.
    
    Args:
        pred: prediction tensor (C, H, W)
        gt: ground truth tensor (C, H, W)
    
    Returns:
        ssim score (float)
    """
    pred_np = pred.detach().cpu().numpy().astype(np.float32)
    gt_np = gt.detach().cpu().numpy().astype(np.float32)
    
    # Compute SSIM (data_range should be 1.0 for normalized values)
    score = ssim(pred_np, gt_np, data_range=1.0, channel_axis=0)
    
    return float(score)


def compute_psnr_metric(pred, gt):
    """Compute PSNR between prediction and ground truth.
    
    Args:
        pred: prediction tensor (C, H, W)
        gt: ground truth tensor (C, H, W)
    
    Returns:
        psnr score (float)
    """
    pred_np = pred.detach().cpu().numpy().astype(np.float32)
    gt_np = gt.detach().cpu().numpy().astype(np.float32)
    
    # Compute PSNR (data_range should be 1.0 for normalized values)
    score = psnr(gt_np, pred_np, data_range=1.0)
    
    return float(score)


def compute_nmi_metric(pred, gt):
    """Compute NMI between prediction and ground truth.
    
    Args:
        pred: prediction tensor (C, H, W)
        gt: ground truth tensor (C, H, W)
    
    Returns:
        nmi score (float)
    """
    pred_np = pred.detach().cpu().numpy().astype(np.float32)
    gt_np = gt.detach().cpu().numpy().astype(np.float32)
    
    # Compute NMI
    score = nmi(gt_np, pred_np)
    
    return float(score)

def save_prediction_overlay_gif(prediction_maps, gt_binary, output_path,
                                frame_duration_ms=100, scale=8):
    """Save a GIF of predicted maps with GT occupied cells highlighted in red."""
    frames = []

    for frame_idx in range(prediction_maps.shape[0]):
        pred_map = prediction_maps[frame_idx, 0].detach().cpu().clamp(0, 1).numpy()
        gt_map = gt_binary[frame_idx, 0].detach().cpu().numpy() > 0.5

        pred_uint8 = (pred_map * 255).astype(np.uint8)
        rgb_frame = np.stack([pred_uint8, pred_uint8, pred_uint8], axis=-1)
        rgb_frame[gt_map] = np.array([255, 0, 0], dtype=np.uint8)

        pil_frame = Image.fromarray(rgb_frame, mode="RGB")
        if scale != 1:
            pil_frame = pil_frame.resize(
                (IMG_SIZE * scale, IMG_SIZE * scale),
                Image.Resampling.NEAREST,
            )
        frames.append(pil_frame)

    if frames:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration_ms,
            loop=0,
        )


def save_iou_tables(all_rows_by_thr, output_dir):
    for occ_thr, rows in all_rows_by_thr.items():
        csv_path = os.path.join(output_dir, f"eval_table_iou_{occ_thr:.2f}.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

def save_ssim_table(all_ssim_rows, output_dir):
    """Save SSIM metrics to CSV file."""
    if all_ssim_rows:
        csv_path = os.path.join(output_dir, "eval_table_ssim.csv")
        pd.DataFrame(all_ssim_rows).to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

def save_psnr_table(all_psnr_rows, output_dir):
    """Save PSNR metrics to CSV file."""
    if all_psnr_rows:
        csv_path = os.path.join(output_dir, "eval_table_psnr.csv")
        pd.DataFrame(all_psnr_rows).to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

def save_nmi_table(all_nmi_rows, output_dir):
    """Save NMI metrics to CSV file."""
    if all_nmi_rows:
        csv_path = os.path.join(output_dir, "eval_table_nmi.csv")
        pd.DataFrame(all_nmi_rows).to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")


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
def evaluate_ldm(model, dataloader, device, output_dir, ddim_steps=10, ddim_eta=1.0, 
                 occ_thr=0.1, save_images=False, num_batches=None, num_samples=1):
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
                    batch_size=c_exp.shape[0],
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
            
            # Compute SSIM metrics
            ssim_row = {
                "i": int(batch_idx),
            }
            for n in range(SEQ_LEN):
                gt_map = x_gt[0, n]
                pred_map = prediction_maps[n]
                ssim_value = compute_ssim_metric(pred_map, gt_map)
                ssim_row[f"n={n+1}"] = ssim_value
            all_ssim_rows.append(ssim_row)

            psnr_row = {
                "i": int(batch_idx),
            }
            for n in range(SEQ_LEN):
                gt_map = x_gt[0, n]
                pred_map = prediction_maps[n]
                psnr_value = compute_psnr_metric(pred_map, gt_map)
                psnr_row[f"n={n+1}"] = psnr_value
            all_psnr_rows.append(psnr_row)

            nmi_row = {
                "i": int(batch_idx),
            }
            for n in range(SEQ_LEN):
                gt_map = x_gt[0, n]
                pred_map = prediction_maps[n]
                nmi_value = compute_nmi_metric(pred_map, gt_map)
                nmi_row[f"n={n+1}"] = nmi_value
            all_nmi_rows.append(nmi_row)

            for occ_thr in IOU_THRESHOLDS:
                row = {
                    "i": int(batch_idx),
                    "Inference_time": float(total_time_ms),
                    "occ_thr": float(occ_thr),
                }
                for n in range(SEQ_LEN):
                    gt_map = x_gt[0, n]
                    pred_map = prediction_maps[n]
                    iou_value = float(compute_iou(pred_map, gt_map, occ_thr=occ_thr).item())
                    row[f"n={n+1}"] = iou_value
                all_rows_by_thr[occ_thr].append(row)

            if (batch_idx + 1) % 1000 == 0:
                save_iou_tables(all_rows_by_thr, output_dir)
                save_ssim_table(all_ssim_rows, output_dir)
                save_psnr_table(all_psnr_rows, output_dir)
                save_nmi_table(all_nmi_rows, output_dir)

            if save_images and batch_idx < 500:
                fontsize = 8

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
                    a.set_title(f"n={m+1}", fontdict={'fontsize': fontsize})
                img_path = os.path.join(output_dir, f"mask_{batch_idx}.png")
                plt.savefig(img_path, dpi=500)
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
                plt.savefig(img_path, dpi=500)
                plt.close(fig)

                gif_path = os.path.join(output_dir, f"overlay_{batch_idx}.gif")
                save_prediction_overlay_gif(prediction_maps, x_gt[0], gif_path)

    save_iou_tables(all_rows_by_thr, output_dir)
    save_ssim_table(all_ssim_rows, output_dir)
    save_psnr_table(all_psnr_rows, output_dir)
    save_nmi_table(all_nmi_rows, output_dir)

    return None

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
        default=0.8,
        help="Occupancy threshold for IoU calculation"
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
        num_samples=opt.num_samples
    )
    
    print(f"Evaluation complete! Results saved to {opt.output_dir}")
