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
TRESHOLD_P_OCC = 0.5    # Occupancy threshold
SAMPLE_COLORS = np.array([
    [0.10, 0.45, 0.90],
    [0.90, 0.15, 0.15],
    [0.10, 0.70, 0.30],
    [0.95, 0.65, 0.10],
    [0.55, 0.25, 0.80],
    [0.10, 0.75, 0.75],
    [0.85, 0.35, 0.60],
    [0.50, 0.50, 0.50],
], dtype=np.float32)
all_rows = []    
csv_path = os.path.join("output", "ldm2.0_miou", "eval_table.csv")

def compute_iou(pred, gt, occ_thr=0.5):
    pred_occ = (pred > occ_thr)
    gt_occ   = (gt > occ_thr)

    inter = (pred_occ & gt_occ).sum().float()
    union = (pred_occ | gt_occ).sum().float()

    iou = inter / (union + 1e-6)

    return iou


def compute_miou(pred, gt, occ_thr=0.5):
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


def save_prediction_overlay_gif(prediction_maps, gt_binary, output_path, frame_duration_ms=100, scale=8):
    """Save a GIF of predicted maps with GT occupied cells highlighted in red."""
    frames = []

    if prediction_maps.dim() != 4:
        raise ValueError(f"prediction_maps must have shape (T, 1, H, W), got {tuple(prediction_maps.shape)}")

    if gt_binary.dim() == 3:
        gt_binary = gt_binary.unsqueeze(1)
    elif gt_binary.dim() != 4:
        raise ValueError(f"gt_binary must have shape (T, H, W) or (T, 1, H, W), got {tuple(gt_binary.shape)}")

    if prediction_maps.shape[0] != gt_binary.shape[0]:
        raise ValueError(
            f"prediction_maps and gt_binary must have the same number of frames, "
            f"got {prediction_maps.shape[0]} and {gt_binary.shape[0]}"
        )

    for frame_idx in range(prediction_maps.shape[0]):
        pred_map = prediction_maps[frame_idx, 0].detach().cpu().clamp(0, 1).numpy()
        gt_map = gt_binary[frame_idx, 0].detach().cpu().numpy() > 0.5

        pred_uint8 = (pred_map * 255).astype(np.uint8)
        rgb_frame = np.stack([pred_uint8, pred_uint8, pred_uint8], axis=-1)
        rgb_frame[gt_map] = np.array([255, 0, 0], dtype=np.uint8)

        pil_frame = Image.fromarray(rgb_frame, mode="RGB")
        if scale != 1:
            width, height = pil_frame.size
            pil_frame = pil_frame.resize((width * scale, height * scale), Image.Resampling.NEAREST)
        frames.append(pil_frame)

    if frames:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration_ms,
            loop=0,
        )


def colorize_prediction_samples(predictions, occ_thr=0.5):
    """Render occupied cells from different samples with different colors."""
    if predictions.dim() == 4:
        predictions = predictions.squeeze(1)

    if predictions.dim() != 3:
        raise ValueError(
            f"predictions must have shape (num_samples, H, W) or (num_samples, 1, H, W), got {tuple(predictions.shape)}"
        )

    num_samples, height, width = predictions.shape
    colors = torch.tensor(SAMPLE_COLORS, device=predictions.device, dtype=predictions.dtype)
    color_image = torch.ones(height, width, 3, device=predictions.device, dtype=predictions.dtype)

    for sample_index in range(num_samples):
        color = colors[sample_index % len(SAMPLE_COLORS)]
        mask = predictions[sample_index] > occ_thr
        color_image[mask] = color

    return color_image.detach().cpu()


def save_prediction_samples_gif(prediction_sample_maps, output_path, occ_thr=0.5, frame_duration_ms=100, scale=8):
    """Save a GIF whose frames are time steps with each sample rendered in a different color."""
    frames = []

    if prediction_sample_maps.dim() != 5:
        raise ValueError(
            "prediction_sample_maps must have shape (num_samples, T, 1, H, W), "
            f"got {tuple(prediction_sample_maps.shape)}"
        )

    for frame_idx in range(prediction_sample_maps.shape[1]):
        color_image = colorize_prediction_samples(prediction_sample_maps[:, frame_idx], occ_thr=occ_thr)
        color_uint8 = (color_image.clamp(0, 1).numpy() * 255).astype(np.uint8)

        pil_frame = Image.fromarray(color_uint8, mode="RGB")
        if scale != 1:
            width, height = pil_frame.size
            pil_frame = pil_frame.resize((width * scale, height * scale), Image.Resampling.NEAREST)
        frames.append(pil_frame)

    if frames:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration_ms,
            loop=0,
        )

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
                 occ_thr=0.5, save_images=True, num_batches=None, num_samples=1):
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
            batch_size = x_gt.shape[0]
            cond_samples = c.repeat_interleave(num_samples, dim=0)
            c_exp = cond_samples.repeat_interleave(model.first_stage_model.seq_len, dim=0) # (B*num_samples*T, 32, 16, 16)

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
            pred_seq = model.decode_first_stage(z_samples)  # (B*num_samples, T, 1, H, W)

            # Reprojection for each time step
            prediction_maps = torch.zeros(SEQ_LEN, 1, IMG_SIZE, IMG_SIZE).to(device)
            prediction_sample_maps = torch.zeros(num_samples, SEQ_LEN, 1, IMG_SIZE, IMG_SIZE).to(device)
            pred_seq = pred_seq.view(batch_size, num_samples, SEQ_LEN, 1, IMG_SIZE, IMG_SIZE)

            for t in range(SEQ_LEN):
                pred_seq_t = pred_seq[:, :, t].reshape(batch_size * num_samples, 1, IMG_SIZE, IMG_SIZE)
                x_rel_t = x_rel[:, t].repeat_interleave(num_samples)
                y_rel_t = y_rel[:, t].repeat_interleave(num_samples)
                th_rel_t = th_rel[:, t].repeat_interleave(num_samples)

                pred_map_t, _ = reprojection(pred_seq_t, x_rel_t, y_rel_t, th_rel_t, MAP_X_LIMIT, MAP_Y_LIMIT)
                pred_map_t = pred_map_t.reshape(batch_size, num_samples, 1, IMG_SIZE, IMG_SIZE)
                predictions = pred_map_t[0]  # (num_samples, 1, H, W)
                prediction_sample_maps[:, t] = predictions
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
            results.append(row)

            if (batch_idx + 1) % 100 == 0:
                pd.DataFrame(results).to_csv(csv_path, index=False)

            if save_images:
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

                gif_path = os.path.join(output_dir, f"overlay_{batch_idx}.gif")
                save_prediction_overlay_gif(prediction_maps, x_gt[0], gif_path)

                sample_gif_path = os.path.join(output_dir, f"samples_{batch_idx}.gif")
                save_prediction_samples_gif(
                    prediction_sample_maps,
                    sample_gif_path,
                    occ_thr=occ_thr,
                )
    
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
        default=0.5,
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
        default=5,
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
