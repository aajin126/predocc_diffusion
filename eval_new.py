#!/usr/bin/env python
"""
LDM evaluation script

Evaluates PredOccLatentDiffusion model with DDIM sampling on test/validation dataset.

Features:
- --metric {iou,f1}
- Top-K time-independent
- Top-K time-consistent
- Coverage (IoU-based, per time step)
- Inference time logging
- Visualization saving:
    1) GT maps
    2) mean prediction maps
    3) mean prediction grayscale + GT occupied overlay in red
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
csv_path = os.path.join("output", "v6.6.0_f2_5", "eval_table.csv")

def binarize_occ(x, occ_thr=0.5):
    return x > occ_thr

def compute_iou(pred, gt, occ_thr=0.3):
    pred_occ = (pred > occ_thr)
    gt_occ   = (gt > occ_thr)

    inter = (pred_occ & gt_occ).sum().float()
    union = (pred_occ | gt_occ).sum().float()

    iou = inter / (union + 1e-6)

    return iou

def compute_f1(pred, gt, occ_thr=0.5, eps=1e-6):
    pred_occ = binarize_occ(pred, occ_thr)
    gt_occ = binarize_occ(gt, occ_thr)

    tp = (pred_occ & gt_occ).sum().float()
    fp = (pred_occ & (~gt_occ)).sum().float()
    fn = ((~pred_occ) & gt_occ).sum().float()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return f1

def compute_f2(pred, gt, occ_thr=0.5, eps=1e-6):
    
    beta=2.0
    
    pred_occ = binarize_occ(pred, occ_thr)
    gt_occ = binarize_occ(gt, occ_thr)

    tp = (pred_occ & gt_occ).sum().float()
    fp = (pred_occ & (~gt_occ)).sum().float()
    fn = ((~pred_occ) & gt_occ).sum().float()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    beta2 = beta ** 2
    fbeta = (1.0 + beta2) * precision * recall / (beta2 * precision + recall + eps)
    return fbeta 

def get_metric_fn(metric_name):
    metric_name = metric_name.lower()
    if metric_name == "iou":
        return compute_iou
    if metric_name == "f1":
        return compute_f1
    if metric_name == "f2":
        return compute_f2
    raise ValueError(f"Unsupported metric: {metric_name}")


# -----------------------------------------------------------------------------
# Top-K / Coverage
# -----------------------------------------------------------------------------
def compute_topk_time_independent(pred_samples_t, gt_t, metric_fn, occ_thr=0.5):
    """
    pred_samples_t: (K, 1, H, W) or (K, H, W)
    gt_t:           (H, W) or (1, H, W)

    Returns:
        best_score: scalar tensor
        best_idx: int
        all_scores: list[float]
    """
    scores = []
    for k in range(pred_samples_t.shape[0]):
        score_k = metric_fn(pred_samples_t[k], gt_t, occ_thr=occ_thr)
        scores.append(score_k)

    scores_tensor = torch.stack(scores)  # (K,)
    best_score, best_idx = torch.max(scores_tensor, dim=0)

    return best_score, int(best_idx.item()), [float(s.item()) for s in scores_tensor]

def compute_topk_time_consistent(pred_samples, gt_seq, metric_fn, occ_thr=0.5, reduce="mean"):
    """
    pred_samples: (K, T, 1, H, W)
    gt_seq:       (T, H, W) or (T, 1, H, W)

    One sample k is fixed across all time steps.

    Returns:
        best_seq_score: scalar tensor
        best_sample_idx: int
        sample_scores: list[float]
        per_t_scores_of_best: list[float]
    """
    K, T = pred_samples.shape[0], pred_samples.shape[1]
    sample_scores = []
    per_sample_per_t = []

    for k in range(K):
        per_t = []
        for t in range(T):
            s = metric_fn(pred_samples[k, t], gt_seq[t], occ_thr=occ_thr)
            per_t.append(s)

        per_t_tensor = torch.stack(per_t)  # (T,)
        per_sample_per_t.append(per_t_tensor)

        if reduce == "sum":
            agg = per_t_tensor.sum()
        elif reduce == "mean":
            agg = per_t_tensor.mean()
        else:
            raise ValueError(f"Unsupported reduce: {reduce}")

        sample_scores.append(agg)

    sample_scores_tensor = torch.stack(sample_scores)  # (K,)
    best_seq_score, best_sample_idx = torch.max(sample_scores_tensor, dim=0)
    best_per_t = per_sample_per_t[int(best_sample_idx.item())]

    return (
        best_seq_score,
        int(best_sample_idx.item()),
        [float(s.item()) for s in sample_scores_tensor],
        [float(s.item()) for s in best_per_t],
    )


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------
def tensor_to_2d(x):
    """
    Converts (H,W) or (1,H,W) or (1,1,H,W) -> (H,W) numpy-compatible tensor.
    """
    while x.dim() > 2:
        x = x.squeeze(0)
    return x


def save_gt_maps(x_gt, output_dir, batch_idx, seq_len=SEQ_LEN):
    fontsize = 8
    fig = plt.figure(figsize=(8, 1))
    for m in range(seq_len):
        a = fig.add_subplot(1, seq_len, m + 1)
        gt_map = tensor_to_2d(x_gt[0, m]).detach().cpu()
        plt.imshow(gt_map, cmap="gray", vmin=0.0, vmax=1.0)
        plt.xticks([])
        plt.yticks([])
        a.set_title(f"n={m+1}", fontdict={"fontsize": fontsize})
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"mask_{batch_idx}.png"), dpi=500)
    plt.close(fig)


def save_pred_mean_maps(prediction_maps, output_dir, batch_idx, seq_len=SEQ_LEN):
    """
    prediction_maps: (K, T, 1, H, W)
    Saves sample-mean prediction only.
    """
    fontsize = 8
    fig = plt.figure(figsize=(8, 1))
    for m in range(seq_len):
        a = fig.add_subplot(1, seq_len, m + 1)
        pred_mean = prediction_maps[:, m].mean(dim=0)      # (1, H, W)
        pred_mean = tensor_to_2d(pred_mean).detach().cpu()
        plt.imshow(pred_mean, cmap="gray", vmin=0.0, vmax=1.0)
        plt.xticks([])
        plt.yticks([])
        a.set_title(f"n={m+1}", fontdict={"fontsize": fontsize})
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"pred_mean_{batch_idx}.png"), dpi=500)
    plt.close(fig)


def save_pred_mean_overlay_maps(prediction_maps, x_gt, output_dir, batch_idx, occ_thr=0.5, seq_len=SEQ_LEN):
    """
    background: sample mean map in grayscale
    overlay: GT occupied cells in red
    """
    fontsize = 8
    fig = plt.figure(figsize=(8, 1))

    for m in range(seq_len):
        a = fig.add_subplot(1, seq_len, m + 1)

        pred_mean = prediction_maps[:, m].mean(dim=0)      # (1, H, W)
        pred_mean = tensor_to_2d(pred_mean).detach().cpu().numpy()

        gt_map = tensor_to_2d(x_gt[0, m]).detach().cpu().numpy()
        gt_occ = gt_map > occ_thr

        # grayscale background
        plt.imshow(pred_mean, cmap="gray", vmin=0.0, vmax=1.0)

        # red overlay with alpha only on occupied cells
        overlay = np.zeros((IMG_SIZE, IMG_SIZE, 4), dtype=np.float32)
        overlay[..., 0] = 1.0                    # red channel
        overlay[..., 3] = gt_occ.astype(np.float32) * 0.75  # alpha
        plt.imshow(overlay)

        plt.xticks([])
        plt.yticks([])
        a.set_title(f"n={m+1}", fontdict={"fontsize": fontsize})

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"pred_mean_overlay_{batch_idx}.png"), dpi=500)
    plt.close(fig)

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
def evaluate_ldm(model, dataloader, device, output_dir, metric="iou", ddim_steps=10, ddim_eta=1.0, 
                 occ_thr=0.5, save_images=True, num_batches=None, num_samples=1):
    """Evaluate LDM model on test/validation dataset"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize DDIM sampler
    metric_fn = get_metric_fn(metric)
    sampler = DDIMSampler(model)
    
    results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", total=num_batches)):
            if num_batches is not None and batch_idx >= num_batches:
                break
            
            # Get model inputs z, c, x_gt, xrec, xc
            x_in, x_gt, x_occ_map, x_rel, y_rel, th_rel = model.get_input(
                                                            batch,
                                                            process = 'test')
            
            t0 = time.perf_counter()
            c, _ = model.get_encoding(x_in, x_gt, x_occ_map)

            # Expand conditioning for T frames
            seq_len = model.first_stage_model.seq_len
            c_exp = c.repeat_interleave(seq_len, dim=0)  # (N*T, 32, 16, 16)
            c_exp = c_exp.repeat_interleave(num_samples, dim=0)  # (N*T*num_samples, 32, 16, 16)


            # Measure sampling time
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            # DDIM sampling from random noise
            with model.ema_scope("Evaluation"):
                samples, _ = model.sample_log(
                    cond=c_exp,
                    batch_size=c_exp.shape[0],
                    ddim=True,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta
                )

            # samples shape: (N*T*num_samples, 2, 16, 16)
            # decode expects: (B*T, 2, 16, 16) where B=N, T=10

            # Decode latent to sequence
            pred_seq = model.decode_first_stage(samples)  # (num_samples, T, 1, H, W)
            
            # Reprojection for each time step
            prediction_maps = torch.zeros(num_samples, SEQ_LEN, 1, IMG_SIZE, IMG_SIZE, device=device)

            for t in range(SEQ_LEN):
                pred_map_t, _ = reprojection(pred_seq[:, t], x_rel[:, t], y_rel[:, t], th_rel[:, t], MAP_X_LIMIT, MAP_Y_LIMIT) # pred_seq[:, t] -> (num_samples, C, H, W)
                pred_map_t = pred_map_t.reshape(num_samples, 1, IMG_SIZE, IMG_SIZE)  # (num_samples, 1, 1, H, W)
                prediction_maps[:, t] = pred_map_t
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            
            total_time_ms = (t1 - t0) * 1000
            
            # Compute metrics per frame - average across all samples
            row = {"i": batch_idx, "Inference_time": float(total_time_ms)}
            
            # Top-K time-independent + Coverage (per time step)
            for t in range(SEQ_LEN):
                gt_t = x_gt[0, t]              # (H, W) or (1,H,W)
                pred_samples_t = prediction_maps[:, t]  # (K,1,H,W)

                # time-independent Top-K
                topk_ind_score, _, _ = compute_topk_time_independent(
                    pred_samples_t=pred_samples_t,
                    gt_t=gt_t,
                    metric_fn=metric_fn,
                    occ_thr=occ_thr,
                )

                row[f"n={t+1}_topk_{metric}_ind"] = float(topk_ind_score.item())

            # Top-K time-consistent
            gt_seq = x_gt[0]  # (T, H, W) or (T,1,H,W)

            _, best_sample_idx, _, best_per_t_scores = compute_topk_time_consistent(
                pred_samples=prediction_maps,
                gt_seq=gt_seq,
                metric_fn=metric_fn,
                occ_thr=occ_thr,
                reduce="mean",
            )

            for t in range(SEQ_LEN):
                row[f"n={t+1}_topk_{metric}_cons"] = float(best_per_t_scores[t])

            results.append(row)

            if (batch_idx + 1) % 100 == 0:
                pd.DataFrame(results).to_csv(csv_path, index=False)
            
            # -----------------------------------------------------------------
            # Image saving
            # -----------------------------------------------------------------
            if save_images:
                save_gt_maps(x_gt, output_dir, batch_idx, seq_len=SEQ_LEN)
                save_pred_mean_maps(prediction_maps, output_dir, batch_idx, seq_len=SEQ_LEN)
                save_pred_mean_overlay_maps(
                    prediction_maps=prediction_maps,
                    x_gt=x_gt,
                    output_dir=output_dir,
                    batch_idx=batch_idx,
                    occ_thr=occ_thr,
                    seq_len=SEQ_LEN,
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
        "--metric",
        type=str,
        default="iou",
        choices=["iou", "f1", "f2"],
        help="Metric used for Top-K evaluation",
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
