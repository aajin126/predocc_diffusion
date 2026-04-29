import numpy as np
import torch
from torch.utils.data import DataLoader

from .local_occ_grid_map import LocalMap
from .dataloader import PredOccDataset

POINTS = 1080   # number of lidar points
IMG_SIZE = 64
SEQ_LEN = 10

P_prior = 0.5      # Prior occupancy probability
P_occ = 0.7        # Probability that cell is occupied with total confidence
P_free = 0.3       # Probability that cell is free with total confidence
MAP_X_LIMIT = [0.0, 6.4]
MAP_Y_LIMIT = [-3.2, 3.2]
RESOLUTION = 0.1
TRESHOLD_P_OCC = 0.8

def residual_sequence_to_ogm(ref_map, residual_sequence, mode):
    """
    residual_sequence:
        signed:
            (B, T, 1, H, W)
        appear_disappear:
            (B, T, 2, H, W)

    return:
        ogm_sequence: (B, T, 1, H, W)
    """

    cur = ref_map # (B,1,H,W)

    cur = cur.float()
    frames = [cur]

    if mode == "signed":
        for i in range(residual_sequence.shape[1]):
            delta = residual_sequence[:, i].float()   # (B,1,H,W)
            cur = (cur + delta).clamp(0.0, 1.0)


            frames.append(cur)

    elif mode == "appear_disappear":
        for i in range(residual_sequence.shape[1]):
            appear = residual_sequence[:, i, 0:1].float()
            disappear = residual_sequence[:, i, 1:2].float()

            cur = cur * (1.0 - disappear) + (1.0 - cur) * appear
            cur = cur.clamp(0.0, 1.0)

            frames.append(cur)

    else:
        raise ValueError(f"Unknown residual mode: {mode}")

    return torch.stack(frames, dim=1)[:, 1:]  # (B,T,1,H,W)
