import torch
import torch.nn.functional as F
import math

eps = 1e-4

def reprojection(source_map, dx, dy, dtheta, x_lim, y_lim):
    B, C, H, W = source_map.shape
    device, dtype = source_map.device, source_map.dtype

    dx = dx.expand(B)
    dy = dy.expand(B)
    dtheta = dtheta.expand(B)

    x_min, x_max = x_lim
    y_min, y_max = y_lim

    # 1) prob -> logit
    source_logit = torch.logit(source_map.clamp(eps, 1 - eps))

    # 2) target grid centers (t+1 frame) in world coords
    res_x = (x_max - x_min) / H
    res_y = (y_max - y_min) / W

    ix = torch.arange(H, device=device)
    iy = torch.arange(W, device=device)
    ix_grid, iy_grid = torch.meshgrid(ix, iy, indexing='ij')  # (H,W)

    x_tg = x_min + (ix_grid.to(dtype) + 0.5) * res_x
    y_tg = y_min + (iy_grid.to(dtype) + 0.5) * res_y

    x_tg = x_tg.unsqueeze(0).expand(B, -1, -1)  # (B,H,W)
    y_tg = y_tg.unsqueeze(0).expand(B, -1, -1)

    # 3) reprojection : inverse transform to find source sampling locations
    ct = torch.cos(dtheta).view(B, 1, 1).to(dtype)
    st = torch.sin(dtheta).view(B, 1, 1).to(dtype)
    dxv = dx.view(B, 1, 1).to(dtype)
    dyv = dy.view(B, 1, 1).to(dtype)

    x_shift = x_tg + dxv
    y_shift = y_tg + dyv

    x_src =  ct * x_shift - st * y_shift
    y_src =  st * x_shift + ct * y_shift

    # --- 4) world -> normalized coords for grid_sample
    grid_x = 2.0 * (y_src - y_min) / (y_max - y_min) - 1.0  # -> W-axis
    grid_y = 2.0 * (x_src - x_min) / (x_max - x_min) - 1.0  # -> H-axis

    grid = torch.stack([grid_x, grid_y], dim=-1)  # (B,H,W,2)

    # valid mask (normalized range)
    valid = (grid_x >= -1.0) & (grid_x <= 1.0) & (grid_y >= -1.0) & (grid_y <= 1.0)
    valid_mask = valid.to(dtype).unsqueeze(1)  # (B,1,H,W)

    # 5) bilinear warp in logit space, then sigmoid back to prob
    warped_logit = F.grid_sample(source_logit, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    warped_prob = torch.sigmoid(warped_logit)

    warped_prob = warped_prob * valid_mask

    return warped_prob, valid_mask
