import torch
import torch.nn as nn
import torch.nn.functional as F


class AELoss(nn.Module):
    def __init__(
        self,
        logvar_init=0.0,
        kl_weight=0.001,
        stay_eps=0.01,
        lambda_base=1.0,
        lambda_stay=1.0,
        lambda_event=2.0,
        lambda_sign=0.5,
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.stay_eps = stay_eps
        self.lambda_stay = lambda_stay
        self.lambda_event = lambda_event
        self.lambda_sign = lambda_sign

        # kept for compatibility with previous configs/checkpoints
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

    def ae_loss(self, pred, target):
        """
        pred, target: (B, T, 1, H, W), continuous signed residual in [-1, 1].
        stay region  : |target| < stay_eps -> force pred to 0.
        event region : |target| >= stay_eps -> match signed residual magnitude/direction.
        """
        stay_mask = (target.abs() < self.stay_eps).float()
        event_mask = 1.0 - stay_mask

        base_loss = F.smooth_l1_loss(pred, target, reduction="mean")

        # stay: unchanged cells should stay exactly zero
        stay_loss = (stay_mask * pred.abs()).sum() / (stay_mask.sum() + 1e-6)

        # event: changed cells should match residual value
        per_pixel = F.smooth_l1_loss(pred, target, reduction="none")
        event_loss = (event_mask * per_pixel).sum() / (event_mask.sum() + 1e-6)

        # sign consistency: penalize opposite signed prediction on event cells
        sign_loss = (event_mask * F.relu(-pred * target)).sum() / (event_mask.sum() + 1e-6)

        data_loss = (
            self.lambda_base * base_loss
            + self.lambda_stay * stay_loss
            + self.lambda_event * event_loss
            + self.lambda_sign * sign_loss
        )

        logs = {
            "stay_loss": stay_loss.detach(),
            "event_loss": event_loss.detach(),
            "sign_loss": sign_loss.detach(),
        }

        return data_loss, logs
    
    def forward(self, inputs, reconstructions, posteriors, optimizer_idx=None, global_step=None, split="train"):
        """
        inputs, reconstructions: (B, T, C, H, W)
        posteriors: distribution object with .kl()
        """

        data_loss, extra_logs = self.ae_loss(reconstructions, inputs)

        kl_loss = posteriors.kl().mean()
        loss = data_loss + self.kl_weight * kl_loss

        log = {
            f"{split}/total_loss": loss.detach(),
            f"{split}/data_loss": data_loss.detach(),
            f"{split}/kl_loss": kl_loss.detach(),
        }
        for k, v in extra_logs.items():
            log[f"{split}/{k}"] = v

        return loss, log