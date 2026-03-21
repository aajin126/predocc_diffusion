import torch
import torch.nn as nn
from taming.modules.losses.vqperceptual import NLayerDiscriminator, weights_init, hinge_d_loss, vanilla_d_loss, adopt_weight

class AELoss(nn.Module):
    def __init__(
        self,
        logvar_init=0.0,
        kl_weight=0.001,
        disc_start=5000,
        disc_num_layers=3,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        disc_loss="hinge"
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        # Adversarial loss 
        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers
        ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(
        self,
        inputs,
        reconstructions,
        posteriors,
        optimizer_idx,
        global_step,
        last_layer=None,
        split="train"
    ):
        """
        inputs, reconstructions: (B, T, C, H, W)
        posteriors: distribution object with .kl()
        optimizer_idx: 0(generator), 1(discriminator)
        """

        seq_len = 10
        B = inputs.shape[0]

        ce_loss = 0.0

        for k in range(seq_len):
            ce_loss = ce_loss + torch.nn.functional.binary_cross_entropy(
                reconstructions[:, k],
                inputs[:, k],
                reduction="sum"
            ).div(B)
        ce_loss = ce_loss / seq_len
        kl_loss = posteriors.kl().mean()

        # Generator update
        if optimizer_idx == 0:
            # (B, T, C, H, W) -> (B*T, C, H, W)
            logits_fake = self.discriminator(reconstructions.reshape(-1, *reconstructions.shape[2:]))
            g_loss = -torch.mean(logits_fake)
            if torch.is_grad_enabled():
                d_weight = self.calculate_adaptive_weight(ce_loss, g_loss, last_layer)
            else:
                d_weight = torch.tensor(0.0, device=ce_loss.device)
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = ce_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss
            log = {
                f"{split}/total_loss": loss.detach(),
                f"{split}/ce_loss": ce_loss.detach(),
                f"{split}/kl_loss": kl_loss.detach(),
                f"{split}/g_loss": g_loss.detach(),
                f"{split}/d_weight": d_weight.detach(),
                f"{split}/disc_factor": torch.tensor(disc_factor),
            }
            return loss, log

        # Discriminator update
        if optimizer_idx == 1:
            logits_real = self.discriminator(inputs.detach().reshape(-1, *inputs.shape[2:]))
            logits_fake = self.discriminator(reconstructions.detach().reshape(-1, *reconstructions.shape[2:]))
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
            log = {
                f"{split}/disc_loss": d_loss.detach(),
                f"{split}/logits_real": logits_real.detach().mean(),
                f"{split}/logits_fake": logits_fake.detach().mean(),
            }
            return d_loss, log