from types import SimpleNamespace

import torch.nn as nn

from modules.diffusionmodules.better.ncsnpp_more import UNetMore_DDPM


def _to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _to_namespace(val) for key, val in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_namespace(item) for item in value)
    return value


class UNetModel(nn.Module):
    """
    Thin adapter that accepts a single MVCD-style config object and forwards it
    to the original UNetMore_DDPM constructor.
    """

    def __init__(self, config, **kwargs):
        super().__init__()

        del kwargs

        config = _to_namespace(config)
        if not hasattr(config, "model") or not hasattr(config, "data"):
            raise ValueError("config must contain both 'model' and 'data' sections.")
        if getattr(config.model, "spade", False) is not True:
            raise ValueError("predocc_unet.UNetModel expects config.model.spade=True.")

        self.model = UNetMore_DDPM(config)

    def forward(self, x, timesteps=None, context=None, y=None, cond=None, **kwargs):
        del context, y, kwargs
        return self.model(x, timesteps, cond=cond)