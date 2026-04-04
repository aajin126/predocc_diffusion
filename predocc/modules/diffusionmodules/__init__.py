
import torch
import logging
import numpy as np


def get_sigmas(config):

    device = getattr(config, 'device', None)

    T = getattr(config.model, 'num_classes')

    if config.model.sigma_dist == 'geometric':
        sigmas = torch.logspace(np.log10(config.model.sigma_begin), np.log10(config.model.sigma_end),
                                T)
        return sigmas.to(device) if device is not None else sigmas

    elif config.model.sigma_dist == 'linear':
        sigmas = torch.linspace(config.model.sigma_begin, config.model.sigma_end,
                                T)
        return sigmas.to(device) if device is not None else sigmas

    elif config.model.sigma_dist == 'cosine':
        t = torch.linspace(T, 0, T+1)/T
        s = 0.008
        f = torch.cos((t + s)/(1 + s) * np.pi/2)**2
        return f[:-1]/f[-1]

    else:
        raise NotImplementedError('sigma distribution not supported')

