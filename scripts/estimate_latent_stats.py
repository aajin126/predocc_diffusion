
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import sys
import os
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Patch sys.modules so 'modules' points to 'predocc.modules' and 'util' to 'predocc.util'
import importlib

try:
    predocc_modules = importlib.import_module('predocc.modules')
    sys.modules['modules'] = predocc_modules
    predocc_util = importlib.import_module('predocc.util')
    sys.modules['util'] = predocc_util
    predocc_models = importlib.import_module('predocc.models')
    sys.modules['models'] = predocc_models
    predocc_data = importlib.import_module('predocc.data')
    sys.modules['data'] = predocc_data
except ImportError:
    pass

from predocc.util import instantiate_from_config


# config load

config = OmegaConf.load("../configs/autoencoder/autoencoder_predocc.yaml")
# Patch model target to use fully qualified module path if needed
if "target" in config.model and config.model["target"].startswith("models."):
    config.model["target"] = "predocc." + config.model["target"] 

# data root fix
config.data.params.train.params.data_root = "/home/oem/hj/data/OGM-datasets/OGM-Turtlebot2/train/"
config.data.params.validation.params.data_root = "/home/oem/hj/data/OGM-datasets/OGM-Turtlebot2/val/"

model = instantiate_from_config(config.model)
ckpt = torch.load(
    "/home/oem/hj/predocc_diffusion_v0/pretrained_models/first_stage_models/AE/v2.0.1/model.ckpt",
    map_location="cpu",
)
state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
model.load_state_dict(state_dict, strict=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

train_cfg = config.data.params.train
dataset = instantiate_from_config(train_cfg)
dataloader = DataLoader(
    dataset,
    batch_size=config.data.params.batch_size,
    num_workers=config.data.params.num_workers,
    shuffle=False,
)

model.estimate_latent_stats(dataloader, num_batches=500, use_mode=False)
