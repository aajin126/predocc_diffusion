import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', )))
sys.path.append("/home/ewhaglab/develop/predocc_diffusion/predocc")
from predocc.util import instantiate_from_config

# config load
config = OmegaConf.load("../configs/autoencoder/autoencoder_predocc.yaml")

# data root fix
config.data.params.train.params.data_root = "/home/ewhaglab/develop/data/OGM-datasets/OGM-Turtlebot2/train/"
config.data.params.validation.params.data_root = "/home/ewhaglab/develop/data/OGM-datasets/OGM-Turtlebot2/val/"

model = instantiate_from_config(config.model)
ckpt = torch.load(
    "/home/ewhaglab/develop/predocc_diffusion/pretrained_models/first_stage_models/AE/v7.1/model.ckpt",
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
