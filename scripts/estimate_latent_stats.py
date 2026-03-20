import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from util import instantiate_from_config

# config load
config = OmegaConf.load("../configs/autoencoder/autoencoder_predocc.yaml")

# data root fix
config.data.params.train.params.data_root = "/home/ewhaglab/develop/data/OGM-datasets/OGM-Turtlebot2/train/"
config.data.params.validation.params.data_root = "/home/ewhaglab/develop/data/OGM-datasets/OGM-Turtlebot2/val/"

model = instantiate_from_config(config.model)
ckpt = torch.load(
    "/home/ewhaglab/develop/predocc_diffusion/models/first_stage_models/predocc_ae/last.ckpt",
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

model.estimate_latent_stats(dataloader, num_batches=128, use_mode=False)
