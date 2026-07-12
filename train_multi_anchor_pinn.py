import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os, sys, logging
import time
from functools import partial
import src.config as config
from src.config import setup_logging


from src.model import PoissonMLP, poisson_loss_nonlinear
from src.train import train_model
from src.plotting import plot_loss_curves
from src.datasets import TrainingData
from src.devsim_datasource import read_manifest, build_multi_anchor_loaders, DevsimDataSource


torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

logger = logging.getLogger(__name__)

setup_logging(level=logging.INFO)

# load every train-split anchor named in the manifest
logger.info("=== reading training files ===")
train_files = read_manifest("data/records.csv", split="train")
logger.info(f"training on {len(train_files)} anchors")

obs_loader, colloc_loader, bc_loader, val_X, val_y, val_cond = build_multi_anchor_loaders(
    train_files, config.n_obs_per_anchor, config.n_colloc_per_anchor, config.obs_batch, config.device
)
training_data = TrainingData(obs_loader, colloc_loader, bc_loader, val_X, val_y, val_cond)

# train - indim=3 covers x + gamma + nD_scale, same as the single-anchor case
model = PoissonMLP(indim = config.indim, hiddim = config.hiddim, outdim = config.outdim, n_layers = config.n_layers, dropout = config.dropout).to(config.device)
crit  = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr = config.lr)

physics_loss_fn = partial(poisson_loss_nonlinear, spatial_dims = tuple(range(DevsimDataSource.n_spatial_dims)))

t0 = time.time()
model, loss_history = train_model(
    config.n_epochs, physics_loss_fn, training_data, model, crit, config.lambda_phy, config.lambda_bc, optim,
    early_stopping = config.early_stopping, patience = config.patience
)
train_time = time.time() - t0
logger.info(f"Training time: {train_time:.1f} s")

torch.save(model.state_dict(), os.path.join(config.results_dir, "devsim_pinn_multi_anchor.pth"))

fig = plot_loss_curves(loss_history)
fig.savefig(os.path.join(config.results_dir, "devsim_pinn_multi_anchor_loss.png"))
plt.close(fig)

logger.info("done - run evaluate_multi_anchor.py to check per-anchor accuracy and physics residual on the held-out test anchors")
