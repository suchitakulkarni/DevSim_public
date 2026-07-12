import numpy as np
import os, sys, logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import scipy
from scipy import special
from scipy.optimize import newton_krylov, NoConvergence
from scipy.interpolate import RegularGridInterpolator
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
from itertools import cycle 
from functools import partial

import src.config as config
from src.model import poisson_bc
from src.datasets import encode_condition_params

from src.config import setup_logging
logger = logging.getLogger(__name__)

setup_logging(level=logging.INFO)

# ------------------------- training loop  -------------------------
def train_model(n_epochs, physics_loss_fn, training_data, model, crit, lambda_phy, lambda_bc, optim, patience = 100, min_delta=0.0, early_stopping=True):
    # training_data is expected to expose: dataloader, colloc_loader, bc_loader,
    # val_X, val_y - val_X/val_y are the held-out validation split (never trained
    # on directly), used here only to decide when to stop, not to fit the model.
    # early_stopping=False trains the full n_epochs and returns whatever the final
    # epoch's weights are, with no checkpoint restoration - use this when val_loss
    # (pure data-fit MSE) isn't a trustworthy stopping signal for the problem at
    # hand (e.g. the cylinder pipeline, where sparse uniform val sampling barely
    # covers the doping ellipsoid and a tiny lambda_phy means the physics-driven
    # shaping of the solution only emerges over many more epochs than val_loss's
    # early plateau would suggest).
    history = {"MSELoss":[], "physics_loss":[], "bc_loss":[], "total_loss":[], "val_loss":[]}
    best_val_loss = float("inf")
    best_state = None
    epochs_since_improvement = 0

    for i in range(n_epochs):
        # all predefined tensors are on device already so I won't move them here. New tensors go to device
        model.train()
        total_loss = 0
        total_phy_loss = 0
        total_bc_loss = 0
        total_data_loss = 0
        for (x_batch, y_batch, cond_batch), (x_colloc, y_colloc, cond_colloc), (x_bc, y_bc, cond_bc) in zip(
            training_data.dataloader, cycle(training_data.colloc_loader), cycle(training_data.bc_loader)
        ):
            optim.zero_grad()
            x_batch_packed = torch.cat([x_batch, encode_condition_params(cond_batch)], dim = 1)
            #print(f'x_batch_packed = {x_batch_packed}')
            y_pred = model(x_batch_packed)
            #print(f'y_pred = {y_pred}')
            loss = crit(y_pred, y_batch)
            physics_loss = physics_loss_fn(model, x_colloc, y_colloc, cond_colloc)

            x_bc_packed = torch.cat([x_bc, encode_condition_params(cond_bc)], dim = 1)
            physics_loss_bc = poisson_bc(model, x_bc_packed, y_bc)

            combined_loss = loss + lambda_phy*physics_loss + lambda_bc*physics_loss_bc
            #print(f'combined_loss = {combined_loss}')
            total_loss += combined_loss.item()
            total_phy_loss += physics_loss.item()
            total_bc_loss += physics_loss_bc.item()
            total_data_loss += loss.item()
            combined_loss.backward()
            optim.step()
        total_loss = total_loss/len(training_data.dataloader)
        phy_loss = total_phy_loss/len(training_data.dataloader)
        bc_loss = total_bc_loss/len(training_data.dataloader)
        data_loss = total_data_loss/len(training_data.dataloader)

        model.eval()
        with torch.inference_mode():
            x_cond_packed = torch.cat([training_data.val_X, encode_condition_params(training_data.val_cond)], dim = 1)
            val_pred = model(x_cond_packed)
            val_loss = crit(val_pred, training_data.val_y).item()

        history["MSELoss"].append(data_loss)
        history["physics_loss"].append(phy_loss)
        history["bc_loss"].append(bc_loss)
        history["total_loss"].append(total_loss)
        history["val_loss"].append(val_loss)

        if early_stopping:
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

        if i % 20 == 0:
            logging.info(f"curret epoch = {i}, current loss = {total_loss:.3e}, phy_loss = {phy_loss:.3e}, mse_loss = {data_loss:0.3e}, val_loss = {val_loss:.3e}")


        if early_stopping and epochs_since_improvement >= patience:
            logging.info(f"Early stopping at epoch {i}: val_loss has not improved for {patience} epochs (best val_loss = {best_val_loss})")
            break

    if early_stopping and best_state is not None:
        model.load_state_dict(best_state) # restore the best-validation checkpoint, not just whatever the loop ended on

    return model, history
