import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os, sys
import time
from functools import partial
import src.config as config

from src.model import PoissonMLP, poisson_loss_nonlinear
from src.train import train_model
from src.plotting import plot_loss_curves, compare_predictions
from src.datasets import TrainingData, encode_condition_params
from src.devsim_datasource import DevsimDataSource, SlabGenerator, build_loaders, V_T

os.makedirs("results", exist_ok=True)

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
 
datafile = 'data/devsim_diode_1d_diode_L4.26e-05_N1.01e+15_test.npz'
# load devsim output
source = DevsimDataSource(datafile)
print(f"source.gamma = {source.gamma:.03e}")

# build loaders
#pinn_data, dataset, loader, colloc_loader, bc_loader = build_loaders(
#    source, n_obs, n_colloc, n_bc, obs_batch, device
#)

obs_loader, colloc_loader, bc_loader, dataset_val, pot_val, cond_val = build_loaders(
    source, config.n_obs_per_anchor, config.n_colloc_per_anchor, config.n_bc, config.obs_batch, config.device
)

#declare a cond_val tensor of gamma_val, nD_scal_val, here
training_data = TrainingData(obs_loader, colloc_loader, bc_loader, dataset_val, pot_val, cond_val)

# train
model = PoissonMLP(indim = config.indim, hiddim = config.hiddim, outdim = config.outdim, n_layers = config.n_layers, dropout = config.dropout).to(config.device)
crit  = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr = config.lr)

t0 = time.time()
print(f'self.nD_peak_normalised  = {source.nD_peak_normalised }')
#physics_loss_fn = partial(poisson_loss_nonlinear, gamma = source.gamma, nD_scale  = source.nD_peak_normalised )
physics_loss_fn = partial(poisson_loss_nonlinear, spatial_dims=tuple(range(source.n_spatial_dims)) ) 
model, loss_history = train_model(
    config.n_epochs, physics_loss_fn, training_data, model, crit, config.lambda_phy, config.lambda_bc,  optim, early_stopping = config.early_stopping, patience = config.patience)

train_time = time.time() - t0
print(f"Training time: {train_time:.1f} s")

torch.save(model.state_dict(), "results/devsim_pinn_1d.pth")

fig = plot_loss_curves(loss_history)
fig.savefig("results/devsim_pinn_loss.png")
plt.close(fig)

t1 = time.time()
model.eval()
with torch.inference_mode():
    #psi_pred = model(X_test_packed).squeeze(-1).cpu().numpy()
    x_cond_packed = torch.cat([training_data.val_X, encode_condition_params(training_data.val_cond)], dim = 1)
    psi_pred = model(x_cond_packed).squeeze(-1).cpu().numpy()

psi_truth = training_data.val_y.squeeze(-1).cpu().numpy()
x_truth = training_data.val_X.detach().squeeze(-1).cpu().numpy()
assert psi_pred.shape == psi_truth.shape, f"psi_pred {psi_pred.shape} vs psi_truth {psi_truth.shape} shape mismatch"
assert psi_pred.shape == x_truth.shape, f"psi_pred {psi_pred.shape} vs psi_truth {x_truth.shape} shape mismatch"
inference_time = time.time() - t1
print(f"Inference time on {len(training_data.val_X.detach().cpu().numpy())} nodes: {inference_time*1000:.2f} ms")

print(f'psi_pred.shape = {psi_pred.shape}')
print(f'psi_truth.shape = {psi_truth.shape}')

fig = compare_predictions(psi_pred, training_data.val_y.cpu().numpy())
fig.savefig("results/devsim_pinn_comparison_testset.png")
plt.close(fig)

# potential profile plot
fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(x_truth, psi_truth,  label = "DevSim ground truth", linewidth=2)
ax.scatter(x_truth, psi_pred,  label = "PINN reconstruction", linewidth=2)
ax.set_xlabel("x (AU)")
ax.set_ylabel("Potential (AU)")
ax.set_title("1D p-n junction potential: Devsim vs PINN")
ax.legend()
fig.savefig("results/devsim_pinn_potential_profile.png", dpi=150)
plt.close(fig)

# error profile
fig, ax = plt.subplots(figsize=(7, 4))
err_mV = (psi_pred - psi_truth) * V_T * 1e3
ax.scatter(x_truth, err_mV)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("x (AU)")
ax.set_ylabel("Error (AU)")
ax.set_title(f"PINN reconstruction error  (max |err| = {np.max(np.abs(err_mV)):.2f} AU)")
fig.savefig("results/devsim_pinn_error_profile.png", dpi=150)
plt.close(fig)

print(f"Max absolute error: {np.max(np.abs(err_mV)):.2f} mV")
print(f"RMS error: {np.sqrt(np.mean(err_mV**2)):.2f} mV")
print(f"Peak potential: {np.max(np.abs(psi_truth)) * V_T * 1e3:.2f} mV")
