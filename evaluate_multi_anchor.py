import os, logging
import numpy as np
import matplotlib.pyplot as plt
import torch

import src.config as config
from src.config import setup_logging
from src.model import PoissonMLP, poisson_loss_nonlinear
from src.datasets import encode_condition_params
from src.devsim_datasource import DevsimDataSource, read_manifest, V_T
from src.plotting import (plot_phy_loss, plot_generalization_map)

logger = logging.getLogger(__name__)

setup_logging(level=logging.INFO)

os.makedirs(os.path.join(config.results_dir, "eval_multi_anchor"), exist_ok=True)

CHECKPOINT = os.path.join(config.results_dir, "devsim_pinn_multi_anchor.pth")

model = PoissonMLP(indim = config.indim, hiddim = config.hiddim, outdim = config.outdim, n_layers = config.n_layers, dropout = config.dropout).to(config.device)
model.load_state_dict(torch.load(CHECKPOINT, map_location = config.device))
model.eval()

test_files = read_manifest("data/records.csv", split="test")
logger.info(f"evaluating on {len(test_files)} held-out anchors")

summary = []

for anchor_file in test_files:
    tag = os.path.splitext(os.path.basename(anchor_file))[0]
    source = DevsimDataSource(anchor_file)

    x_test_physical = np.load(anchor_file)["x"]
    y_test = np.zeros_like(x_test_physical)
    z_test = np.zeros_like(x_test_physical)
    n_test = len(x_test_physical)

    x_test_norm = np.stack(source.coords_normalised(x_test_physical, y_test, z_test), axis = 1)
    psi_true = source.potential(x_test_physical, y_test, z_test)
    nD_test_np = source.density_normalised(x_test_physical, y_test, z_test)

    gamma_vals_test = np.full(n_test, source.gamma)
    nD_scale_vals_test = np.full(n_test, source.nD_peak_normalised)
    condition_params_test = np.stack([gamma_vals_test, nD_scale_vals_test], axis=1)
    cond_params_test = torch.tensor(condition_params_test, dtype=torch.float32).to(config.device)

    # predictions - plain forward pass, no differentiation needed
    X_test = torch.tensor(x_test_norm, dtype=torch.float32).to(config.device)
    X_test_packed = torch.cat([X_test, encode_condition_params(cond_params_test)], dim=1)
    with torch.inference_mode():
        psi_pred = model(X_test_packed).squeeze(-1).cpu().numpy()

    err_mV = (psi_pred - psi_true) * V_T * 1e3
    max_err = np.max(np.abs(err_mV))
    rms_err = np.sqrt(np.mean(err_mV**2))
    summary.append((tag, source.L, source.nD_peak_normalised, max_err, rms_err, source.psi_peak * V_T * 1e3))
    logger.info(f"{tag}: max|err|={max_err:.2f} mV  rms={rms_err:.2f} mV")

    # potential comparison plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_test_physical, psi_true * V_T * 1e3, label="Devsim (ground truth)", linewidth=2)
    ax.plot(x_test_physical, psi_pred * V_T * 1e3, label="PINN reconstruction",
            linewidth=1.5, linestyle="--")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("Potential (mV)")
    ax.set_title(f"{tag}\nmax|err|={max_err:.2f} mV, rms={rms_err:.2f} mV")
    ax.legend()
    filename = os.path.join(config.results_dir, f"eval_multi_anchor/{tag}_potential.png")
    fig.savefig(filename, dpi=150)
    plt.close(fig)

    # physics residual vs x - needs a fresh differentiable leaf, run outside
    # inference_mode (autograd.grad can't build a graph through inference tensors)
    X_test_grad = torch.tensor(x_test_norm, dtype=torch.float32, requires_grad=True).to(config.device)
    nD_test = torch.tensor(nD_test_np, dtype=torch.float32).unsqueeze(-1).to(config.device)
    residual = poisson_loss_nonlinear(
        model, X_test_grad, nD_test, cond_params_test, reduction="none", spatial_dims=(tuple(range(source.n_spatial_dims)))
    )
    residual_np = residual.detach().cpu().numpy().squeeze(-1)

    fig = plot_phy_loss(x_test_physical, residual_np)
    fig.axes[0].set_title(f"{tag} - physics residual vs x")
    filename = os.path.join(config.results_dir, f"eval_multi_anchor/{tag}_residual.png")
    fig.savefig(filename, dpi=150)
    plt.close(fig)

fig = plot_generalization_map(summary)
fig.savefig(os.path.join(config.results_dir, "generalization_map.png"), dpi = 150)
logger.info("summary (tag, L, nD_peak_normalised, max_err_mV, rms_err_mV):")
for row in summary:
    logger.info(row)


