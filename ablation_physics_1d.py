import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from functools import partial

import src.config as config
from src.model import PoissonMLP, poisson_loss_nonlinear
from src.train import train_model
from src.plotting import plot_loss_curves
from src.datasets import TrainingData, encode_condition_params
from src.devsim_datasource import DevsimDataSource, build_loaders, V_T

os.makedirs("results", exist_ok=True)

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)


def run_single_trajectory(datafile, lambda_phy, lambda_bc, suffix, seed):
    # Same seed reset right before model construction, and build_loaders draws from
    # its own numpy Generator (not torch's global RNG) - so w_physics/no_physics get
    # identical weight init and identical DataLoader shuffle/dropout draws throughout
    # training. lambda_phy is the only thing that actually differs between the runs.
    torch.manual_seed(seed)

    source = DevsimDataSource(datafile)
    print(f"[{suffix}] gamma = {source.gamma:.3e}, nD_peak_normalised = {source.nD_peak_normalised:.3e}")

    obs_loader, colloc_loader, bc_loader, val_X, val_y, val_cond = build_loaders(
        source, config.n_obs_per_anchor, config.n_colloc_per_anchor, config.n_bc,
        config.obs_batch, config.device
    )
    training_data = TrainingData(obs_loader, colloc_loader, bc_loader, val_X, val_y, val_cond)

    model = PoissonMLP(indim=config.indim, hiddim=config.hiddim, outdim=config.outdim,
                        n_layers=config.n_layers, dropout=config.dropout).to(config.device)
    crit = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=config.lr)
    physics_loss_fn = partial(poisson_loss_nonlinear, spatial_dims=tuple(range(source.n_spatial_dims)))

    t0 = time.time()
    model, loss_history = train_model(
        config.n_epochs, physics_loss_fn, training_data, model, crit,
        lambda_phy, lambda_bc, optim, early_stopping = config.early_stopping, patience=config.patience
    )
    print(f"[{suffix}] training time: {time.time() - t0:.1f} s")

    torch.save(model.state_dict(), f"results/devsim_pinn_1d_{suffix}.pth")

    fig = plot_loss_curves(loss_history)
    fig.savefig(f"results/devsim_pinn_loss_{suffix}.png")
    plt.close(fig)

    model.eval()
    with torch.inference_mode():
        x_cond_packed = torch.cat([training_data.val_X, encode_condition_params(training_data.val_cond)], dim=1)
        psi_pred = model(x_cond_packed).squeeze(-1).cpu().numpy()
    psi_truth = training_data.val_y.squeeze(-1).cpu().numpy()
    x_norm = training_data.val_X.detach().cpu().numpy().squeeze(-1)

    assert psi_pred.shape == psi_truth.shape, f"[{suffix}] shape mismatch: psi_pred {psi_pred.shape} vs psi_truth {psi_truth.shape}"
    assert psi_pred.shape == x_norm.shape, f"[{suffix}] shape mismatch: psi_pred {psi_pred.shape} vs x_norm {x_norm.shape}"

    x_um = (x_norm * source.L + source.x_min) * 1e4  # cm -> um

    err_mV = (psi_pred - psi_truth) * V_T * 1e3
    peak_mV = np.max(np.abs(psi_truth)) * V_T * 1e3
    max_err = np.max(np.abs(err_mV))
    rms_err = np.sqrt(np.mean(err_mV**2))

    print(f"[{suffix}] Peak potential: {peak_mV:.2f} mV")
    print(f"[{suffix}] Max absolute error: {max_err:.2f} mV")
    print(f"[{suffix}] RMS error: {rms_err:.2f} mV")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x_um, psi_truth * V_T * 1e3, label="DEVSIM ground truth", s=25)
    ax.scatter(x_um, psi_pred * V_T * 1e3, label="PINN reconstruction", s=25, marker="x")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("Potential (mV)")
    ax.set_title(f"1D p-n junction potential [{suffix}]")
    ax.legend()
    fig.savefig(f"results/devsim_pinn_potential_profile_{suffix}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x_um, err_mV, s=25)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("Error (mV)")
    ax.set_title(f"PINN reconstruction error [{suffix}] (max |err| = {max_err:.2f} mV)")
    fig.savefig(f"results/devsim_pinn_error_profile_{suffix}.png", dpi=150)
    plt.close(fig)

    return {"suffix": suffix, "max_err_mV": max_err, "rms_err_mV": rms_err, "peak_mV": peak_mV}


if __name__ == "__main__":
    # Heavy-doping anchor: nD_scale = 3.01e18/1.5e10 ~= 2.0e8, well above the ~1e7
    # threshold generalization_map.png flags as the hard corner - unlike the
    # light-doping anchor tried earlier, sparse uniform sampling here plausibly
    # misses the depletion transition, so physics has real work to do if the
    # per-trajectory mechanism is real.
    datafile = "data/devsim_diode_1d_diode_L5.98e-05_N3.01e+18_test.npz"
    seed = 123

    result_w = run_single_trajectory(datafile, lambda_phy=config.lambda_phy, lambda_bc=config.lambda_bc,
                                      suffix="w_physics", seed=seed)
    result_no = run_single_trajectory(datafile, lambda_phy=0.0, lambda_bc=config.lambda_bc,
                                       suffix="no_physics", seed=seed)

    print("\n--- Ablation summary ---")
    print(f"w_physics : max={result_w['max_err_mV']:.2f} mV, rms={result_w['rms_err_mV']:.2f} mV, peak={result_w['peak_mV']:.2f} mV")
    print(f"no_physics: max={result_no['max_err_mV']:.2f} mV, rms={result_no['rms_err_mV']:.2f} mV, peak={result_no['peak_mV']:.2f} mV")
