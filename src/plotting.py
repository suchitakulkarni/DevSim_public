import numpy as np
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
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

# ------------------------- Plot styling: charge density/potential are signed
# (charges can be +/-), so they use a diverging colormap, not a sequential one.
DIVERGING_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "poisson_diverging", ["#e34948", "#f0efec", "#2a78d6"]
)
CHARGE_COLOR_POS = "#2a78d6"  # blue - matches the positive pole of DIVERGING_CMAP
CHARGE_COLOR_NEG = "#e34948"  # red  - matches the negative pole of DIVERGING_CMAP

# Field magnitude |E| is unsigned, so it gets its own single-hue sequential
# ramp (aqua) rather than reusing the diverging blue/red, which are already
# spoken for as the charge-sign colors in the same figures.
SEQUENTIAL_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "poisson_sequential_field", ["#f0efec", "#1baf7a"]
)

def _diverging_norm(values):
    vmax = np.max(np.abs(values))
    vmax = vmax if vmax > 0 else 1.0
    return mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

def plot_phy_loss(x, loss):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.scatter(x, loss, label = "Physics loss")
    ax.set_xlabel("x")
    ax.set_ylabel("Physics Loss")
    ax.legend()
    return fig

def plot_loss_curves(loss_history):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.plot(loss_history["MSELoss"], label = "MSE Loss")
    ax.plot(loss_history["physics_loss"], label = "Physics loss")
    ax.plot(loss_history["bc_loss"], label = "BC loss")
    ax.plot(loss_history["total_loss"], label = "total loss")
    ax.set_yscale("log")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    return fig

def compare_predictions(y_pred, y_data):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.scatter(y_data, y_pred)
    ax.set_xlabel("Predictions from data")
    ax.set_ylabel("Predictions from model")
    return fig


def compare_predictions_vs_r(r, y_pred, y_data):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.scatter(r, y_data, label = 'Observed')
    ax.scatter(r, y_pred, label = 'Predicted')
    ax.set_xlabel("r")
    ax.set_ylabel("Potential")
    ax.legend()
    return fig

def plot_uncertainty_band(x, mean, std, n_std=1.0):
    """Prediction curve (e.g. MC-dropout mean) with a shaded uncertainty band
    (+/- n_std*std) around it. x, mean, std must be the same length - squeezes
    any trailing singleton dims itself, but does not sort for you unless x is
    already ordered... actually sorts by x internally, since fill_between
    needs points in x-order or the shaded region self-intersects."""
    x = np.asarray(x).squeeze()
    mean = np.asarray(mean).squeeze()
    std = np.asarray(std).squeeze()

    order = np.argsort(x)
    x_s, mean_s, std_s = x[order], mean[order], std[order]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.fill_between(x_s, mean_s - n_std * std_s, mean_s + n_std * std_s,
                     color="#89CFF0", alpha=0.4, label=rf"uncertainty ($\pm{n_std:g}\sigma$)")
    ax.plot(x_s, mean_s, color="#1f6fa8", linewidth=1.5, label="prediction (mean)")
    ax.set_xlabel("x")
    ax.set_ylabel("Potential")
    ax.legend()
    return fig

# ------------------------- Point-charge visualizations -------------------------
def plot_3d_config(charges, x, y, z, values, value_label="charge density", point_size=6):
    """3D scatter of the given evaluation points colored by density/potential
    (diverging, since the field is signed), with the charges themselves overlaid
    as larger markers colored by sign. Call this with any point cloud (training,
    collocation, or validation) against the same `charges` object."""
    norm = _diverging_norm(values)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x, y, z, c=values, cmap=DIVERGING_CMAP, norm=norm,
                     s=point_size, alpha=0.5, linewidths=0)

    ax.scatter(charges.xi, charges.yi, charges.zi, c=_charge_marker_colors(charges), s=160,
               edgecolors="black", linewidths=1.2, label="charges")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect([1, 1, 1])
    fig.colorbar(sc, ax=ax, shrink=0.6, label=value_label)
    return fig

def _cross_section_grid(charges, rad, axis="z", slice_value=None, n_grid=200):
    """Shared grid setup for cross-section plots: builds the disk-shaped slice
    through the sphere and the flat (xg, yg, zg) query points on it, plus
    everything needed to reshape/mask/overlay results computed on that grid."""
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be 'x', 'y', or 'z'")

    charge_coord = {"x": charges.xi, "y": charges.yi, "z": charges.zi}[axis]
    if slice_value is None:
        slice_value = np.mean(charge_coord)

    r_slice = np.sqrt(max(rad**2 - slice_value**2, 0.0))
    u = np.linspace(-r_slice, r_slice, n_grid)
    v = np.linspace(-r_slice, r_slice, n_grid)
    U, V = np.meshgrid(u, v)
    mask = U**2 + V**2 <= r_slice**2

    other_axes = [a for a in ("x", "y", "z") if a != axis]
    coords = {other_axes[0]: U, other_axes[1]: V, axis: np.full_like(U, slice_value)}
    xg, yg, zg = coords["x"].ravel(), coords["y"].ravel(), coords["z"].ravel()
    return u, v, U, V, mask, r_slice, xg, yg, zg, other_axes, charge_coord, slice_value

def _overlay_charge_markers(ax, charges, charge_coord, slice_value, rad, other_axes):
    """Every charge is shown, always - a hard "near the slice" cutoff can end up
    excluding every single charge (common with only a few of them) and silently
    produce a plot with no markers at all. Instead, fade by distance from the slice."""
    dist_from_slice = np.abs(charge_coord - slice_value)
    charge_alphas = np.clip(1.0 - 0.8 * dist_from_slice / rad, 0.2, 1.0)
    proj = {"x": charges.xi, "y": charges.yi, "z": charges.zi}
    ax.scatter(proj[other_axes[0]], proj[other_axes[1]],
               c=_charge_marker_colors(charges, charge_alphas),
               s=140, edgecolors="black", linewidths=1.2, zorder=5)

def plot_cross_section(charges, rad, quantity="density", axis="z", slice_value=None, n_grid=200):
    """2D diverging heatmap of density or potential on a disk-shaped cross-section
    of the sphere, with the charges near that slice projected onto it."""
    u, v, U, V, mask, r_slice, xg, yg, zg, other_axes, charge_coord, slice_value = \
        _cross_section_grid(charges, rad, axis, slice_value, n_grid)

    if quantity == "density":
        vals = charges.density(xg, yg, zg)
        label = r"charge density $\rho$"
    elif quantity == "potential":
        vals = charges.potential(xg, yg, zg)
        label = r"potential $\phi$"
    else:
        raise ValueError("quantity must be 'density' or 'potential'")

    vals = vals.reshape(U.shape)
    vals = np.where(mask, vals, np.nan)

    norm = _diverging_norm(vals[mask])
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(vals, extent=[-r_slice, r_slice, -r_slice, r_slice], origin="lower",
                   cmap=DIVERGING_CMAP, norm=norm)
    fig.colorbar(im, ax=ax, label=label)
    _overlay_charge_markers(ax, charges, charge_coord, slice_value, rad, other_axes)

    ax.set_xlabel(other_axes[0])
    ax.set_ylabel(other_axes[1])
    ax.set_title(f"{axis} = {slice_value:.2f} cross-section")
    ax.set_aspect("equal")
    return fig

def _charge_marker_colors(charges, alphas=None):
    """Shared helper: charges colored by sign (blue +, red -), matching
    CHARGE_COLOR_POS/NEG, optionally faded by a per-charge alpha."""
    if alphas is None:
        alphas = np.ones(charges.n_charges)
    return [
        mcolors.to_rgba(CHARGE_COLOR_POS if q > 0 else CHARGE_COLOR_NEG, alpha=a)
        for q, a in zip(charges.qi, alphas)
    ]

def plot_field_lines(charges, rad, axis="z", slice_value=None, n_grid=200, density=1.2):
    """Streamline plot of the electric field on the same disk-shaped
    cross-section used by plot_cross_section: line direction/curvature traces
    the in-plane field, line color/width encode |E| (sequential, since
    magnitude is unsigned), and nearby charges are overlaid as in the other
    cross-section plot."""
    u, v, U, V, mask, r_slice, xg, yg, zg, other_axes, charge_coord, slice_value = \
        _cross_section_grid(charges, rad, axis, slice_value, n_grid)

    Ex, Ey, Ez = charges.electric_field(xg, yg, zg)
    E = {"x": Ex, "y": Ey, "z": Ez}
    E_u = E[other_axes[0]].reshape(U.shape)
    E_v = E[other_axes[1]].reshape(U.shape)
    E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2).reshape(U.shape)

    # Line width by magnitude computed before masking (so it stays NaN-free);
    # cells outside the disk are then dropped by feeding streamplot NaN u/v,
    # which it treats as blocked rather than integrating through.
    lw = 0.5 + 2.0 * E_mag / E_mag.max()
    E_u = np.where(mask, E_u, np.nan)
    E_v = np.where(mask, E_v, np.nan)
    E_mag = np.where(mask, E_mag, np.nan)

    fig, ax = plt.subplots(figsize=(6, 6))
    strm = ax.streamplot(u, v, E_u, E_v, color=E_mag, cmap=SEQUENTIAL_CMAP,
                          linewidth=lw, density=density, arrowsize=1.2)
    fig.colorbar(strm.lines, ax=ax, label=r"field magnitude $|E|$")
    _overlay_charge_markers(ax, charges, charge_coord, slice_value, rad, other_axes)

    ax.set_xlabel(other_axes[0])
    ax.set_ylabel(other_axes[1])
    ax.set_title(f"electric field lines, {axis} = {slice_value:.2f} cross-section")
    ax.set_xlim(-r_slice, r_slice)
    ax.set_ylim(-r_slice, r_slice)
    ax.set_aspect("equal")
    return fig

# ------------------------- PINN reconstruction vs. analytic charges -------------------------
def _model_fields_on_grid(model, rad, rho0, epsilon_physical, xg, yg, zg):
    """Evaluate the trained PINN on a physical-unit point cloud (xg, yg, zg) and
    rescale its output, gradient, and Laplacian back to physical units, so the
    reconstruction can be compared directly against the analytic `charges`
    potential/density/field on the same points. Mirrors the forward pass and
    autograd derivatives used in poisson_loss, but run on a plotting grid
    instead of the collocation points, and without building a training graph."""
    phi0 = rho0 * rad**2 / epsilon_physical

    X = torch.tensor(np.stack([xg, yg, zg], axis=1), dtype=torch.float32, device = config.device)
    X = (X / rad).requires_grad_(True)

    u = model(X)
    grad_u = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    d2u_dx2 = torch.autograd.grad(grad_u[:, 0].sum(), X, create_graph=True)[0][:, 0]
    d2u_dy2 = torch.autograd.grad(grad_u[:, 1].sum(), X, create_graph=True)[0][:, 1]
    d2u_dz2 = torch.autograd.grad(grad_u[:, 2].sum(), X, create_graph=True)[0][:, 2]
    laplacian = (d2u_dx2 + d2u_dy2 + d2u_dz2).detach()

    # phi = phi0 * u(x/rad); rho = -rho0 * laplacian_xtilde(u); E = -(phi0/rad) * grad_xtilde(u)
    phi_model = (u.detach().squeeze(-1) * phi0).cpu().numpy()
    rho_model = (-laplacian * rho0).cpu().numpy()
    field_scale = -(phi0 / rad)
    Ex_model = (grad_u[:, 0].detach() * field_scale).cpu().numpy()
    Ey_model = (grad_u[:, 1].detach() * field_scale).cpu().numpy()
    Ez_model = (grad_u[:, 2].detach() * field_scale).cpu().numpy()
    return phi_model, rho_model, Ex_model, Ey_model, Ez_model

def plot_reconstructed_cross_section(model, charges, rad, rho0, epsilon_physical,
                                      quantity="potential", axis="z", slice_value=None, n_grid=200):
    """Same disk-shaped cross-section as plot_cross_section, shown as three
    panels: the analytic ground truth, the trained model's reconstruction (same
    color scale as the ground truth, so the two are directly comparable), and
    their percent difference. The percent difference is taken relative to the
    ground truth's peak magnitude on this slice rather than pointwise, since
    density truly vanishes over most of the domain and a pointwise percent
    error would blow up to +-inf there."""
    u, v, U, V, mask, r_slice, xg, yg, zg, other_axes, charge_coord, slice_value = \
        _cross_section_grid(charges, rad, axis, slice_value, n_grid)

    phi_model, rho_model, *_ = _model_fields_on_grid(model, rad, rho0, epsilon_physical, xg, yg, zg)

    if quantity == "density":
        true_vals = charges.density(xg, yg, zg)
        model_vals = rho_model
        label = r"charge density $\rho$"
    elif quantity == "potential":
        true_vals = charges.potential(xg, yg, zg)
        model_vals = phi_model
        label = r"potential $\phi$"
    else:
        raise ValueError("quantity must be 'density' or 'potential'")

    true_vals = np.where(mask.ravel(), true_vals, np.nan).reshape(U.shape)
    model_vals = np.where(mask.ravel(), model_vals, np.nan).reshape(U.shape)
    pct_diff = (true_vals - model_vals) / np.nanmax(np.abs(true_vals)) * 100

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    shared_norm = _diverging_norm(np.concatenate([true_vals[mask], model_vals[mask]]))
    im0 = axes[0].imshow(true_vals, extent=[-r_slice, r_slice, -r_slice, r_slice],
                          origin="lower", cmap=DIVERGING_CMAP, norm=shared_norm)
    fig.colorbar(im0, ax=axes[0], label=f"ground truth {label}")
    _overlay_charge_markers(axes[0], charges, charge_coord, slice_value, rad, other_axes)
    axes[0].set_title(f"ground truth, {axis} = {slice_value:.2f}")

    im1 = axes[1].imshow(model_vals, extent=[-r_slice, r_slice, -r_slice, r_slice],
                          origin="lower", cmap=DIVERGING_CMAP, norm=shared_norm)
    fig.colorbar(im1, ax=axes[1], label=f"reconstructed {label}")
    _overlay_charge_markers(axes[1], charges, charge_coord, slice_value, rad, other_axes)
    axes[1].set_title(f"reconstructed, {axis} = {slice_value:.2f}")

    pct_norm = _diverging_norm(pct_diff[mask])
    im2 = axes[2].imshow(pct_diff, extent=[-r_slice, r_slice, -r_slice, r_slice],
                          origin="lower", cmap=DIVERGING_CMAP, norm=pct_norm)
    fig.colorbar(im2, ax=axes[2], label=f"% difference (of peak {label})")
    _overlay_charge_markers(axes[2], charges, charge_coord, slice_value, rad, other_axes)
    axes[2].set_title(f"difference, {axis} = {slice_value:.2f}")

    for a in axes:
        a.set_xlabel(other_axes[0])
        a.set_ylabel(other_axes[1])
        a.set_aspect("equal")
    return fig

def plot_reconstructed_field_lines(model, charges, rad, rho0, epsilon_physical,
                                    axis="z", slice_value=None, n_grid=200, density=1.2):
    """Same disk cross-section as plot_field_lines, but the field lines are
    traced from -grad(model potential) instead of the analytic charges."""
    u, v, U, V, mask, r_slice, xg, yg, zg, other_axes, charge_coord, slice_value = \
        _cross_section_grid(charges, rad, axis, slice_value, n_grid)

    _, _, Ex_model, Ey_model, Ez_model = _model_fields_on_grid(
        model, rad, rho0, epsilon_physical, xg, yg, zg)

    E = {"x": Ex_model, "y": Ey_model, "z": Ez_model}
    E_u = E[other_axes[0]].reshape(U.shape)
    E_v = E[other_axes[1]].reshape(U.shape)
    E_mag = np.sqrt(Ex_model**2 + Ey_model**2 + Ez_model**2).reshape(U.shape)

    lw = 0.5 + 2.0 * E_mag / E_mag.max()
    E_u = np.where(mask, E_u, np.nan)
    E_v = np.where(mask, E_v, np.nan)
    E_mag = np.where(mask, E_mag, np.nan)

    fig, ax = plt.subplots(figsize=(6, 6))
    strm = ax.streamplot(u, v, E_u, E_v, color=E_mag, cmap=SEQUENTIAL_CMAP,
                          linewidth=lw, density=density, arrowsize=1.2)
    fig.colorbar(strm.lines, ax=ax, label=r"reconstructed field magnitude $|E|$")
    _overlay_charge_markers(ax, charges, charge_coord, slice_value, rad, other_axes)

    ax.set_xlabel(other_axes[0])
    ax.set_ylabel(other_axes[1])
    ax.set_title(f"reconstructed field lines, {axis} = {slice_value:.2f} cross-section")
    ax.set_xlim(-r_slice, r_slice)
    ax.set_ylim(-r_slice, r_slice)
    ax.set_aspect("equal")
    return fig

def plot_field_quiver_3d(charges, rad, n_points=250, seed=123, length_scale=0.2):
    """3D quiver of the electric field, sampled on a fresh volume-uniform point
    cloud (denser scatter clouds make quiver arrows unreadable), colored by
    |E| with the same sequential ramp as plot_field_lines, with charges
    overlaid as in plot_3d_config."""
    gen = SphereGenerator(radius=rad, n_points=n_points)
    _, _, _, x, y, z = gen.sample_volume_uniform(seed=seed)
    Ex, Ey, Ez = charges.electric_field(x, y, z)
    E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)

    norm = mcolors.Normalize(vmin=0.0, vmax=E_mag.max())
    sm = plt.cm.ScalarMappable(cmap=SEQUENTIAL_CMAP, norm=norm)
    sm.set_array([])

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.quiver(x, y, z, Ex, Ey, Ez, colors=SEQUENTIAL_CMAP(norm(E_mag)),
              length=length_scale, normalize=True, linewidth=1.0)
    ax.scatter(charges.xi, charges.yi, charges.zi,
               c=_charge_marker_colors(charges), s=160,
               edgecolors="black", linewidths=1.2, label="charges")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect([1, 1, 1])
    fig.colorbar(sm, ax=ax, shrink=0.6, label=r"field magnitude $|E|$")
    return fig

def animate_rotating_view(fig, filename, n_frames=120, elev=20, fps=24):
    """Steady-state visualizations here have no time axis to animate - nothing
    is advancing - so the only honest animation is rotating the camera around
    the static field/charge configuration. Spins every 3D axes in the given
    figure through a full azimuth turn in sync (so multi-panel figures, e.g.
    ground-truth/reconstructed side by side, rotate together rather than just
    the first panel) and saves it as a GIF (Pillow writer, no ffmpeg
    dependency). Filters to axes with view_init (3D only) so an attached
    colorbar's 2D axes - which also lives in fig.axes - is skipped rather than
    crashing on a method it doesn't have."""
    axes_3d = [ax for ax in fig.axes if hasattr(ax, "view_init")]

    def _update(frame):
        azim = frame * 360.0 / n_frames
        for ax in axes_3d:
            ax.view_init(elev=elev, azim=azim)
        return ()

    anim = animation.FuncAnimation(fig, _update, frames=n_frames, blit=False)
    anim.save(filename, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)

# ------------------------- CylinderCharges (nonlinear PB) visualization -------------------------
def plot_cylindercharges_solution(cyl):
    """2-panel (r,z) cross-section of the solved CylinderCharges state: doping
    density nD (sequential cmap, since nD >= 0 by construction) and the solved
    potential psi (diverging cmap, since psi can be either sign), mirrored
    across r=0 so the axisymmetric slice reads as a full disk cross-section
    rather than a half-plane. Reuses the exact (rtilde, ztilde) solver grid -
    no separate re-sampling/interpolation query needed for the density panel,
    and the potential panel queries the interpolator only at those same grid
    nodes (exact, not an approximation)."""
    r_half = cyl.rtilde
    z_half = cyl.ztilde
    r_full = np.concatenate([-r_half[:0:-1], r_half]) * cyl.rad
    z_full = z_half * cyl.rad
    extent = [z_full.min(), z_full.max(), r_full.min(), r_full.max()]

    nD_full = np.concatenate([cyl.nD_grid[:0:-1], cyl.nD_grid], axis=0)

    psi_grid = cyl._psi_interp(np.stack([cyl.rvals, cyl.zvals], axis=-1))
    psi_full = np.concatenate([psi_grid[:0:-1], psi_grid], axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    nD_norm = mcolors.Normalize(vmin=0.0, vmax=nD_full.max())
    im0 = axes[0].imshow(nD_full, extent=extent, origin="lower", cmap=SEQUENTIAL_CMAP, norm=nD_norm)
    fig.colorbar(im0, ax=axes[0], label=r"doping density $\tilde{n}_D$")
    axes[0].set_title("doping density")

    psi_norm = _diverging_norm(psi_full)
    im1 = axes[1].imshow(psi_full, extent=extent, origin="lower", cmap=DIVERGING_CMAP, norm=psi_norm)
    fig.colorbar(im1, ax=axes[1], label=r"potential $\psi$")
    axes[1].set_title("solved potential")

    for ax in axes:
        ax.add_patch(mpatches.Circle((0, 0), cyl.rad, fill=False, edgecolor="black",
                                      linestyle="--", linewidth=1, label="outer boundary"))
        ax.add_patch(mpatches.Ellipse((cyl.z0, 0), width=2 * cyl.az, height=2 * cyl.ar,
                                       fill=False, edgecolor="black", linestyle=":", linewidth=1,
                                       label="doping ellipsoid"))
        ax.set_xlabel("z")
        ax.set_ylabel("r")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")

    fig.suptitle(rf"CylinderCharges solved state ($\gamma$ = {cyl.gamma})")
    return fig

def plot_cylindercharges_reconstruction(model, cyl, n_grid=200):
    """Cross-section comparison of the trained model's reconstructed potential
    against CylinderCharges' own solved ground truth: three panels (ground
    truth, reconstruction, percent difference), mirrored across r=0 like
    plot_cylindercharges_solution. Queried on the phi=0 half-plane (y=0) -
    the true solution is axisymmetric, so any phi slice is identical, matching
    how CylinderCharges' own nD_grid/_solve() already work on that same slice.
    Ground truth and model output are both converted from the dimensionless
    PB potential psi=phi/V_T to physical volts (cyl.potential_physical()/V_T
    respectively) - this is a presentation-only rescale, psi is still what the
    model was actually trained on and what the physics loss operates on.
    Call model.eval() before calling this."""
    r_half = np.linspace(0, 1, n_grid)
    z_half = np.linspace(-1, 1, n_grid)
    rr, zz = np.meshgrid(r_half, z_half, indexing="ij")
    mask = (rr**2 + zz**2) <= 1  # box corners beyond the unit sphere were never part of the training/physical domain

    xg = (rr * cyl.rad).ravel()
    yg = np.zeros_like(xg)
    zg = (zz * cyl.rad).ravel()

    phi_true = cyl.potential_physical(xg, yg, zg).reshape(rr.shape)

    X = torch.tensor(np.stack([xg, yg, zg], axis=1), dtype=torch.float32, device = config.device) / cyl.rad
    with torch.inference_mode():
        phi_model = (model(X).squeeze(-1).cpu().numpy() * cyl.V_T).reshape(rr.shape)

    r_full = np.concatenate([-r_half[:0:-1], r_half]) * cyl.rad
    z_full = z_half * cyl.rad
    extent = [z_full.min(), z_full.max(), r_full.min(), r_full.max()]

    mask_full = np.concatenate([mask[:0:-1], mask], axis=0)
    phi_true_full = np.where(mask_full, np.concatenate([phi_true[:0:-1], phi_true], axis=0), np.nan)
    phi_model_full = np.where(mask_full, np.concatenate([phi_model[:0:-1], phi_model], axis=0), np.nan)
    pct_diff = (phi_true_full - phi_model_full) / np.nanmax(np.abs(phi_true_full)) * 100

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    shared_norm = _diverging_norm(np.concatenate([phi_true_full[mask_full], phi_model_full[mask_full]]))
    im0 = axes[0].imshow(phi_true_full, extent=extent, origin="lower", cmap=DIVERGING_CMAP, norm=shared_norm)
    fig.colorbar(im0, ax=axes[0], label=r"ground truth potential $\phi$ (V)")
    axes[0].set_title("ground truth")

    im1 = axes[1].imshow(phi_model_full, extent=extent, origin="lower", cmap=DIVERGING_CMAP, norm=shared_norm)
    fig.colorbar(im1, ax=axes[1], label=r"reconstructed potential $\phi$ (V)")
    axes[1].set_title("reconstructed")

    pct_norm = _diverging_norm(pct_diff[mask_full])
    im2 = axes[2].imshow(pct_diff, extent=extent, origin="lower", cmap=DIVERGING_CMAP, norm=pct_norm)
    fig.colorbar(im2, ax=axes[2], label=r"% difference (of peak $\phi$)")
    axes[2].set_title("difference")

    for ax in axes:
        ax.add_patch(mpatches.Circle((0, 0), cyl.rad, fill=False, edgecolor="black",
                                      linestyle="--", linewidth=1, label="outer boundary"))
        ax.add_patch(mpatches.Ellipse((cyl.z0, 0), width=2 * cyl.az, height=2 * cyl.ar,
                                       fill=False, edgecolor="black", linestyle=":", linewidth=1,
                                       label="doping ellipsoid"))
        ax.set_xlabel("z")
        ax.set_ylabel("r")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")

    fig.suptitle(rf"CylinderCharges reconstruction ($\gamma$ = {cyl.gamma})")
    return fig

def plot_cylindercharges_reconstruction_xy_slice(model, cyl, z_slice=None, n_grid=200):
    """Cross-section PERPENDICULAR to the cylinder axis (an x-y disk at fixed
    z), complementing plot_cylindercharges_reconstruction's y=0 longitudinal
    slice. The true solution is axisymmetric (depends on r,z only, never on
    phi), but the model has no such constraint built in - it was only ever
    evaluated on the single y=0 plane by the longitudinal plot, so a phi-
    dependent error there would never show up. This checks whether the
    reconstruction stays circularly uniform at a given height instead.
    Defaults to slicing through the doping ellipsoid's center (cyl.z0), where
    the doping/potential signal is strongest. Call model.eval() before this."""
    if z_slice is None:
        z_slice = cyl.z0

    r_slice = np.sqrt(max(cyl.rad**2 - z_slice**2, 0.0))
    u = np.linspace(-r_slice, r_slice, n_grid)
    v = np.linspace(-r_slice, r_slice, n_grid)
    U, V = np.meshgrid(u, v)
    mask = U**2 + V**2 <= r_slice**2

    xg, yg = U.ravel(), V.ravel()
    zg = np.full_like(xg, z_slice)

    phi_true = cyl.potential_physical(xg, yg, zg).reshape(U.shape)

    X = torch.tensor(np.stack([xg, yg, zg], axis=1), dtype=torch.float32, device=config.device) / cyl.rad
    with torch.inference_mode():
        phi_model = (model(X).squeeze(-1).cpu().numpy() * cyl.V_T).reshape(U.shape)

    phi_true = np.where(mask, phi_true, np.nan)
    phi_model = np.where(mask, phi_model, np.nan)
    pct_diff = np.where(mask, (phi_true - phi_model) / np.nanmax(np.abs(phi_true)) * 100, np.nan)

    extent = [-r_slice, r_slice, -r_slice, r_slice]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    shared_norm = _diverging_norm(np.concatenate([phi_true[mask], phi_model[mask]]))
    im0 = axes[0].imshow(phi_true, extent=extent, origin="lower", cmap=DIVERGING_CMAP, norm=shared_norm)
    fig.colorbar(im0, ax=axes[0], label=r"ground truth potential $\phi$ (V)")
    axes[0].set_title("ground truth")

    im1 = axes[1].imshow(phi_model, extent=extent, origin="lower", cmap=DIVERGING_CMAP, norm=shared_norm)
    fig.colorbar(im1, ax=axes[1], label=r"reconstructed potential $\phi$ (V)")
    axes[1].set_title("reconstructed")

    pct_norm = _diverging_norm(pct_diff[mask])
    im2 = axes[2].imshow(pct_diff, extent=extent, origin="lower", cmap=DIVERGING_CMAP, norm=pct_norm)
    fig.colorbar(im2, ax=axes[2], label=r"% difference (of peak $\phi$)")
    axes[2].set_title("difference")

    # doping ellipsoid's own circular cross-section at this height, if the
    # slice actually cuts through it: (r/ar)^2 + ((z_slice-z0)/az)^2 = 1
    ellipse_term = 1 - ((z_slice - cyl.z0) / cyl.az) ** 2
    ellipse_r = cyl.ar * np.sqrt(ellipse_term) if ellipse_term > 0 else 0.0

    for ax in axes:
        ax.add_patch(mpatches.Circle((0, 0), r_slice, fill=False, edgecolor="black",
                                      linestyle="--", linewidth=1, label="outer boundary"))
        if ellipse_r > 0:
            ax.add_patch(mpatches.Circle((0, 0), ellipse_r, fill=False, edgecolor="black",
                                          linestyle=":", linewidth=1, label="doping boundary"))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")

    fig.suptitle(rf"CylinderCharges cross-section at z = {z_slice:.2e} cm ($\gamma$ = {cyl.gamma})")
    return fig

def _sphere_wireframe_coords(rad, n_u=30, n_v=20):
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    xs = rad * np.outer(np.cos(u), np.sin(v))
    ys = rad * np.outer(np.sin(u), np.sin(v))
    zs = rad * np.outer(np.ones_like(u), np.cos(v))
    return xs, ys, zs

def _ellipsoid_wireframe_coords(ar, az, z0, n_u=30, n_v=20):
    """ar is the radial (x,y) semi-axis, az the axial (z) semi-axis - same
    convention as the mpatches.Ellipse overlay in the 2D cross-section plots
    (width=2*az, height=2*ar), just in 3D as a solid of revolution about z."""
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    xs = ar * np.outer(np.cos(u), np.sin(v))
    ys = ar * np.outer(np.sin(u), np.sin(v))
    zs = z0 + az * np.outer(np.ones_like(u), np.cos(v))
    return xs, ys, zs

def plot_cylindercharges_reconstruction_3d(model, cyl, n_grid=120):
    """Two 3D panels (ground truth, reconstructed) for animate_rotating_view:
    the outer sphere and doping ellipsoid are drawn as translucent reference
    wireframes (rotationally invariant shapes, so their own rotation carries
    no information - the point of orbiting the camera around this static
    scene is that the sphere/ellipsoid/disk all appear to spin together,
    which is the honest way to depict an axisymmetric solution). The same
    (r,z) cross-section data as plot_cylindercharges_reconstruction is
    embedded as a flat colored disk at phi=0/pi. Call model.eval() first,
    same convention as the 2D version."""
    r_half = np.linspace(0, 1, n_grid)
    z_half = np.linspace(-1, 1, n_grid)
    rr, zz = np.meshgrid(r_half, z_half, indexing="ij")

    xg = (rr * cyl.rad).ravel()
    yg = np.zeros_like(xg)
    zg = (zz * cyl.rad).ravel()

    psi_true = cyl.potential(xg, yg, zg).reshape(rr.shape)

    X = torch.tensor(np.stack([xg, yg, zg], axis=1), dtype=torch.float32, device = config.device) / cyl.rad
    with torch.inference_mode():
        psi_model = model(X).squeeze(-1).cpu().numpy().reshape(rr.shape)

    r_full = np.concatenate([-r_half[:0:-1], r_half]) * cyl.rad
    z_full = z_half * cyl.rad
    psi_true_full = np.concatenate([psi_true[:0:-1], psi_true], axis=0)
    psi_model_full = np.concatenate([psi_model[:0:-1], psi_model], axis=0)

    R_disk, Z_disk = np.meshgrid(r_full, z_full, indexing="ij")
    X_disk = R_disk           # r_full already carries the sign (negative r = phi=pi half)
    Y_disk = np.zeros_like(R_disk)

    shared_norm = _diverging_norm(np.concatenate([psi_true_full.ravel(), psi_model_full.ravel()]))
    sphere_xyz = _sphere_wireframe_coords(cyl.rad)
    ellipsoid_xyz = _ellipsoid_wireframe_coords(cyl.ar, cyl.az, cyl.z0)

    fig = plt.figure(figsize=(14, 7))
    for i, (title, field) in enumerate([("ground truth", psi_true_full), ("reconstructed", psi_model_full)]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")

        facecolors = DIVERGING_CMAP(shared_norm(field))
        ax.plot_surface(X_disk, Y_disk, Z_disk, facecolors=facecolors,
                         rstride=1, cstride=1, shade=False, antialiased=False)
        ax.plot_wireframe(*sphere_xyz, color="black", alpha=0.12, linewidth=0.5)
        ax.plot_wireframe(*ellipsoid_xyz, color="black", alpha=0.35, linewidth=0.7)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_box_aspect([1, 1, 1])
        ax.set_title(rf"{title} $\psi$")

    sm = plt.cm.ScalarMappable(cmap=DIVERGING_CMAP, norm=shared_norm)
    sm.set_array([])
    fig.colorbar(sm, ax=fig.axes, shrink=0.6, label=r"potential $\psi$")
    fig.suptitle(rf"CylinderCharges reconstruction ($\gamma$ = {cyl.gamma})")
    return fig


# sequential blue ramp, light->dark (dataviz skill reference palette,
# references/palette.md "Sequential hue") - single hue for magnitude,
# deliberately not a rainbow colormap.
_SEQ_BLUE_HEX = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
_SEQ_BLUE_CMAP = mcolors.LinearSegmentedColormap.from_list("seq_blue", _SEQ_BLUE_HEX)


def plot_generalization_map(summary, metric="rms", train_summary=None):
    """Two-panel 2D map of held-out generalization error across the
    (L, nD_scale) sweep plane: absolute error (mV) on the left, error as a
    percentage of that anchor's own peak potential swing on the right. The
    absolute panel alone can't say whether a given mV error is negligible or
    huge for that anchor, since psi_peak varies substantially across the
    doping sweep.

    summary: list of (tag, L, nD_scale, max_err_mV, rms_err_mV, psi_peak_mV)
    tuples - extend evaluate_multi_anchor.py's summary.append(...) with
    `source.psi_peak * V_T * 1e3` as the sixth element (V_T is already
    imported there) to use this. metric: "rms" or "max" selects which error
    column drives both panels' color.

    Colored scatter, not an interpolated heatmap/contour - with only ~15
    LHS-sampled anchors, smoothing between them would imply resolution the
    sweep doesn't have. train_summary (optional, same tuple shape): plotted
    as small uncolored markers for sweep-coverage context only - train
    anchors have no held-out error to show.
    """
    col = 4 if metric == "rms" else 3
    metric_name = "RMS error" if metric == "rms" else "Max |error|"

    L        = np.array([float(row[1]) for row in summary])
    nD_scale = np.array([float(row[2]) for row in summary])
    err_mV   = np.array([float(row[col]) for row in summary])
    peak_mV  = np.array([float(row[5]) for row in summary])
    err_pct  = err_mV / peak_mV * 100

    logL, logN = np.log10(L), np.log10(nD_scale)
    if train_summary is not None:
        L_tr = np.array([float(row[1]) for row in train_summary])
        nD_tr = np.array([float(row[2]) for row in train_summary])
        logL_tr, logN_tr = np.log10(L_tr), np.log10(nD_tr)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    panels = [(axes[0], err_mV, f"{metric_name} (mV)"),
              (axes[1], err_pct, f"{metric_name} (% of peak potential)")]

    for ax, values, cbar_label in panels:
        if train_summary is not None:
            ax.scatter(logL_tr, logN_tr, s=40, marker="x", color="#c3c2b7",
                       label="train anchors", zorder=1)
        sc = ax.scatter(logL, logN, c=values, cmap=_SEQ_BLUE_CMAP,
                        s=180, edgecolor="#0b0b0b", linewidth=0.6, zorder=2)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(cbar_label)
        ax.set_xlabel("log10(L [cm])")
        ax.set_ylabel("log10(nD_scale)")
        if train_summary is not None:
            ax.legend()

    fig.suptitle("Generalization error across the (L, nD0) sweep")
    return fig


def plot_cost_sensitivity(C_run_measured, C_train, C_infer, N_anchors,
                           C_run_range=None, regime_markers=None):
    """N_breakeven vs. C_run, per problem_statement.md's cost model:

        N_breakeven = (N_anchors * C_run + C_train) / (C_run - C_infer)

    C_run is deliberately the swept x-axis, not a single asserted number -
    only C_run_measured is an actually-measured value (a real DEVSIM solve,
    timed directly); the rest of the curve is the honest sensitivity, and
    regime_markers are illustrative order-of-magnitude orientation labels,
    not claimed costs.

    regime_markers: optional list of (label, C_run_value) tuples for
    reference vlines (e.g. "detailed TCAD (~10 min)", 600.0). If None,
    only the measured point is annotated.
    """
    if C_run_range is None:
        lo = min(C_run_measured, C_infer * 20)
        C_run_range = np.logspace(np.log10(lo), 6, 400)  # up to ~11.6 days

    C_run_range = C_run_range[C_run_range > C_infer]
    N_breakeven = (N_anchors * C_run_range + C_train) / (C_run_range - C_infer)

    line_color = "#184f95"   # dark step of the house sequential-blue ramp
    ref_color  = "#8a8a82"   # recessive muted ink for reference lines/labels

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(C_run_range, N_breakeven, color=line_color, linewidth=2, zorder=3)

    N_at_measured = (N_anchors * C_run_measured + C_train) / (C_run_measured - C_infer)
    ax.scatter([C_run_measured], [N_at_measured], s=90, marker="o",
               facecolor="white", edgecolor=line_color, linewidth=2, zorder=4)
    ax.annotate(f"this 1D toy DEVSIM\nC_run={C_run_measured*1e3:.0f} ms\nN_breakeven≈{N_at_measured:,.0f}",
                xy=(C_run_measured, N_at_measured), xytext=(12, 14),
                textcoords="offset points", fontsize=9, color="#3a3a35")

    if regime_markers:
        blended = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        for label, c_run_val in regime_markers:
            if c_run_val <= C_infer:
                continue
            ax.axvline(c_run_val, color=ref_color, linestyle="--", linewidth=1, zorder=1)
            ax.text(c_run_val, 0.97, label, transform=blended,
                    rotation=90, va="top", ha="right", fontsize=8, color=ref_color)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("C_run: cost of one classical (DEVSIM) evaluation [s]")
    ax.set_ylabel("N_breakeven: queries until the surrogate is cheaper")
    ax.set_title("Where the surrogate pays off, as a function of how expensive\none classical run actually is")
    fig.tight_layout()
    return fig
