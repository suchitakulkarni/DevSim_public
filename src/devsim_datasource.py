import csv
import numpy as np
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader
import torch
import logging
from src.datasets import ObservedDataset, CollocDataset, BCDataset

from src.config import setup_logging
logger = logging.getLogger(__name__)

setup_logging(level=logging.INFO)

q         = 1.602e-19
k_B       = 1.381e-23
T         = 300.0
ni        = 1.5e10
epsilon_0 = 8.854e-14
epsilon_r = 11.7
V_T       = k_B * T / q
L_D       = np.sqrt(epsilon_r * epsilon_0 * k_B * T / (q**2 * ni))


class DevsimDataSource:
    """
    Wraps .npz output of run_devsim_diode.py.

    Design contract (matches CylinderCharges):
        - ALL public methods receive PHYSICAL coordinates (cm)
        - coords_normalised() is the single place that converts physical -> normalised
        - Interpolators are built on physical x internally
        - potential()          returns phi/V_T         (dimensionless, O(10))
        - potential_normalised() returns phi/(V_T*psi_peak) (O(1), for plotting)
        - density()            returns NetDoping/ni    (dimensionless, O(67000))
        - density_normalised() returns NetDoping/ni/nD_peak_normalised (O(1))
        - nD_peak_normalised   scalar to pass as nD_scale to poisson_loss_nonlinear
    """
    n_spatial_dims = 1
    def __init__(self, npz_path):
        data           = np.load(npz_path)
        x_raw          = data["x"]            # cm, physical
        potential_raw  = data["potential"]    # V,  physical
        net_doping_raw = data["net_doping"]   # cm^-3, physical
        bc_x_raw       = data["bc_x"]         # cm, physical
        bc_pot_raw     = data["bc_pot"]        # V,  physical    

        # geometry in physical units - callers use these to set up SlabGenerator
        self.x_min = float(x_raw.min())
        self.x_max = float(x_raw.max())
        self.L     = data["length"]

        # peak values for normalisation - exposed so training script can pass to loss
        self.psi_peak            = float(np.max(np.abs(potential_raw / V_T)))
        self.nD_peak_normalised  = float(np.max(np.abs(data["nD0"] / ni)))

        # boundary conditions in dimensionless units
        self.bc_x_physical = bc_x_raw                  # cm - pass to SlabGenerator
        self.bc_pot        = bc_pot_raw / V_T           # phi/V_T, for BCDataset

        # interpolators built on PHYSICAL x - no hidden normalisation
        self._pot_interp = interp1d(x_raw, potential_raw / V_T,
                                    kind="cubic",
                                    bounds_error=False, fill_value="extrapolate")
        self._nD_interp  = interp1d(x_raw, net_doping_raw / ni,
                                    kind="linear",
                                    bounds_error=False, fill_value=0.0)

        # Debye length and gamma
        self.gamma = self.L / L_D
        logger.info(f"Device length   L    = {self.L:.3e} cm")
        logger.info(f"Debye length    L_D  = {L_D:.3e} cm")
        logger.info(f"gamma                = {self.gamma:.3e}")
        logger.info(f"nD_peak_normalised   = {self.nD_peak_normalised:.3e}")
        logger.info(f"psi_peak             = {self.psi_peak:.3f}")

    def coords_normalised(self, x, y, z):
        """Physical cm -> normalised [0,1]. Single normalisation point."""
        #this needs extension once we go to 2D/3D, we will need to add (y - self.y_min) / self.W and (z - self.z_min) / self.h
        return ((x - self.x_min) / self.L, )

    def potential(self, x, y, z):
        """Physical x (cm) -> phi/V_T. O(10)."""
        return self._pot_interp(x)

    def potential_normalised(self, x, y, z):
        """Physical x (cm) -> phi/(V_T*psi_peak). O(1). For plotting only."""
        return self._pot_interp(x) / self.psi_peak

    def density(self, x, y, z):
        """Physical x (cm) -> NetDoping/ni. O(67000)."""
        return self._nD_interp(x)

    def density_normalised(self, x, y, z):
        """Physical x (cm) -> NetDoping/ni/nD_peak_normalised. O(1).
        Pass to CollocDataset. Pass nD_peak_normalised as nD_scale to loss."""
        return self._nD_interp(x) / self.nD_peak_normalised
    
def sample_junction(nD_scale, L, seed = None):
    W = np.sqrt( 8 * (1/nD_scale) * np.log(nD_scale))* L_D
    sigma = W/2
    sigma_clamped = np.minimum(sigma, L/2)
    return sigma_clamped

class SlabGenerator:
    """
    Samples points in physical coordinates (cm) matching DevsimDataSource.
    y and z are zero throughout - 1D device embedded in 3D PINN.
    """

    def __init__(self, x_min, x_max, n_points):
        self.x_min    = x_min
        self.x_max    = x_max
        self.n_points = n_points


    def sample_interior(self, seed=None):
        """Uniform sample in physical x (cm)."""
        rng = np.random.default_rng(seed)
        x   = rng.uniform(self.x_min, self.x_max, self.n_points)
        y   = np.zeros_like(x)
        z   = np.zeros_like(x)
        return x, y, z

    def sample_interior_biased(self, source, seed=None, n_oversample=5):
        """Gaussian sampler biased toward the junction (x = x_min + L/2), sigma
        from sample_junction(). source: DevsimDataSource. Physical coordinates."""

        sigma_clamped = sample_junction(source.nD_peak_normalised, source.L)
        mu = self.x_min + source.L / 2

        rng    = np.random.default_rng(seed)
        x_grid = rng.normal(mu, sigma_clamped, self.n_points * n_oversample)
        mask   = (x_grid >= self.x_min) & (x_grid <= self.x_max)
        x_acc  = x_grid[mask][:self.n_points]
        if len(x_acc) < self.n_points:
            x_fill = rng.uniform(self.x_min, self.x_max, self.n_points - len(x_acc))
            x_acc  = np.concatenate([x_acc, x_fill])
        return x_acc, np.zeros(self.n_points), np.zeros(self.n_points)

    def sample_boundary(self):
        """Two endpoints in physical coordinates."""
        x = np.array([self.x_min, self.x_max])
        y = np.zeros(2)
        z = np.zeros(2)
        return x, y, z


def build_loaders(source, n_obs, n_colloc, n_bc, obs_batch, device,
                  obs_seed = 42, val_seed = 99, colloc_seed = 86):
    """
    Builds DataLoaders and validation tensors for train_model().
    All sampling in physical coordinates; normalisation applied here
    before tensors are built so the network always sees normalised input.
    """
    gen = SlabGenerator(source.x_min, source.x_max, n_obs)

    # training observations
    x_obs, y_obs, z_obs = gen.sample_interior(seed = obs_seed)
    # This is going to unwrap the tuple and produce shape(n_points, n_spatial_dims)
    xn_obs = np.stack(source.coords_normalised(x_obs, y_obs, z_obs), axis = 1)
    pot_obs  = source.potential(x_obs, y_obs, z_obs)
    X_obs = torch.tensor(xn_obs, dtype=torch.float32).to(device)

    gamma_vals_obs = np.full(n_obs, source.gamma)
    nD_scale_vals_obs = np.full(n_obs, source.nD_peak_normalised)
    condition_params_obs =  np.stack([gamma_vals_obs, nD_scale_vals_obs], axis = 1) # this needs to be checked if it is nD_peak_normalised

    obs_ds   = ObservedDataset(X_obs, pot_obs, condition_params = condition_params_obs)
    obs_loader = DataLoader(obs_ds, batch_size = obs_batch, shuffle = True)

    # validation observations - different seed, never seen in training
    x_val, y_val, z_val = gen.sample_interior(seed = val_seed)
    xn_val = np.stack(source.coords_normalised(x_val, y_val, z_val), axis = 1)
    pot_val  = source.potential(x_val, y_val, z_val)
    X_val = torch.tensor(xn_val, dtype=torch.float32).to(device)

    gamma_vals_val = np.full(n_obs, source.gamma) # the validation and observation have same number of points
    nD_scale_vals_val = np.full(n_obs, source.nD_peak_normalised)
    condition_params_val =  np.stack([gamma_vals_val, nD_scale_vals_val], axis = 1) # this needs to be checked if it is nD_peak_normalised

    val_ds   = ObservedDataset(X_val, pot_val, condition_params = condition_params_val)

    # collocation points
    gen_c = SlabGenerator(source.x_min, source.x_max, n_colloc)
    x_c, y_c, z_c = gen_c.sample_interior_biased(source, seed = colloc_seed)
    xn_c = np.stack(source.coords_normalised(x_c, y_c, z_c), axis = 1)
    X_colloc  = torch.tensor(xn_c, dtype=torch.float32).to(device)
    nD_colloc = source.density_normalised(x_c, y_c, z_c)

    gamma_vals_colloc = np.full(n_colloc, source.gamma)
    nD_scale_vals_colloc = np.full(n_colloc, source.nD_peak_normalised)
    condition_params_colloc =  np.stack([gamma_vals_colloc, nD_scale_vals_colloc], axis = 1) 

    colloc_ds = CollocDataset(X_colloc = X_colloc, rho_colloc = nD_colloc, condition_params = condition_params_colloc)
    colloc_batch = max(1, obs_batch * n_colloc // n_obs)
    colloc_loader = DataLoader(colloc_ds, batch_size=colloc_batch, shuffle=True)

    # boundary conditions
    x_b, y_b, z_b = SlabGenerator(source.x_min, source.x_max, 2).sample_boundary()
    xn_b = np.stack(source.coords_normalised(x_b, y_b, z_b), axis = 1)
    X_bc   = torch.tensor(xn_b, dtype=torch.float32).to(device)

    gamma_vals_bc = np.full(n_bc, source.gamma)
    nD_scale_vals_bc = np.full(n_bc, source.nD_peak_normalised)
    condition_params_bc =  np.stack([gamma_vals_bc, nD_scale_vals_bc], axis = 1) 
    bc_ds  = BCDataset(X_bc = X_bc, pot_bc = source.bc_pot, condition_params = condition_params_bc)
    bc_loader = DataLoader(bc_ds, batch_size=2, shuffle=False)

    return obs_loader, colloc_loader, bc_loader, val_ds.X, val_ds.y, val_ds.cond_params


def read_manifest(manifest_path="data/records.csv", split=None):
    """Reads the sweep manifest written by run_devsim_diode.py's __main__.
    split=None returns every row; split='train'/'test' filters to that column."""
    rows = []
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if split is None or row["split"] == split:
                rows.append(row["filename"])
    return rows


def build_multi_anchor_loaders(anchor_files, n_obs_per_anchor, n_colloc_per_anchor, obs_batch, device,
                                obs_seed=42, val_seed=99, colloc_seed=86):
    """
    Multi-anchor version of build_loaders(): loops over several anchor .npz
    files, samples each anchor's own observed/collocation/BC points using its
    own DevsimDataSource (own x_min/x_max/gamma/nD_peak_normalised), then
    concatenates across anchors into ONE combined set of loaders.

    train_model / poisson_loss_nonlinear need zero changes to consume this -
    they already treat condition_params as per-point data, not per-run, so a
    batch mixing points from many different devices is exactly what they
    already expect. Only the data-loading step changes.

    Returns the same shape as build_loaders: obs_loader, colloc_loader,
    bc_loader, val_X, val_y, val_cond.
    """
    X_obs_list, y_obs_list, cond_obs_list = [], [], []
    X_val_list, y_val_list, cond_val_list = [], [], []
    X_colloc_list, nD_colloc_list, cond_colloc_list = [], [], []
    X_bc_list, pot_bc_list, cond_bc_list = [], [], []

    for anchor_file in anchor_files:
        source = DevsimDataSource(anchor_file)
        gen = SlabGenerator(source.x_min, source.x_max, n_obs_per_anchor)

        # observed
        x_obs, y_obs, z_obs = gen.sample_interior(seed=obs_seed)
        xn_obs = np.stack(source.coords_normalised(x_obs, y_obs, z_obs), axis = 1)
        pot_obs = source.potential(x_obs, y_obs, z_obs)
        gamma_vals = np.full(n_obs_per_anchor, source.gamma)
        nD_scale_vals = np.full(n_obs_per_anchor, source.nD_peak_normalised)
        X_obs_list.append(xn_obs)
        y_obs_list.append(pot_obs)
        cond_obs_list.append(np.stack([gamma_vals, nD_scale_vals], axis=1))

        # validation - different seed, never trained on, same anchor
        x_val, y_val, z_val = gen.sample_interior(seed=val_seed)
        xn_val = np.stack(source.coords_normalised(x_val, y_val, z_val), axis = 1)
        pot_val = source.potential(x_val, y_val, z_val)
        X_val_list.append(xn_val)
        y_val_list.append(pot_val)
        cond_val_list.append(np.stack([gamma_vals, nD_scale_vals], axis=1))

        # collocation
        gen_c = SlabGenerator(source.x_min, source.x_max, n_colloc_per_anchor)
        x_c, y_c, z_c = gen_c.sample_interior_biased(source, seed=colloc_seed)
        xn_c = np.stack(source.coords_normalised(x_c, y_c, z_c), axis = 1)
        nD_c = source.density_normalised(x_c, y_c, z_c)
        gamma_vals_c = np.full(n_colloc_per_anchor, source.gamma)
        nD_scale_vals_c = np.full(n_colloc_per_anchor, source.nD_peak_normalised)
        X_colloc_list.append(xn_c)
        nD_colloc_list.append(nD_c)
        cond_colloc_list.append(np.stack([gamma_vals_c, nD_scale_vals_c], axis=1))

        # boundary - always exactly 2 points per anchor
        x_b, y_b, z_b = SlabGenerator(source.x_min, source.x_max, 2).sample_boundary()
        xn_b = np.stack(source.coords_normalised(x_b, y_b, z_b), axis = 1)
        gamma_vals_b = np.full(2, source.gamma)
        nD_scale_vals_b = np.full(2, source.nD_peak_normalised)
        X_bc_list.append(xn_b)
        pot_bc_list.append(source.bc_pot)
        cond_bc_list.append(np.stack([gamma_vals_b, nD_scale_vals_b], axis=1))
        

    X_obs, y_obs, cond_obs = (np.concatenate(v, axis=0) for v in (X_obs_list, y_obs_list, cond_obs_list))
    X_val, y_val, cond_val = (np.concatenate(v, axis=0) for v in (X_val_list, y_val_list, cond_val_list))
    X_colloc, nD_colloc, cond_colloc = (np.concatenate(v, axis=0) for v in (X_colloc_list, nD_colloc_list, cond_colloc_list))
    X_bc, pot_bc, cond_bc = (np.concatenate(v, axis=0) for v in (X_bc_list, pot_bc_list, cond_bc_list))

    obs_ds = ObservedDataset(X_obs, y_obs, condition_params=cond_obs)
    obs_loader = DataLoader(obs_ds, batch_size=obs_batch, shuffle=True)

    val_ds = ObservedDataset(X_val, y_val, condition_params=cond_val)

    colloc_ds = CollocDataset(X_colloc=X_colloc, rho_colloc=nD_colloc, condition_params=cond_colloc)
    n_obs_total = n_obs_per_anchor * len(anchor_files)
    n_colloc_total = n_colloc_per_anchor * len(anchor_files)
    colloc_batch = max(1, obs_batch * n_colloc_total // n_obs_total)
    colloc_loader = DataLoader(colloc_ds, batch_size=colloc_batch, shuffle=True)

    bc_ds = BCDataset(X_bc=X_bc, pot_bc=pot_bc, condition_params=cond_bc)
    bc_loader = DataLoader(bc_ds, batch_size=len(anchor_files) * 2, shuffle=False)

    return obs_loader, colloc_loader, bc_loader, val_ds.X, val_ds.y, val_ds.cond_params
