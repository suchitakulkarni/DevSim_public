import numpy as np
import os, sys
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

# ------------------------- unit conversion helpers -------------------------
def normalise_coords(x, y, z, rad):
    """Shared across all three physics classes - coordinates always enter the
    model/loss as x/rad, y/rad, z/rad, regardless of which physics case they
    came from. One shared function instead of three copies of the same divide."""
    return x/rad, y/rad, z/rad

def encode_condition_params(condition_params):
    """log10-transform gamma/nD_scale before they reach the model as input -
    both span many decades (nD_scale ~1e5-1e9 across the sweep), which saturates
    a tanh MLP if fed raw (confirmed: pre-activations of ~1e8, Tanh output pinned
    to exactly +-1, zero gradient). Only for the model's own input encoding -
    the physics loss must keep using the untransformed condition_params, since
    the governing equation needs the real physical gamma/nD_scale, not their logs.
    TODO (tomorrow): centre/rescale properly per-column instead of bare log10;
    this is the fast fix to unblock training tonight, not the tuned version."""
    return torch.log10(condition_params)

# ------------------------- Datasets -------------------------
class ObservedDataset(Dataset):
    def __init__(self, X, y, condition_params = None):
        if condition_params is not None: 
            self.cond_params = torch.tensor(condition_params, dtype = torch.float32, requires_grad = False).to(config.device)
        else: 
            condition_params = np.zeros((len(X), 0))
            self.cond_params = torch.tensor(condition_params, dtype = torch.float32, requires_grad = False).to(config.device)
        self.X = torch.tensor(X, dtype = torch.float32, requires_grad = True).to(config.device)
        self.y = torch.tensor(y, dtype = torch.float32).unsqueeze(-1).to(config.device)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.cond_params[idx]
            
class CollocDataset(Dataset):
    def __init__(self, X_colloc, rho_colloc, condition_params = None):
        if condition_params is not None: 
            self.cond_params = torch.tensor(condition_params, dtype = torch.float32, requires_grad = False).to(config.device)
        else: 
            condition_params = np.zeros((len(X_colloc), 0))
            self.cond_params = torch.tensor(condition_params, dtype = torch.float32, requires_grad = False).to(config.device)
        self.X_colloc = torch.tensor(X_colloc, dtype = torch.float32, requires_grad = True).to(config.device)
        self.rho_colloc = torch.tensor(rho_colloc, dtype = torch.float32).unsqueeze(-1).to(config.device)
    def __len__(self):
        return len(self.rho_colloc)
    def __getitem__(self, idx):
        return self.X_colloc[idx], self.rho_colloc[idx], self.cond_params[idx]

class BCDataset(Dataset):
    def __init__(self, X_bc, pot_bc, condition_params = None):
        if condition_params is not None: 
            self.cond_params = torch.tensor(condition_params, dtype = torch.float32, requires_grad = False).to(config.device)
        else: 
            condition_params = np.zeros((len(X_bc), 0))
            self.cond_params = torch.tensor(condition_params, dtype = torch.float32, requires_grad = False).to(config.device)
        self.X_bc = torch.tensor(X_bc, dtype = torch.float32, requires_grad = True).to(config.device)
        self.pot_bc = torch.tensor(pot_bc, dtype = torch.float32).unsqueeze(-1).to(config.device)
    def __len__(self):
        return len(self.pot_bc)
    def __getitem__(self, idx):
        return self.X_bc[idx], self.pot_bc[idx], self.cond_params[idx]
 
class TrainingData:
    def __init__(self, dataloader, colloc_loader, bc_loader, val_X, val_y, val_cond):
        self.dataloader = dataloader
        self.colloc_loader = colloc_loader
        self.bc_loader = bc_loader
        self.val_X = val_X
        self.val_y = val_y
        self.val_cond = val_cond
        
# ------ dataset generators

class UniformDensity:
    ''' Takes as inputs the sampled x, y, z points, uniform charge density and returns potential'''
    def __init__(self, x = [], y = [], z = [], rho = 1, epsilon = 1, rad = None):
        self.x = np.array(x)
        self.y = np.array(y)
        self.z = np.array(z)
        self.r = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        self.R = max(self.r)
        self.rad = rad if rad is not None else self.R # rad can be passed explicitly; falls back to the sampled max radius otherwise
        self.rho = rho
        self.epsilon = epsilon
        self.epsilon_physical = epsilon
        self.epsilon_normalised = 1.0
        self.rho0 = self.rho # the density is spatially uniform, so it is already its own normalisation reference

    def PoissonUniformChargePotential(self, r):
        return self.rho/(6*self.epsilon)*(3*self.R**2 - r**2)

    def potential_physical(self, r):
        return self.PoissonUniformChargePotential(r)

    def potential_normalised(self, r):
        return self.potential_physical(r) * self.epsilon_physical / (self.rho0 * self.rad**2)

    def density_physical(self, x, y, z):
        return np.full_like(np.asarray(x, dtype=float), self.rho)

    def density_normalised(self, x, y, z):
        return self.density_physical(x, y, z) / self.rho0 # == 1 everywhere, by construction

    def coords_normalised(self, x, y, z):
        return normalise_coords(x, y, z, self.rad)

class CylinderCharges:
    ''' this is a cylindric charge distribution'''
    def __init__(self, rad, epsilon = 1, gamma = 1, z0 = 0.1, ar = 0.6, az= 0.8, nD0 = 1E16, ni = 1.5E10, w = 0.1,
                 q = 1.602e-19, k = 1.381e-23, T = 300): #z0, ar and az are relative to radius
        self.rad = rad
        self.epsilon = epsilon
        self.ar = ar*rad
        self.az = az*rad
        self.z0 = z0*rad
        self.nD0 = nD0
        self.w = w
        self.ni = ni
        self.nD_peak_normalised = self.nD0 / self.ni  # density() saturates to nD0 at the doping core; this is its peak in ni-normalised units, used to bring the PB source term to O(1) in the physics loss
        self.gamma = gamma
        self.q = q
        self.k = k
        self.T = T
        self.V_T = k*T/q # thermal voltage - converts normalised psi to physical volts
        self.rtilde = np.linspace(0, 1, 100) # these are normalised r, it can not be larger than radius
        self.ztilde = np.linspace(-1, 1, 100) # this is normalized z, it can not be larger than radius either
        self.rvals, self.zvals = np.meshgrid(self.rtilde, self.ztilde, indexing = 'ij') # this defines a box grid
        self.exterior = np.sqrt(self.rvals**2 + self.zvals**2) >= 1 # this defines what's in the box but outside the sphere
        self.nD_grid = self.density(self.rvals*self.rad, np.zeros_like(self.rvals), self.zvals*self.rad) # the density is computed over entire box not just inside the sphere, we apply mask later to remove unwanted points, leads to wasted compute
        self._solve()

    def _solve(self):
        def psi_residual(psi):
            dr = self.rtilde[1] - self.rtilde[0]
            dz = self.ztilde[1] - self.ztilde[0]
            d2psi_dr2 = (psi[2:, 1:-1] - 2*psi[1:-1, 1:-1] + psi[:-2, 1:-1])/dr**2
            dpsi_dr = (psi[2:, 1:-1] -psi[:-2, 1:-1])/(2*dr)
            d2psi_dz2 = (psi[1:-1, 2:] - 2*psi[1:-1, 1:-1] + psi[1:-1, :-2])/dz**2
            d2psi_drz2 = d2psi_dr2 + d2psi_dz2 + dpsi_dr/self.rvals[1:-1, 1:-1]
            R = np.zeros_like(psi)
            R[1:-1, 1:-1] = d2psi_drz2 + gamma**2*(self.nD_grid[1:-1, 1:-1] -2*np.sinh(psi[1:-1, 1:-1]))
            R[self.exterior] = psi[self.exterior] # this sets the field outside the sphere but inside the box to zero
            axis_r_term = 4*(psi[1, 1:-1] - psi[0, 1:-1])/dr**2
            axis_z_term = (psi[0, 2:] - 2*psi[0, 1:-1] + psi[0, :-2])/dz**2
            axis_laplacian = axis_r_term + axis_z_term
            R[0, 1:-1] = axis_laplacian + gamma**2*(self.nD_grid[0, 1:-1] -2*np.sinh(psi[0, 1:-1]))
            return R

        gamma_ramp = np.geomspace(1e-3, self.gamma, 80)
        current_guess = np.zeros_like(self.rvals)
        for gamma in gamma_ramp:
            try:
                current_guess = newton_krylov(psi_residual, current_guess, maxiter=400, f_tol=1e-5)
            except NoConvergence as e:
                current_guess = e.args[0]

        psi_solution = current_guess
        self._psi_interp = RegularGridInterpolator((self.rtilde, self.ztilde), psi_solution,
                                                    bounds_error=False, fill_value=0.0)

    def density(self, x, y, z):
        r = np.sqrt(x**2 + y**2)
        cylinder_geometry_sq = (r/self.ar)**2 + ((z - self.z0)/self.az)**2 # this is the squared normalised geometric coordinate
        nD = 0.5*self.nD0*(1 - np.tanh((cylinder_geometry_sq - 1)/self.w)) # this is where the density is modelled to 'fall off' smoothly utside the cylinder
        return nD/self.ni

    def potential(self, x, y, z):
        r_query = np.sqrt(x**2 + y**2) / self.rad
        z_query = z / self.rad
        return self._psi_interp(np.stack([r_query, z_query], axis=-1))

    def density_normalised(self, x, y, z):
        return self.density(x, y, z) / self.nD_peak_normalised

    def boundary_indicator(self, x, y, z, width=None):
        """Bump function peaked exactly on the doping ellipsoid's surface
        (cylinder_geometry_sq == 1, the tanh transition in density()), for veto-sampling
        collocation points concentrated on that thin shell - unlike density_normalised,
        which is high throughout the interior and doesn't specifically enrich the
        transition band itself. width defaults to the same self.w used by the density
        tanh, so the shell matches the actual physical transition thickness."""
        if width is None:
            width = self.w
        r = np.sqrt(x**2 + y**2)
        cylinder_geometry_sq = (r/self.ar)**2 + ((z - self.z0)/self.az)**2
        return np.exp(-((cylinder_geometry_sq - 1) / width)**2)

    def density_physical(self, x, y, z):
        return self.density(x, y, z) * self.ni

    def potential_normalised(self, x, y, z):
        return self.potential(x, y, z)

    def potential_physical(self, x, y, z):
        return self.potential(x, y, z) * self.V_T

    def coords_normalised(self, x, y, z):
        return normalise_coords(x, y, z, self.rad)

    def electric_field(self, x, y, z):
        """E = -grad(phi), evaluated in closed form via Gauss's law: each smeared
        charge is spherically symmetric about its center, so the field it produces
        at distance r is just its enclosed-charge fraction (the same erf term as
        in potential(), minus the Gaussian derivative piece) over 4*pi*epsilon0*r^2,
        directed radially away from that charge's center."""
        #xidist = x[:, None] - self.xi[None, :]
        #yidist = y[:, None] - self.yi[None, :]
        #zidist = z[:, None] - self.zi[None, :]
        #dist = np.maximum(self._dist(x, y, z), 1E-10)

        #enclosed_frac = special.erf(dist / (np.sqrt(2) * self.sigma)) - \
        #    np.sqrt(2 / np.pi) * (dist / self.sigma) * np.exp(-dist**2 / (2 * self.sigma**2))
        #E_over_r = self.qi / (4 * np.pi * self.epsilon0 * dist**3) * enclosed_frac

        #Ex = np.sum(E_over_r * xidist, axis=1)
        #Ey = np.sum(E_over_r * yidist, axis=1)
        #Ez = np.sum(E_over_r * zidist, axis=1)
        #return Ex, Ey, Ez
        pass

class PointCharges:
    ''' A fixed configuration of n_charges Gaussian-smeared point charges placed
    inside a sphere of radius rad. Sampled once at construction; density/potential
    can then be evaluated at as many different point clouds as needed, all against
    this same charge configuration. '''
    def __init__(self, rad, n_charges, epsilon = 1, seed=None):
        self.rad = rad
        self.epsilon = epsilon
        self.n_charges = n_charges

        gen = SphereGenerator(radius=0.8*rad, n_points=n_charges)
        _, _, _, self.xi, self.yi, self.zi = gen.sample_volume_uniform(seed=seed)

        rng = np.random.default_rng(seed)
        self.sigma = rng.uniform(0.05*rad, 0.2*rad, size=n_charges)
        self.qi = rng.choice([+1, -1], size=n_charges)*1.602e-19 # charges in coloumb

        self.epsilon_physical = epsilon
        self.epsilon_normalised = 1.0 # follows from choosing phi0 = rho0*rad**2/epsilon_physical below - not an arbitrary placeholder

        probe_gen = SphereGenerator(radius=rad, n_points=20000)
        _, _, _, xp, yp, zp = probe_gen.sample_volume_uniform(seed=999)
        self.rho0 = np.abs(self.density(xp, yp, zp)).max() # owned here instead of being recomputed by every pipeline call site

    def _dist(self, x, y, z):
        xidist = x[:, None] - self.xi[None, :]  # (n_points, n_charges)
        yidist = y[:, None] - self.yi[None, :]
        zidist = z[:, None] - self.zi[None, :]
        return np.sqrt(xidist**2 + yidist**2 + zidist**2)

    def density(self, x, y, z):
        dist = self._dist(x, y, z)
        norm = self.qi / (self.sigma * np.sqrt(2 * np.pi))**3
        rhoi = norm * np.exp(-dist**2 / (2 * self.sigma**2))
        return np.sum(rhoi, axis=1)

    def potential(self, x, y, z):
        dist = np.maximum(self._dist(x, y, z), 1E-10) # This will need careful checking
        phi_i = self.qi / (4 * np.pi * self.epsilon * dist) * special.erf(dist / (np.sqrt(2) * self.sigma))
        return np.sum(phi_i, axis=1)

    def density_physical(self, x, y, z):
        return self.density(x, y, z)

    def density_normalised(self, x, y, z):
        return self.density(x, y, z) / self.rho0

    def potential_physical(self, x, y, z):
        return self.potential(x, y, z)

    def potential_normalised(self, x, y, z):
        return self.potential(x, y, z) * self.epsilon_physical / (self.rho0 * self.rad**2)

    def coords_normalised(self, x, y, z):
        return normalise_coords(x, y, z, self.rad)

    def electric_field(self, x, y, z):
        """E = -grad(phi), evaluated in closed form via Gauss's law: each smeared
        charge is spherically symmetric about its center, so the field it produces
        at distance r is just its enclosed-charge fraction (the same erf term as
        in potential(), minus the Gaussian derivative piece) over 4*pi*epsilon*r^2,
        directed radially away from that charge's center."""
        xidist = x[:, None] - self.xi[None, :]
        yidist = y[:, None] - self.yi[None, :]
        zidist = z[:, None] - self.zi[None, :]
        dist = np.maximum(self._dist(x, y, z), 1E-10)

        enclosed_frac = special.erf(dist / (np.sqrt(2) * self.sigma)) - \
            np.sqrt(2 / np.pi) * (dist / self.sigma) * np.exp(-dist**2 / (2 * self.sigma**2))
        E_over_r = self.qi / (4 * np.pi * self.epsilon * dist**3) * enclosed_frac

        Ex = np.sum(E_over_r * xidist, axis=1)
        Ey = np.sum(E_over_r * yidist, axis=1)
        Ez = np.sum(E_over_r * zidist, axis=1)
        return Ex, Ey, Ez


class SphereGenerator:
    """Generate points on or inside a sphere and plot them in Cartesian coordinates."""
    def __init__(self, radius, n_points, results_dir="results"):
        self.radius = radius
        self.n_points = n_points
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        self.r = None
        self.theta = None
        self.phi = None
        self.x = None
        self.y = None
        self.z = None

    def sample_surface_uniform(self, seed=None):
        """Sample points uniformly on the sphere surface (fixed r = radius).
        Uses u = cos(theta) ~ Uniform(-1, 1) to correct for the sin(theta)
        Jacobian in the spherical volume element, rather than sampling
        theta ~ Uniform(0, pi) directly, which would cluster points at the poles.
        """
        rng = np.random.default_rng(seed)

        u = rng.uniform(-1.0, 1.0, self.n_points)
        self.theta = np.arccos(u)
        self.phi = rng.uniform(0.0, 2.0 * np.pi, self.n_points)
        self.r = np.full(self.n_points, self.radius)

        self._to_cartesian()
        return  self.r, self.theta, self.phi, self.x, self.y, self.z

    def sample_volume_uniform(self, seed=None):
        """Sample points uniformly on the sphere volume  (r <= radius).

        Uses u = cos(theta) ~ Uniform(-1, 1) to correct for the sin(theta)
        Jacobian in the spherical volume element, rather than sampling
        theta ~ Uniform(0, pi) directly, which would cluster points at the poles.
        """
        rng = np.random.default_rng(seed)

        u = rng.uniform(-1.0, 1.0, self.n_points)
        self.theta = np.arccos(u)
        self.phi = rng.uniform(0.0, 2.0 * np.pi, self.n_points)
        self.r = self.radius * rng.uniform(0.0, 1.0, self.n_points) ** (1.0 / 3.0)

        self._to_cartesian()
        return self.r, self.theta, self.phi, self.x, self.y, self.z

    def sample_volume_radial_geometry(self, seed=None):
        """
        Here we sample from non-trivial radial geometry. To achieve that we take the inverse CDF of the normalised target density. 
        """
        def gaussian(r_grid):
            # This is the physical density profile of choice
            sigma = 5.0
            return np.exp(-r_grid**2/sigma**2)
            
        rng = np.random.default_rng(seed)

        u = rng.uniform(-1.0, 1.0, self.n_points)
        self.theta = np.arccos(u)
        self.phi = rng.uniform(0.0, 2.0 * np.pi, self.n_points)

        r_grid = np.linspace(0, self.radius, 2000)
        integrand = r_grid**2 * gaussian(r_grid)
        
        F_grid = scipy.integrate.cumulative_trapezoid(integrand, r_grid, initial=0.0)
        # THIS IS IMPORTANT DO NOT FORGET TO NORMALISE
        F_grid = F_grid / F_grid[-1]
        
        inverse_cdf = scipy.interpolate.interp1d(F_grid, r_grid)
        
        v = rng.uniform(0.0, 1.0, size = self.n_points)
        self.r = inverse_cdf(v)

        self._to_cartesian()
        return self.r, self.theta, self.phi, self.x, self.y, self.z

    def sample_volume_via_veto(self, target_density, seed=None):
        """
        Here is a random target function or density kernel which can not easily be separated, nor inverted. 
        We therefore use the veto method. The idea is 
        1. sample from a large enough grid of r, theta, phi values. (Note, the theta can be flat distribution, veto takes care of the rest)
        2. Estimate the maximum value of target function for that large enough grid. The assumption is that the max lies in the sample, 
           this is of course hard if the function has a narrow spike
        3. Now you generate points, for every point, sample a random between 0 and max signifying acceptance threshold. 
           Draw a random r, theta, phi point and check if the target function for that r, theta, phi is smaller or larger than acceptance threshold. 
           Accept or reject. 
        """
        rng = np.random.default_rng(seed)
        _, _, _, grid_x, grid_y, grid_z = self.sample_volume_uniform()
        M = np.max(target_density(grid_x, grid_y, grid_z))
        accepted = 0
        x_accepted = []
        y_accepted = []
        z_accepted = []
        while accepted < self.n_points:
            rng_for_draw = np.random.default_rng()
            seed = rints = rng_for_draw.integers(low = 0, high = 1000, size = 1)
            _, _, _, xtest, ytest, ztest = self.sample_volume_uniform(seed)
            yproposed = rng.uniform(0, M, size = len(xtest))
            target_drawn = target_density(xtest, ytest, ztest)
            accept = yproposed < target_drawn
            x_accepted.append(xtest[accept]) #this appends xtest[accept] array to x_accepted array, not elementwise
            y_accepted.append(ytest[accept])
            z_accepted.append(ztest[accept])
            accepted += accept.sum()
        self.x = np.concatenate(x_accepted) # because x_accepted etc have now become array of arrays
        self.y = np.concatenate(y_accepted)
        self.z = np.concatenate(z_accepted)
        self.r = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        self.phi = np.arctan2(self.y, self.x)
        self.theta = np.arccos(self.z/self.r)
        # we will need to convert to r, theta, phi for keeping the signature same
        return self.r, self.theta, self.phi, self.x, self.y, self.z

    def _to_cartesian(self):
        """Convert stored spherical coordinates (r, theta, phi) to Cartesian (x, y, z)."""
        self.x = self.r * np.sin(self.theta) * np.cos(self.phi)
        self.y = self.r * np.sin(self.theta) * np.sin(self.phi)
        self.z = self.r * np.cos(self.theta)

    def plot(self, filename="sphere.png", point_size=4, elev=20, azim=35):
        """Plot the current point cloud in 3D and save to the results directory."""
        if self.x is None:
            raise RuntimeError("No points generated yet. Call sample_surface_uniform first.")

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(self.x, self.y, self.z, s=point_size, c=self.z, cmap="viridis")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)

        out_path = os.path.join(self.results_dir, filename)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        return fig
