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
from src.datasets import encode_condition_params

# ------------------------- Physics loss -------------------------
def poisson_loss_nonlinear(model, X, nD, condition_params, reduction = 'mean', spatial_dims=(0,1,2)):
    eps = 1e-12
    gamma = condition_params[:, 0:1]
    nD_scale = condition_params[:, 1:2]

    X_packed = torch.cat([X, encode_condition_params(condition_params)], dim=1)
    u = model(X_packed)
    grad_u = torch.autograd.grad(u, X, grad_outputs = torch.ones_like(u), create_graph = True)[0]

    d2u_dx2 = [torch.autograd.grad(grad_u[:, i].sum(), X, create_graph = True)[0][:, i] for i in spatial_dims]

    laplacian = sum(d2u_dx2).unsqueeze(-1) # shape (N, 1)
    sinh = torch.sinh(u)
    # nD is expected peak-normalised (O(1), e.g. CylinderCharges.density_normalised),
    # not the raw physical nD/ni which can be ~1e5-1e6 for realistic doping. The true
    # residual is laplacian + gamma**2*(nD_scale*nD - 2*sinh); dividing that whole
    # "=0" equation by the positive constant gamma**2*nD_scale leaves its zero (i.e.
    # the physics being enforced) unchanged, but keeps the MSE/gradients O(1) instead
    # of being dominated by the huge raw doping peak regardless of network capacity.
    #non_linear_eqn = laplacian/(gamma**2*nD_scale) + nD - 2*sinh/nD_scale
    raw = laplacian + gamma**2 * (nD_scale * nD - 2 * sinh)
    scale = (laplacian.abs() + gamma**2 * nD_scale * nD.abs() + 2 * gamma**2 * sinh.abs()).detach() + eps
    #eps = torch.finfo(scale.dtype).eps * gamma**2 * nD_scale # we have to think if this goes to  a UnitRemover class
    non_linear_eqn = raw / scale
    if reduction == 'mean':
        loss = torch.mean(non_linear_eqn**2)
    else: loss = non_linear_eqn # we choose to return the loss, to check if the prediction under/overshoots physics
    return loss

def poisson_loss_linear(model, X, rho, epsilon):
    X = X.requires_grad_(True).to(config.device)
    u = model(X)
    grad_u = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    
    d2u_dx2 = torch.autograd.grad(grad_u[:, 0].sum(), X, create_graph=True)[0][:, 0]
    d2u_dy2 = torch.autograd.grad(grad_u[:, 1].sum(), X, create_graph=True)[0][:, 1]
    d2u_dz2 = torch.autograd.grad(grad_u[:, 2].sum(), X, create_graph=True)[0][:, 2]
    
    laplacian = (d2u_dx2 + d2u_dy2 + d2u_dz2).unsqueeze(-1)
    rho_tensor = rho.to(X.device)
    loss = torch.mean((laplacian + rho_tensor/epsilon)**2)
    return loss

def poisson_bc(model, X, pot_bc):
    # either we take R as input or we find out the max from the sampling.
    X = X.to(config.device)
    pot_bc_tensor = pot_bc.to(config.device)
    BC_pred = model(X).to(config.device)
    loss = torch.mean((BC_pred - pot_bc_tensor)**2)
    return loss

# ------------------------- ML architectures -------------------------
class PoissonTransformer(nn.Module):
    def __init__(self, d_model=32, nhead=4, dim_feedforward=128, n_layers=2, dropout=0):
        super().__init__()
        self.input_embedding = nn.Linear(3, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_projection = nn.Linear(d_model, 1)

    def forward(self, X):
        x = self.input_embedding(X)    # (batch, d_model)
        x = x.unsqueeze(1)            # (batch, 1, d_model)
        x = self.encoder(x)           # (batch, 1, d_model)
        x = x.squeeze(1)              # (batch, d_model)
        return self.output_projection(x)  # (batch, 1)

class PoissonMLP(nn.Module):
    def __init__(self, indim = 3, hiddim = 32, outdim = 1, n_layers = 2, w = 0.0, num_frequencies = 5, dropout = 0.0):
        super().__init__()
        self.w = w
        self.num_frequencies = num_frequencies
        if self.w > 0:
            self.indim = 2 * num_frequencies * 2 # we get 2*2 here since we convert the inputs into r, z, so we loose one of three dimensions
            self.frequencies = torch.tensor(np.geomspace(1, 1/self.w, self.num_frequencies), dtype = torch.float32).to(config.device)
        else: self.indim = indim
        layers = [nn.Linear(self.indim, hiddim), nn.Tanh(),]
        if dropout > 0:
            layers += [nn.Dropout(p = dropout),]
        for _ in range(n_layers):
            layers += [nn.Linear(hiddim, hiddim), nn.Tanh(),]
            if dropout > 0:
                layers += [nn.Dropout(p = dropout),]
        layers += [nn.Linear(hiddim, outdim)]
        self.sequential = nn.Sequential(*layers)
    def forward(self, X):
        if self.w > 0:
            # PLEASE PELASE PLEASE fix this, this is brittle, it works for ellipsoid/cylindrical goem but it's really brittle.
            indim = X.shape[-1]
            r = torch.sqrt(X[:, 0:1]**2 + X[:, 1:2]**2)
            z = X[:, 2:3]
            sinr = torch.sin(self.frequencies * r)
            cosr = torch.cos(self.frequencies * r)
            sinz = torch.sin(self.frequencies * z)
            cosz = torch.cos(self.frequencies * z)
            sin_inp =  [sinr, sinz]
            cos_inp = [cosr, cosz]
            input_X = torch.cat(sin_inp + cos_inp, dim = -1) 
            return self.sequential(input_X)
        else: return self.sequential(X)


class PoissonMLP_generic(nn.Module):
    def __init__(self, indim = 3, hiddim = 32, outdim = 1, n_layers = 2, w = 0.0, num_frequencies = 5, dropout = 0.0):
        super().__init__()
        self.w = w
        self.num_frequencies = num_frequencies
        if self.w > 0:
            self.indim = indim * num_frequencies * 2
            self.frequencies = torch.tensor(np.geomspace(1, 1/self.w, self.num_frequencies), dtype = torch.float32).to(config.device)
        else: self.indim = indim
        layers = [nn.Linear(self.indim, hiddim), nn.Tanh(),]
        if dropout > 0:
            layers += [nn.Dropout(p = dropout),]
        for _ in range(n_layers):
            layers += [nn.Linear(hiddim, hiddim), nn.Tanh(),]
            if dropout > 0:
                layers += [nn.Dropout(p = dropout),]
        layers += [nn.Linear(hiddim, outdim)]
        self.sequential = nn.Sequential(*layers)
    def forward(self, X):
        if self.w > 0:
            indim = X.shape[-1]
            sinx = [torch.sin(self.frequencies * X[:, i:i+1]) for i in range(indim)]
            cosx = [torch.cos(self.frequencies * X[:, i:i+1]) for i in range(indim)]
            input_X = torch.cat(sinx + cosx, dim = -1)
            return self.sequential(input_X)
        else: return self.sequential(X)
