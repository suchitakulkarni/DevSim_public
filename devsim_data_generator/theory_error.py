import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton_krylov, NoConvergence
from scipy.interpolate import interp1d
import os
import time

os.makedirs("results", exist_ok=True)

# physical constants
q         = 1.602e-19
k_B       = 1.381e-23
T         = 300.0
ni        = 1.5e10
epsilon_0 = 8.854e-14
epsilon_r = 11.7
V_T       = k_B * T / q


class PoissonBoltzmann1D:
    """
    Solves the 1D nonlinear Poisson-Boltzmann equation:

        d2psi/dx2 + gamma^2 * (nD(x) - 2*sinh(psi)) = 0

    on a uniform grid with Dirichlet BCs at both ends.
    psi = phi / V_T  (dimensionless potential)
    nD  = NetDoping / ni (dimensionless doping)
    x   = physical coordinate / L (normalised to [0,1])

    Uses the same gamma-continuation strategy as CylinderCharges._solve()
    in poisson.py to handle the nonlinearity.
    """

    def __init__(self, x_norm, nD_norm, psi_left, psi_right, gamma, n_grid=500):
        self.gamma     = gamma
        self.psi_left  = psi_left
        self.psi_right = psi_right

        # build uniform grid and interpolate nD onto it
        self.x_grid = np.linspace(0.0, 1.0, n_grid)
        nD_interp   = interp1d(x_norm, nD_norm, kind="linear",
                               bounds_error=False, fill_value=0.0)
        self.nD_grid = nD_interp(self.x_grid)
        self.dx      = self.x_grid[1] - self.x_grid[0]
        self.psi     = None

    def _residual(self, psi):
        dx   = self.dx
        R    = np.zeros_like(psi)

        # interior: standard second-order finite difference Laplacian
        d2psi = (psi[2:] - 2*psi[1:-1] + psi[:-2]) / dx**2
        R[1:-1] = d2psi + self.gamma**2 * (self.nD_grid[1:-1] - 2*np.sinh(psi[1:-1]))

        # Dirichlet BCs: residual is just the boundary mismatch
        R[0]  = psi[0]  - self.psi_left
        R[-1] = psi[-1] - self.psi_right
        return R

    def solve(self):
        # gamma continuation: ramp from small gamma to full gamma
        # same strategy as CylinderCharges._solve()
        gamma_ramp    = np.geomspace(1e-3, self.gamma, 80)
        current_guess = np.zeros(len(self.x_grid))

        # seed BCs into initial guess via linear ramp
        current_guess = np.linspace(self.psi_left, self.psi_right, len(self.x_grid))

        saved_gamma = self.gamma
        for g in gamma_ramp:
            self.gamma = g
            try:
                current_guess = newton_krylov(
                    self._residual, current_guess, maxiter=400, f_tol=1e-8
                )
            except NoConvergence as e:
                current_guess = e.args[0]

        self.gamma = saved_gamma
        self.psi   = current_guess
        return self.psi


def quantify_theory_error(npz_path="devsim_diode_1d.npz"):

    data      = np.load(npz_path)
    x_raw     = data["x"]
    pot_raw   = data["potential"]
    nD_raw    = data["net_doping"]
    bc_pot    = data["bc_pot"]

    L         = x_raw.max() - x_raw.min()
    x_norm    = (x_raw - x_raw.min()) / L
    psi_devsim= pot_raw / V_T
    nD_norm   = nD_raw / ni

    # boundary conditions in dimensionless units — taken directly from Devsim
    psi_left  = bc_pot[0] / V_T
    psi_right = bc_pot[1] / V_T

    L_D   = np.sqrt(epsilon_r * epsilon_0 * k_B * T / (q**2 * ni))
    gamma = L / L_D
    print(f"gamma = {gamma:.2f}")
    print(f"Solving 1D Poisson-Boltzmann on {500} grid points...")

    t0  = time.time()
    pb  = PoissonBoltzmann1D(x_norm, nD_norm, psi_left, psi_right, gamma, n_grid=500)
    psi_pb = pb.solve()
    pb_time = time.time() - t0
    print(f"PB solve time: {pb_time*1000:.2f} ms")

    # interpolate Devsim solution onto the PB uniform grid for clean comparison
    devsim_interp = interp1d(x_norm, psi_devsim, kind="cubic")
    psi_devsim_on_grid = devsim_interp(pb.x_grid)

    theory_error_mV = (psi_pb - psi_devsim_on_grid) * V_T * 1e3
    x_um = pb.x_grid * L * 1e4   # normalised -> micrometers

    print(f"Max theory error: {np.max(np.abs(theory_error_mV)):.2f} mV")
    print(f"RMS theory error: {np.sqrt(np.mean(theory_error_mV**2)):.2f} mV")
    print(f"Peak Devsim potential: {np.max(np.abs(psi_devsim_on_grid)) * V_T * 1e3:.2f} mV")

    # three panel plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(x_um, psi_devsim_on_grid * V_T * 1e3, label="Devsim (drift-diffusion)", linewidth=2)
    axes[0].plot(x_um, psi_pb * V_T * 1e3, label="Poisson-Boltzmann (numerical)", linewidth=1.5, linestyle="--")
    axes[0].set_xlabel("x (um)")
    axes[0].set_ylabel("Potential (mV)")
    axes[0].set_title("Potential profiles")
    axes[0].legend()

    axes[1].plot(x_um, theory_error_mV, color="crimson", linewidth=1.5)
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("x (um)")
    axes[1].set_ylabel("Error (mV)")
    axes[1].set_title(f"Theory error: PB minus Devsim\n(max={np.max(np.abs(theory_error_mV)):.2f} mV, "
                      f"RMS={np.sqrt(np.mean(theory_error_mV**2)):.2f} mV)")

    # doping profile for context
    ax2 = axes[2]
    nD_on_grid = pb.nD_grid * ni
    ax2.plot(x_um, nD_on_grid / 1e18, color="steelblue", linewidth=1.5)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("x (um)")
    ax2.set_ylabel("NetDoping (x 1e18 cm^-3)")
    ax2.set_title("Doping profile (context)")

    fig.suptitle("1D Poisson-Boltzmann vs Devsim: theory error quantification", y=1.02)
    fig.tight_layout()
    fig.savefig("results/theory_error_PB_vs_devsim.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved results/theory_error_PB_vs_devsim.png")

    return pb.x_grid, psi_pb, psi_devsim_on_grid, theory_error_mV


if __name__ == "__main__":
    quantify_theory_error()
