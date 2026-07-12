import torch
import os, logging
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# please please please be careful, this is a bit brittle, indim = physical dimensions + gamma, nD
indim = 3
hiddim = 128 
outdim = 1
n_layers = 4 
dropout = 0
early_stopping = False # early stopping true on a single trajectory showed that loss curves remain flat for a long time and early stopping is a bad idea
patience = 100
lr = 5e-5
#n_epochs           = 3000
n_epochs           = 3000
n_obs_per_anchor   = 20     # sparse per anchor - total obs = n_obs_per_anchor * n_train_anchors
n_colloc_per_anchor = 200
n_bc               = 2 # do not change this for 1D diode, there are only two boundary points nothing more
obs_batch          = 32
lambda_phy         = 10
lambda_bc          = 1e-2
os.makedirs("results", exist_ok=True)
results_dir = 'results/results_plw_%0.3e_bclw_%0.3e' %(lambda_phy, lambda_bc)
os.makedirs(results_dir, exist_ok = True)


def setup_logging(level=logging.INFO):
    """
    Call once from main.py. All modules use logging.getLogger(__name__)
    and inherit this configuration automatically.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(results_dir, "run.log"), mode="a")
        ]
    )
