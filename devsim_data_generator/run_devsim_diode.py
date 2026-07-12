import sys, os
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import devsim
from devsim import (
    add_1d_contact,
    add_1d_mesh_line,
    add_1d_region,
    create_1d_mesh,
    create_device,
    finalize_mesh,
    get_contact_list,
    get_node_model_values,
    set_parameter,
    solve,
    delete_mesh,
    delete_device,
)
from devsim.python_packages.model_create import CreateNodeModel, CreateSolution
from devsim.python_packages.simple_physics import (
    GetContactBiasName,
    SetSiliconParameters,
    CreateSiliconPotentialOnly,
    CreateSiliconPotentialOnlyContact,
)
REGION = "Si"

def create_mesh(step_pos = 0.5e-5, device_name = "diode"):
    create_1d_mesh(mesh = "dio")
    add_1d_mesh_line(mesh = "dio", pos = 0,      ps = 1e-7, tag = "top")
    add_1d_mesh_line(mesh = "dio", pos = step_pos, ps = 1e-9, tag = "mid") # this is the position of the step potential
    add_1d_mesh_line(mesh = "dio", pos = 2*step_pos,   ps = 1e-7, tag = "bot") # this is a hardcoded number, let's check if that's a good idea later
    add_1d_contact(mesh = "dio", name = "top", tag = "top", material = "metal")
    add_1d_contact(mesh = "dio", name = "bot", tag = "bot", material = "metal")
    add_1d_region(mesh = "dio", material = "Si", region = REGION, tag1 = "top", tag2 = "bot")
    finalize_mesh(mesh = "dio")
    create_device(mesh = "dio", device=device_name)
    delete_mesh(mesh = "dio")


def set_doping(step_pos = 0.5e-5, doping = 1e18, device_name = "diode"):
    # step junction: p-type left half, n-type right half, 1e18 cm^-3 each
    CreateNodeModel(device_name, REGION, "Acceptors", "%s*step(%s-x)" %(doping, step_pos))
    CreateNodeModel(device_name, REGION, "Donors",    "%s*step(x-%s)" %(doping, step_pos))
    CreateNodeModel(device_name, REGION, "NetDoping", "Donors-Acceptors")


def solve_potential_only(device_name = "diode"):
    SetSiliconParameters(device_name, REGION, 300)
    CreateSolution(device_name, REGION, "Potential")
    CreateSiliconPotentialOnly(device_name, REGION)
    for contact in get_contact_list(device=device_name):
        set_parameter(device=device_name, name=GetContactBiasName(contact), value=0.0)
        CreateSiliconPotentialOnlyContact(device_name, REGION, contact)
    # zero bias DC solve - this is the equilibrium Poisson-only solution    
    solve(type="dc", absolute_error=1.0, relative_error = 1e-8, maximum_iterations = 30)



def extract_and_save(output_file = "test.npz", device_name = "diode"):
    x         = np.array(get_node_model_values(device=device_name, region=REGION, name="x"))
    potential = np.array(get_node_model_values(device=device_name, region=REGION, name="Potential"))
    net_doping= np.array(get_node_model_values(device=device_name, region=REGION, name="NetDoping"))

    # identify boundary nodes: first and last node are the contacts
    # devsim 1d meshes are sorted by x
    sort_idx  = np.argsort(x)
    x         = x[sort_idx]
    potential = potential[sort_idx]
    net_doping= net_doping[sort_idx]

    bc_indices = np.array([0, len(x) - 1])
    bc_x       = x[bc_indices]
    bc_pot     = potential[bc_indices]

    print(f'computed length of device is {2*np.max(np.abs(x))}')
    np.savez(
        output_file,
        length          = x.max() - x.min(), # this is hard coded, one has to check if this is right
        nD0             = max(net_doping),
        x          = x,
        potential  = potential,
        net_doping = net_doping,
        bc_x       = bc_x,
        bc_pot     = bc_pot,
    )
    print(f"Saved {len(x)} nodes to {output_file}")
    print(f"x range: {x.min():.3e} to {x.max():.3e} cm")
    print(f"Potential range: {potential.min():.4f} to {potential.max():.4f} V")
    print(f"NetDoping range: {net_doping.min():.3e} to {net_doping.max():.3e} cm^-3")
    print(f"BC potentials: top={bc_pot[0]:.4f} V, bot={bc_pot[1]:.4f} V")


def generate_samples(L_in = [1e-6, 1e-4], nD_in = [1e15, 1e19], sample_type = 'train'):
    for L, nD in zip(L_in, nD_in):
        manifest_file = open('data/records.csv', 'a+')
        device_name = f"L{L:.2e}_N{nD:.2e}_{sample_type}"
        OUTPUT_FILE = f"data/devsim_diode_1d_diode_{device_name}.npz" 
        manifest_file.write('%s\t%16.8e\t%16.8e\t%s\n' %(OUTPUT_FILE, L, nD, sample_type))
        step_position = L/2
        create_mesh(step_pos = step_position, device_name = device_name)
        set_doping(step_pos = step_position, doping = nD, device_name = device_name)
        solve_potential_only(device_name = device_name)
        extract_and_save(output_file = OUTPUT_FILE, device_name = device_name)
        delete_device(device = device_name)
        manifest_file.close()

# decided sweep ranges 1e15 to 1e19 cm^-3
# L: 1e-5 to 1e-3 cm (100 nm – 10 microm) 
if __name__ == "__main__":
    os.makedirs('data', exist_ok = True)
    manifest_file = open('data/records.csv', 'w') 
    manifest_file.write('filename\tL\tnD0\tsplit\n')
    manifest_file.close() 
    
    L_min = 1e-5 
    L_max = 1e-3
    nD_min = 1e15
    nD_max = 1e19

    n_set = int(70) # total number of points to be generated, half will be used for training, other half for test
    sampler = stats.qmc.LatinHypercube(d = 2)
    sample = sampler.random(n = n_set)
    # let me get indices to split train and test
    rng = np.random.default_rng()
    n_train = 15
    n_test = n_set - n_train
    train_indices = rng.choice(n_set, size = n_train, replace = False)
    test_indices = np.ones(n_set, dtype = np.bool)
    test_indices[[train_indices]] = False

    # let me now filter the sample array
    sample_test = sample[test_indices]
    sample_train = sample[train_indices]
    plt.scatter(sample_test[:, 0], sample_test[:, 1], label = 'test configurations')
    plt.scatter(sample_train[:, 0], sample_train[:, 1], label = 'train configurations')
    plt.legend()
    plt.savefig('results/train_test_regions.png')
    plt.close()
    if not (len(sample_train) + len(sample_test) == n_set): print(f'train test split is not done correctly will exit'); sys.exit()
    if not (set(train_indices) & set(np.where(test_indices)[0]) == set()): print(f'common train test elements found, will exit now'); sys.exit()
    t0 = time.time()
    log10_nD = np.log10(nD_min) + sample_train[:,0]*(np.log10(nD_max) - np.log10(nD_min))
    nD = 10**log10_nD
    log10_L = np.log10(L_min) + sample_train[:,1]*(np.log10(L_max) - np.log10(L_min))
    L = 10**log10_L
    generate_samples(L_in = L, nD_in = nD, sample_type = 'train')
    print('generated train samples')

    log10_nD = np.log10(nD_min) + sample_test[:,0]*(np.log10(nD_max) - np.log10(nD_min))
    nD = 10**log10_nD
    log10_L = np.log10(L_min) + sample_test[:,1]*(np.log10(L_max) - np.log10(L_min))
    L = 10**log10_L
    generate_samples(L_in = L, nD_in = nD, sample_type = 'test')
    print('generated test samples')
    devsim_solve_time = time.time() - t0
    print(f"Devsim solve time: {devsim_solve_time*1000:.2f} ms")
