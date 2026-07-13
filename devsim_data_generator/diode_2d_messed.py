# Copyright 2013 DEVSIM LLC
#
# SPDX-License-Identifier: Apache-2.0

from devsim import set_parameter, solve

from devsim import (
    add_1d_contact,
    add_1d_mesh_line,
    add_1d_region,
    add_2d_contact,
    add_2d_mesh_line,
    add_2d_region,
    add_gmsh_contact,
    add_gmsh_region,
    create_1d_mesh,
    create_2d_mesh,
    create_device,
    create_gmsh_mesh,
    finalize_mesh,
    get_contact_list,
    set_node_values,
    set_parameter,
)

from devsim.python_packages.model_create import CreateNodeModel, CreateSolution

from devsim.python_packages.simple_physics import (
    GetContactBiasName,
    SetSiliconParameters,
    CreateSiliconPotentialOnly,
    CreateSiliconPotentialOnlyContact,
    CreateSiliconDriftDiffusion,
    CreateSiliconDriftDiffusionAtContact,
)

from devsim.python_packages.simple_physics import GetContactBiasName, PrintCurrents
import diode_common

# dio1
#
# Make doping a step function
# print dat to text file for viewing in grace
# verify currents analytically
# in dio2 add recombination
#

device = "diode_2d"
region = "MyRegion"

def Create2DMesh(device, region):
    create_2d_mesh(mesh="dio")
    add_2d_mesh_line(mesh="dio", dir="x", pos=0, ps=1e-6)
    add_2d_mesh_line(mesh="dio", dir="x", pos=0.5e-5, ps=1e-8)
    add_2d_mesh_line(mesh="dio", dir="x", pos=1e-5, ps=1e-6)
    add_2d_mesh_line(mesh="dio", dir="y", pos=0, ps=1e-6)
    add_2d_mesh_line(mesh="dio", dir="y", pos=1e-5, ps=1e-6)

    add_2d_mesh_line(mesh="dio", dir="x", pos=-1e-8, ps=1e-8)
    add_2d_mesh_line(mesh="dio", dir="x", pos=1.001e-5, ps=1e-8)

    add_2d_region(mesh="dio", material="Si", region=region)
    add_2d_region(mesh="dio", material="Si", region="air1", xl=-1e-8, xh=0)
    add_2d_region(mesh="dio", material="Si", region="air2", xl=1.0e-5, xh=1.001e-5)

    add_2d_contact(
        mesh="dio",
        name="top",
        material="metal",
        region=region,
        yl=0.8e-5,
        yh=1e-5,
        xl=0,
        xh=0,
        bloat=1e-10,
    )
    add_2d_contact(
        mesh="dio",
        name="bot",
        material="metal",
        region=region,
        xl=1e-5,
        xh=1e-5,
        bloat=1e-10,
    )

    finalize_mesh(mesh="dio")
    create_device(mesh="dio", device=device)

def SetParameters(device, region):
    """
    Set parameters for 300 K
    """
    SetSiliconParameters(device, region, 300)


def SetNetDoping(device, region):
    """
    NetDoping
    """
    CreateNodeModel(device, region, "Acceptors", "1.0e18*step(0.5e-5-x)")
    CreateNodeModel(device, region, "Donors", "1.0e18*step(x-0.5e-5)")
    CreateNodeModel(device, region, "NetDoping", "Donors-Acceptors")

def InitialSolution(device, region, circuit_contacts=None):
    # Create Potential, Potential@n0, Potential@n1
    CreateSolution(device, region, "Potential")

    # Create potential only physical models
    CreateSiliconPotentialOnly(device, region)

    # Set up the contacts applying a bias
    for i in get_contact_list(device=device):
        if circuit_contacts and i in circuit_contacts:
            CreateSiliconPotentialOnlyContact(device, region, i, True)
        else:
            ###print "FIX THIS"
            ### it is more correct for the bias to be 0, and it looks like there is side effects
            set_parameter(device=device, name=GetContactBiasName(i), value=0.0)
            CreateSiliconPotentialOnlyContact(device, region, i)

def DriftDiffusionInitialSolution(device, region, circuit_contacts=None):
    ####
    #### drift diffusion solution variables
    ####
    CreateSolution(device, region, "Electrons")
    CreateSolution(device, region, "Holes")

    ####
    #### create initial guess from dc only solution
    ####
    set_node_values(
        device=device, region=region, name="Electrons", init_from="IntrinsicElectrons"
    )
    set_node_values(
        device=device, region=region, name="Holes", init_from="IntrinsicHoles"
    )

    ###
    ### Set up equations
    ###
    CreateSiliconDriftDiffusion(device, region)
    for i in get_contact_list(device=device):
        if circuit_contacts and i in circuit_contacts:
            CreateSiliconDriftDiffusionAtContact(device, region, i, True)
        else:
            CreateSiliconDriftDiffusionAtContact(device, region, i)

def generate_samples(L_xin = [1e-6, 1e-4],L_yin = [1e-6, 1e-4],  nD_in = [1e15, 1e19], sample_type = 'train'):
    for Lx, Ly, nD in zip(Lx_in, Ly_in, nD_in):
        manifest_file = open('data/records_2d.csv', 'a+')
        device_name = f"Lx{Lx:.2e}_Ly{Ly:.2e}_N{nD:.2e}_{sample_type}"
        OUTPUT_FILE = f"data/devsim_diode_1d_diode_{device_name}.npz"
        manifest_file.write('%s\t%16.8e\t%16.8e\t%16.8e\t%s\n' %(OUTPUT_FILE, Lx, Ly, nD, sample_type))
        #step_position = L/2
        Create2DMesh(device, region)
        #create_mesh(step_pos = step_position, device_name = device_name)
        SetNetDoping(device = device_name, region = region)
        solve_potential_only(device_name = device_name)
        extract_and_save(output_file = OUTPUT_FILE, device_name = device_name)
        delete_device(device = device_name)
        manifest_file.close()

Create2DMesh(device, region)

SetParameters(device=device, region=region)

SetNetDoping(device=device, region=region)

InitialSolution(device, region)

# Initial DC solution
solve(type="dc", absolute_error=1.0, relative_error=1e-12, maximum_iterations=30)

DriftDiffusionInitialSolution(device, region)
###
### Drift diffusion simulation at equilibrium
###
solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=30)

####
#### solve for minBias
####
v = 0.0
set_parameter(device=device, name=GetContactBiasName("top"), value=v)
solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=30)
PrintCurrents(device, "top")
PrintCurrents(device, "bot")

val = 10
for i in range(2):
    set_parameter(device=device, name=GetContactBiasName("top"), value=val)
    data = solve(
        type="dc",
        absolute_error=1e10,
        relative_error=1e-10,
        maximum_iterations=30,
        info=True,
    )
    print(data["converged"])
    if not data["converged"]:
        val = 0.6

#print(data)
for i in data["iterations"]:
    for d in i["devices"]:
        for r in d["regions"]:
            for e in r["equations"]:
                #print(e)
                if e['name'] == "PotentialEquation": print(e)
