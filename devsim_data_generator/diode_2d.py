# Copyright 2013 DEVSIM LLC
#
# SPDX-License-Identifier: Apache-2.0

from devsim import set_parameter, solve

from devsim.python_packages.simple_physics import GetContactBiasName, PrintCurrents
import diode_common
from run_devsim_diode import extract_and_save

# dio1
#
# Make doping a step function
# print dat to text file for viewing in grace
# verify currents analytically
# in dio2 add recombination
#

device = "MyDevice"
region = "Si"

diode_common.Create2DMesh(device, region)

diode_common.SetParameters(device=device, region=region)

diode_common.SetNetDoping(device=device, region=region)

diode_common.InitialSolution(device, region)

# Initial DC solution
solve(type="dc", absolute_error=1.0, relative_error=1e-12, maximum_iterations=30)

v = 0.0
set_parameter(device=device, name=GetContactBiasName("top"), value=v)
solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=30)
extract_and_save(output_file = "test_2d.npz", device_name = device, dims = "2d")
