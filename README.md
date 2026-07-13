# Digital surrogate for TCAD Semiconductor Device Simulator and framework for deployment criteria

## Primary result
Our digital surrogate becomes cheaper than the TCAD simulator for larger than 21k queries. 

<img src="https://github.com/suchitakulkarni/DevSim_public/blob/main/results/results_plw_1.000e%2B01_bclw_1.000e-02/cost_sensitivity.png" width="600" />

The cost sensitivity is not a flat estimate, it depends on the complexity of the solver and the train time of the surrogate. The deployment benefit is quickly reached for more complex 2D/3D solvers which will be analysed in the near future. 

Our surrogate delivers results at < 10% accuracy over device lengths of $0.1\rm{micron} < L < 3.16 \rm{micron}$ and doping concentrations of $10^5 < n_{D_0}/n_i < 10^7$ for a 1D diode. We explicitly mark areas where the current model is not valid and further improvements are necessary. Generalisations beyond 1D diodes are foreseen in code design.

Here is a dense heatmap representation of generalisation result
![Model animation](https://github.com/suchitakulkarni/DevSim_public/blob/main/results/results_plw_1.000e%2B01_bclw_1.000e-02/potential_sweep.gif)

![Model generalisation](https://github.com/suchitakulkarni/DevSim_public/blob/main/results/results_plw_1.000e%2B01_bclw_1.000e-02/generalization_heatmap_rms.png)

This is a scatter plot, for visual comparison
![Model generalisation](https://github.com/suchitakulkarni/DevSim_public/blob/main/results/results_plw_1.000e%2B01_bclw_1.000e-02/generalization_map.png)

## Problem statement
Digital surrogates designed using e.g., physics-informed neural network, for TCAD Semiconductor Device Simulator are fast and can be efficient in industrial workflows. In this work we develop such a digital surrogate and estimate the number of queries beyond which the digital surrogate becomes cheaper than the full TCAD simulator. 

## Methodology
We implement non-linear Poisson-Boltzmann equations in a physics-informed neural network along with data generated using [DevSim](https://devsim.org/index.html) as a standin for industrial TCAD simulators. In total we create 70 datasets, out of which 30 are used for training and rest are used for inference. The network is supplemented with dedicated geometry sampler which creates collocation points grid, where oversampling is implemented at the device junction. 

## Physics importance
<img src="https://github.com/suchitakulkarni/DevSim_public/blob/main/results/devsim_pinn_error_profile_w_physics.png" width="600" />
<img src="https://github.com/suchitakulkarni/DevSim_public/blob/main/results/devsim_pinn_error_profile_no_physics.png" width="600" />
The two images show $~9x$ error reduction at the junction, showing importance of including physics in the simulations.

### In progress 2D results 
<img src="https://github.com/suchitakulkarni/DevSim_public/blob/main/results/devsim_pinn_potential_profile.png"/>
This result is for one point and an initial training, it shows that the script is fully wired for a single point 2D run. It has not been yet wired for a multi anchor run, i.e. data genereator needs to be wired in, but the first results are promising.

## How to run
* Use `pip install -r requirements.txt`, this will automatically install `DevSim`.
* The repository contains a single datafile which can be used to run 
`python ablation_physics_1d.py`
This allows to tune physics weight for one datapoint.
* In case further analysis is desired, the code needs `DevSim`. The easiest way to install DevSim is 
`pip install devsim`. 
* Once installed run 
`python devsim_data_generator/run_devsim_diode.py`
* Two scripts `train_multi_anchor_pinn.py` and `evaluate_multi_anchor.py` will create the physics results. Alternatively, there are CUDA trained model check points, which can be loaded only for inference purposes and training can be skipped. All hyper-parameters are mentioed in the `src/config.py` file.
* Adjust numbers in `cost_model.py` script to generate the breakeven cost estimates for your scenario. 

## Note:
* Plotting scripts were generated using claude.
