# Digitial surrogate for TCAD Semiconductor Device Simulator and framework for deployment criteria

## Primary result
Our digital surrogate becomes cheaper than the TCAD simulator for larger than 21k queries. 

![alt text](https://github.com/suchitakulkarni/DevSim_public/blob/main/results/results_plw_1.000e%2B01_bclw_1.000e-02/cost_sensitivity.png)

The surrogates deliveres results at < 10% accuracy over device lenghts of $0.1 micron < L < 3.16 micron$ and doping concentrations of $10^5 < n_{D_0} < 10^7$. We explicitly mark areas where the current model is not valid and further improvements are necessary. 

![alt text](https://github.com/suchitakulkarni/DevSim_public/blob/main/results/results_plw_1.000e%2B01_bclw_1.000e-02/generalization_map.png)

## Problem statement
Digitial surrogates designed using e.g., physics-inspired neural network, for TCAD Semiconductor Device Simulator are fast and can be efficient in industrial workflows. In this work we develop such a digital surrogate and estimate the number of queries beyond which the digital surrogate becomes cheaper than the full TCAD simulator. 