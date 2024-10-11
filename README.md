# Membrane Fouling Model Fitting

This repository contains code that implements and fits five different membrane fouling models to experimental data. The resulting fitted models can then be used to make predictions in similar systems regarding process duration and membrane sizing. The models are adapted from Bolton, LaCasse, and Kuriyel (2006) and are used to analyze flux decay in membrane filtration systems for a constant transmembrane pressure. Experimental data for the time, volume filtered, and initial volumetric flow rate for the original testing and implementation was obtained from the dataset provided by Mayani et al. (2023) for lentiviral vector clarification using depth filtration.

## Models Implemented

1. Cake-complete
2. Cake-intermediate
3. Cake-standard
4. Complete-standard
5. Intermediate-standard

## Features
- Fits experimental time and volume data to five distinct membrane fouling models.
- Outputs the best-fit model parameters and calculates the mean squared error (MSE) for each model.
- Plots experimental data alongside fitted models to visualize the performance of each fouling model.

## Data Requirements
The CSV file should include time (s) and volume filtered (m³) data from a membrane filtration experiment. Within the `model_fit.py` script you will also need to provide an initial volumetric flow rate value, J0 (m³/s). 

## References
- Bolton, G., LaCasse, D., & Kuriyel, R. (2006). Combined models of membrane fouling: Development and application to microfiltration and ultrafiltration of biological fluids. *Journal of Membrane Science*, 277(1-2). doi:[10.1016/j.memsci.2004.12.053](https://doi.org/10.1016/j.memsci.2004.12.053)
- Mayani, M. et al. (2023). Depth filtration for clarification of intensified lentiviral vector suspension cell culture. *Biotechnology Progress*, 40(2). doi:[10.1002/btpr.3409](https://doi.org/10.1002/btpr.3409)
