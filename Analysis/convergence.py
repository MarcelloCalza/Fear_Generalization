"""
Module: convergence.py

### Context and Source
This module is inspired by the project associated with the study:

"Humans display interindividual differences in the latent mechanisms underlying
fear generalization behaviour."

Contributors: Kenny Yu, Francis Tuerlinckx, Wolf Vanpaemel, Jonas Zaman
Affiliated institutions: KU Leuven
Identifier: DOI 10.17605/OSF.IO/SXJAK

The original R code and the JAGS models, created as part of the study, are available
at the project repository:
https://osf.io/sxjak/

The experimental study utilizes the datasets of two other studies:
- Experiment 1 simple conditioning: https://osf.io/b4ngs
- Experiment 2 differential conditioning: https://osf.io/t4bzs

### About This Module
This module performs convergence diagnostics on Bayesian CLG2D model fits for two studies:
simple conditioning (Study 1) and differential conditioning (Study 2). It loads precomputed
InferenceData objects from netCDF files, prints summary statistics (R̂, ESS) for both priors
and hyperpriors, and generates trace and pair plots for key parameter sets to visually assess
MCMC mixing and convergence.

### Functions:
- convergence_plots: Iterates over experiments and parameter-set definitions and uses ArviZ to
  plot posterior trace, density and pair plots for each relevant variable.
"""

import arviz as az
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import sys

np.set_printoptions(threshold=sys.maxsize)

pd.options.display.max_rows = 1000

# Create a directory to save the plots.
os.makedirs("../Plots", exist_ok=True)

# Load results and data for plotting.
# Load study 1 CLG2D model results.
Result_Study1_CLG2D = az.from_netcdf(
    "../Fitting_results/" "model_1v_LG2D/Results_Study1_CLG2D.nc"
)
# Load study 2 CLG2D model results.
Result_Study2_CLG2D = az.from_netcdf(
    "../Fitting_results/" "model_2v_LG2D/Results_Study2_CLG2D.nc"
)
results = {"s1": {"CLG2D": Result_Study1_CLG2D}, "s2": {"CLG2D": Result_Study2_CLG2D}}

# Define parameters for plotting and printing.
model_names = ["model_1v_LG2D", "model_2v_LG2D"]
experiments = [
    "Experiment 1 Simple Conditioning",
    "Experiment 2 differential conditioning",
]
parameter_sets = {
    "priors": ["alpha_1", "lambda_1", "lambda_2", "d_sigma", "w1_1", "w0", "pi"],
    "hyperpriors": [
        "alpha_mu",
        "alpha_kappa",
        "lambda_mu",
        "lambda_sigma",
        "w0_mu",
        "w0_sigma",
        "w1_a",
        "w1_b",
    ],
}

# Count and print number of divergences.
divergences1 = Result_Study1_CLG2D.sample_stats["diverging"].sum().item()
divergences2 = Result_Study2_CLG2D.sample_stats["diverging"].sum().item()
print(f"Number of divergences: {divergences1} {divergences2}")

# Print study 1 convergence statistics for label switching invariant parameters
# and relabelled sigma, pi.
summary1 = az.summary(
    results["s1"]["CLG2D"],
    var_names=parameter_sets["hyperpriors"]
    + ["sigma", "pi", "alpha", "lambda", "w1_1", "w0", "d_sigma"],
)
print("Convergence statistics for study 1 simple conditioning:\n", summary1)

# Print study 2 convergence statistics for label switching invariant parameters
# and relabelled sigma, pi.
summary2 = az.summary(
    results["s2"]["CLG2D"],
    var_names=parameter_sets["hyperpriors"]
    + ["sigma", "pi", "alpha", "lambda", "w1_1", "w0", "d_sigma"],
)
print("Convergence statistics for study 2 differential conditioning:\n", summary2)

# Create directory to save the plots.
os.makedirs("../Plots/Convergence", exist_ok=True)


# Convergence plotting function.
def convergence_plots(
    results, parameter_sets, experiments, output_dir="../Plots/Convergence/"
):
    """
    Generates and saves trace and pair plots for MCMC convergence diagnostics across
    multiple experiments and parameter sets.

    It uses ArviZ's `plot_trace` and `plot_pair` to produce compact trace, density and
    pair plots of the posterior samples for the specified variables.

    Args:
        results (dict):
            Dictionary of studies' model fits.
        parameter_sets (dict):
            Dictionary of parameters sets to plot.
        experiments (list):
            Name of the experiments connected to the studies.
        output_dir (str):
            Directory where the plots will be saved.
    """
    # Iterate through the studies' results.
    for experiment, result in zip(experiments, results.values()):

        # Iterate through the parameter sets.
        for param_set_name, parameters in parameter_sets.items():

            # Plot parameters' traces.
            az.plot_trace(
                result["CLG2D"],
                var_names=parameters,
                compact=True,
                combined=False,
                figsize=(8, 12),
                divergences="bottom",
            )

            plt.tight_layout()

            # Save plots in output directory.
            plt.savefig(
                f"{output_dir}/{experiment.replace(' ', '_')}_{param_set_name}_trace.png",
                dpi=300,
            )
            plt.close()

        with az.rc_context(rc={"plot.max_subplots": 100}):

            # Plot hyperparameters' pair plots.
            az.plot_pair(
                result["CLG2D"],
                var_names=parameter_sets["hyperpriors"]
                + ["sigma_mean_learners", "sigma_mean_nonlearners"],
                divergences=True,
                kind="scatter",
            )

            plt.tight_layout()

            # Save plots in output directory.
            plt.savefig(
                f"{output_dir}/{experiment.replace(' ', '_')}_pair_plots.png", dpi=300
            )
            plt.close()


# Plot both studies' convergence plots.
convergence_plots(results, parameter_sets, experiments)
