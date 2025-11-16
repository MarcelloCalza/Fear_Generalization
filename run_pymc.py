"""
Module: run_pymc.py

### Context and Source
This module is inspired by the project associated with the study:

"Humans display interindividual differences in the latent mechanisms underlying
fear generalization behaviour."

Contributors: Kenny Yu, Francis Tuerlinckx, Wolf Vanpaemel, Jonas Zaman
Affiliated institutions: KU Leuven
Identifier: DOI 10.17605/OSF.IO/SXJAK

The original R code and the JAGS models, created as part of
the study, are available at the project repository:
https://osf.io/sxjak/

The experimental study utilizes the datasets of two other studies:
experiment 1 simple conditioning (https://osf.io/b4ngs),
exepriment 2 differential conditioning (https://osf.io/t4bzs)
These studies were conducted by one of the contributors: Jonas Zaman.

### About This Module
This Python module serves as a general conversion of the original R code, focusing
on managing the Bayesian sampling process for fear generalization models.
It executes model sampling based on model definitions, manages output, and handles
various data inputs effectively.

The module ensures compatibility and reproducibility with Python-based workflows
and includes functionality for dynamic import of model modules, execution of model
sampling, creation and storage of model graphs, and storage of fitting results.

### Functions:
- `sampling_fun`: Executes the Bayesian sampling process for a specified model,
  handles directory creation for outputs, and manages the saving of model graphs
  and fitting results.
"""

import os
import pymc as pm
import arviz as az
import importlib
import sys
import platform
import pickle
import numpy as np


def random_gp_init(n_participants, n_groups, rng=None):
    """
    Returns an array of shape (number of participants, ) with random
    groups in [0..3], ensuring at least 1 participant in each group.
    """
    if rng is None:
        rng = np.random.default_rng()
    # draw from 0..n_groups-1
    gp_init = rng.integers(0, n_groups, size=n_participants)
    # force at least one of each
    for g in range(min(n_groups, n_participants)):
        gp_init[g] = g
    return gp_init


# Print the path to the Python interpreter.
print(f"Python Interpreter Path: {sys.executable}")

# Print the Python version.
print(f"Python Version: {platform.python_version()}")

# Print the PyMC version.
print(f"PyMC version: {pm.__version__}")


def sampling_fun(
    datafile, model_module_name, model_name, n_chains=4, tune=1000, draws=1000, seed=42
):
    """
    Execute model sampling for Bayesian inference using PyMC.

    This function dynamically imports a model definition from a specified module,
    creates and initializes the model with provided data, and then proceeds with
    Bayesian sampling to generate posterior distributions. It also handles the
    creation of necessary directories for saving the model's graphical representation
    and fitting results.

    The sampling settings are pre-configured to optimize the convergence and accuracy
    of the model's parameters estimation. Additionally, the function computes and
    stores the posterior predictive distribution and can optionally calculate and
    store the log-likelihood for each model.

    Args:
        datafile (dict):
            The path to the dictionary containing the data needed by the model.
        model_module_name (str):
            The name of the Python module where the model function is defined.
        model_name (str):
            The name of the function within the module that returns the PyMC model.

    Returns:
        idata (arviz.InferenceData):
            An object containing the posterior samples, posterior predictive samples,
            and optionally the log-likelihoods.
    """
    # Create directories for saving model graphs and fitting results.
    os.makedirs("Model_graphs", exist_ok=True)
    os.makedirs(f"Fitting_results/{model_name}", exist_ok=True)

    # Dynamically import the model definition module.
    model_module = importlib.import_module(model_module_name)

    # Get the specific model function from the module.
    model_func = getattr(model_module, model_name)

    # Use the function to create the model.
    model = model_func(datafile)

    # Create and save the model graph.
    graph = pm.model_to_graphviz(model)
    filename = f"Model_graphs/{model_name}_graph"
    graph.render(filename, format="pdf", cleanup=True)

    # Debug the model before sampling.
    model.debug()

    with model:

        # Set seed for reproducibility.
        rng = np.random.default_rng(seed)

        # Set up the Gibbs categorical sampler for gp.
        gibbs_step = pm.CategoricalGibbsMetropolis(
            vars=[model.gp], proposal="uniform", rng=rng
        )

        # Set up the NUTS sampler for the continuous variables.
        nuts_step = pm.NUTS(
            vars=[rv for rv in model.free_RVs if rv.name not in ("gp")],
            max_treedepth=12,
            target_accept=0.93,
        )

        # gp initialization.
        gp_init_1 = random_gp_init(
            datafile["Nparticipants"],
            model.initial_point()["pi_simplex__"].shape[0] + 1,
            rng,
        )

        # Sample.
        idata = pm.sample(
            tune=tune,
            draws=draws,
            chains=n_chains,
            cores=n_chains,
            step=[nuts_step, gibbs_step],
            initvals={"gp": gp_init_1},
            init="jitter+adapt_diag_grad",
            random_seed=rng,
            compile_kwargs=dict(mode="NUMBA"),
            progressbar=True,
            return_inferencedata=True,
            compute_convergence_checks=True,
            discard_tuned_samples=True,
        )

        # Compute posterior predictive distribution.
        posterior_predictive = pm.sample_posterior_predictive(
            idata, var_names=["y_pre"]
        )

        """# Compute log-likelihood.
        log_likelihood = pm.compute_log_likelihood(idata, var_names=["y_pre"])"""

        # Add predition to the results.
        idata.add_groups(
            {"posterior_predictive": posterior_predictive.posterior_predictive}
        )
        """idata.add_groups({'log_likelihood': log_likelihood.log_likelihood})"""

    return idata


if __name__ == "__main__":

    # Sample and save fitting results for experiment 2: differential conditioning.
    # Sample with model: LG2D.
    Result_Study2_CLG2D = sampling_fun(
        pickle.load(open("PYMC_input_data/Data2_PYMCinput_CLG.pkl", "rb")),
        "models_definitions",
        "model_2v_LG2D",
        draws=3000,
        tune=3000,
        n_chains=4,
    )
    # Save results.
    az.to_netcdf(
        Result_Study2_CLG2D,
        "Fitting_results/model_2v_LG2D/Results_Study2_CLG2D.nc",
    )

    # Sample and save fitting results for experiment 1: simple conditioning.-
    # Sample with model: LG2D.
    Result_Study1_CLG2D = sampling_fun(
        pickle.load(open("PYMC_input_data/Data1_PYMCinput_CLG.pkl", "rb")),
        "models_definitions",
        "model_1v_LG2D",
        draws=10000,
        tune=10000,
        n_chains=4,
    )
    # Save results.
    az.to_netcdf(
        Result_Study1_CLG2D, "Fitting_results/model_1v_LG2D/Results_Study1_CLG2D.nc"
    )

    # Sample and save fitting results for the simulation.
    # Sample with full model: CLG2D.
    Result_Sim_CLG2D = sampling_fun(
        pickle.load(open("PYMC_input_data/Sim_PYMCinput_CLG.pkl", "rb")),
        "models_definitions",
        "model_2v_LG2D",
        draws=3000,
        tune=3000,
        n_chains=4,
    )
    # Save results.
    az.to_netcdf(Result_Sim_CLG2D, "Fitting_results/model_2v_LG2D/Result_Sim_CLG2D.nc")
    # Sample with simplified model: LGPHY.
    Result_Sim_CLGPHY = sampling_fun(
        pickle.load(open("PYMC_input_data/Sim_PYMCinput_CLG.pkl", "rb")),
        "models_definitions",
        "model_2v_LGPHY",
        draws=3000,
        tune=3000,
        n_chains=4,
    )
    # Save results.
    az.to_netcdf(
        Result_Sim_CLGPHY, "Fitting_results/model_2v_LGPHY/Result_Sim_CLGPHY.nc"
    )
    # Sample with simplified model: G2D.
    Result_Sim_G2D = sampling_fun(
        pickle.load(open("PYMC_input_data/Sim_PYMCinput_G.pkl", "rb")),
        "models_definitions",
        "model_2v_G2D",
        draws=3000,
        tune=3000,
        n_chains=4,
    )
    # Save results.
    az.to_netcdf(Result_Sim_G2D, "Fitting_results/model_2v_G2D/Result_Sim_G2D.nc")
