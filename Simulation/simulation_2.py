"""
Module: simulation_2.py

### Context and Source
This module is inspired by the project associated with the study:

"Humans display interindividual differences in the latent mechanisms underlying
fear generalization behaviour."

Contributors: Kenny Yu, Francis Tuerlinckx, Wolf Vanpaemel, Jonas Zaman
Affiliated institutions: KU Leuven
Identifier: DOI 10.17605/OSF.IO/SXJAK

The original R code and the JAGS models, created as part of the study, are available at
the project repository:
https://osf.io/sxjak/

The experimental study utilizes the datasets of two other studies:
- Experiment 1 simple conditioning: https://osf.io/b4ngs
- Experiment 2 differential conditioning: https://osf.io/t4bzs

### About This Module
This Python module is an independent implementation that converts the original R
simulation code into Python. It generates synthetic data of 200 participants across the
four latent groups, simulates their learning and generalization behavior, fits Bayesian
models to the simulated data to assess parameter recovery, and provides a suite of
visualization tools for exploratory analysis.

### Functions:
- decay_plot: Generates two subplots illustrating how similarity decays as a function of
  generalization rate (lambda) and across a range of distances.
- extract_variable: Extracts a 1D variable from long-format data, reshapes it to 1xNtrials
  and repeats it for all participants.
- lg_fun: Simulates the trial-by-trial learning process, computing associative strengths,
  perceptual distances, similarities, and final responses for each participant, then
  returns a long-format DataFrame.
- lr_plot: Plots associative strength trajectories over learning trials, showing
  individual mean participant lines and group means.
- sim_plot: Displays physical and perceptual similarity to the CS+ across adjusted
  distances for each participant group.
- ge_plot: Plots overall generalized responses (simulated US expectancies) for all
  stimuli, combining group-level and individual means.
- ge_gr_plot: Creates a grid of plots showing generalized responses separately for each
  latent group.
- prepare_data_for_pymc: Constructs and slices the simulation data into dictionaries
  suitable for PyMC model fitting, optionally selecting only generalization or the full
  trial set.
- recovery_plot: Produces a scatter plot comparing true versus inferred parameter values
  for a single model and parameter.
- all_recovery_plot: Assembles a figure (including zoomed inset) to display recovery
  performance across models.
- vs_recovery_plot: Generates side-by-side comparison plots of lambda and alpha true versus
  inferred values.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from functools import reduce
import matplotlib.ticker as ticker
import pickle
from run_pymc import sampling_fun
import arviz as az
from matplotlib.gridspec import GridSpec
from data_utils import ReshapedInferenceData
from group_allocation import gp_allocation, plot_gp_allocation

# Part 2: 200 hypothesized participants.

# Load data for participant 1.
data = pickle.load(open("../Preprocessed_data/Data_s2.pkl", "rb"))
data = data[data["participant"] == 1]
data = data.sort_values(by=["participant", "trials"])

# Calculate the maximum and second-maximum distances in the data.
max_d = data["d_phy_p"].max()
sorted_unique_d = np.sort(data["d_phy_p"].unique())
min_d = sorted_unique_d[-2]

# Simulate decay values for generalization rate range.
lambdas = np.arange(0, 0.501, 0.001)
sim_max = np.exp(-lambdas * max_d)
sim_min = np.exp(-lambdas * min_d)

# Set the minimum and maximum percentages for similarity.
min_perc = 0.7
max_perc = 0.95

# Calculate the minimum and maximum generalization rate values that produce the desired
# percentage of similarity.
min_lambda = round(-np.log(min_perc) / max_d, 4)
max_lambda = round(-np.log(max_perc) / min_d, 4)

# Prepare the distance decay data.
d = np.arange(101)
sim_max_d = np.exp(-min_lambda * d)

# Create directory to save the plots.
SIM_dir = f"../Plots/Simulation/Simulation_part_2"
os.makedirs(SIM_dir, exist_ok=True)


# Create a list of plots showing the decay of similarity with distance.
def decay_plot(lambdas, sim_max, min_lambda, min_perc, distances, sim_max_d):
    """
    Generates two subplots visualizing the decay of similarity with respect to
    generalization rate values and distances.

    The first subplot displays how the maximum similarity decays as a function of
    the generalization rate. The second subplot shows the decay of similarity over a
    range of distances.

    Args:
        lambdas (array-like):
            Sequence of generalization rate values.
        sim_max (array-like):
            Maximum similarity values.
        min_lambda (float):
            Generalization rate value where the similarity reaches the minimum threshold.
        min_perc (float):
            Similarity threshold value that is used as a reference in the decay plot.
        distances (array-like):
            Sequence of distances over which the similarity decay is measured.
        sim_max_d (array-like):
            Similarity values corresponding to each distance value.
    """

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # First subplot
    axs[0].plot(
        lambdas, sim_max, label="Decay with max distance", color="black", zorder=0
    )
    axs[0].axvline(
        x=min_lambda, color="gray", linestyle="--", label=f"Min $\\lambda$", zorder=0
    )
    axs[0].axhline(
        y=min_perc, color="blue", linestyle="--", label=f"Min Percentage", zorder=0
    )
    axs[0].text(
        min_lambda + 0.01,
        min_perc + 0.05,
        f"$\\lambda$ = {min_lambda}",
        fontsize=10,
        color="black",
    )
    axs[0].text(min_lambda - 0.06, min_perc, f"{min_perc}", fontsize=10, color="black")
    axs[0].scatter(min_lambda, min_perc, color="red", s=100, zorder=1)
    axs[0].set_xlabel("$\\mathbf{{\\lambda}}$", fontsize=12)
    axs[0].set_ylabel(
        f"Similarity given distance = {max_d}", fontsize=12, weight="bold"
    )
    axs[0].legend(fontsize=12, fancybox=True, shadow=True)
    axs[0].set_facecolor("whitesmoke")

    # Second subplot
    axs[1].plot(distances, sim_max_d, label="Decay across distances", color="black")
    axs[1].text(40, 0.85, f"$\\lambda$ = {min_lambda}", fontsize=10, color="black")
    axs[1].set_xlabel("Distance", fontsize=12, weight="bold")
    axs[1].set_ylabel("Similarity", fontsize=12, weight="bold")
    axs[1].set_ylim(-0.05, 1.05)
    axs[1].set_yticks(np.linspace(0, 1, 5))
    axs[1].legend(fontsize=12, fancybox=True, shadow=True)
    axs[1].set_facecolor("whitesmoke")

    plt.tight_layout()
    plt.savefig(f"{SIM_dir}/Simulation_part_2_Lambda_values.png", dpi=300)


# Plot decay of similarity.
decay_plot(lambdas, sim_max, min_lambda, min_perc, d, sim_max_d)

# Simulation parameters for hypothesised participants in 4 latent groups.

# Set the number of participants in each group.
Ngroup1, Ngroup2, Ngroup3, Ngroup4 = 50, 50, 50, 50

# Calculate the total number of participants.
Nparticipants = Ngroup1 + Ngroup2 + Ngroup3 + Ngroup4

# Get the number of trials from the data.
Ntrials = int(data["trials"].max())

# Set the number of learning trials.
Nactrials = 24

# Set the group names and colors for plots.
group_names = [
    "Non-Learners",
    "Overgeneralizers",
    "Physical Generalizers",
    "Perceptual Generalizers",
]
color_gp = ["#CC79A7", "#F0E442", "#56B4E9", "#D55E00"]


# Function to extract variables from the experimental data.
def extract_variable(data, variable_name):
    """
    Extracts and formats a specific variable from experimental data.

    This function performs the following steps:
      1. Retrieves the data for the given variable name from the input DataFrame
        (assumed to be in long format).
      2. Reshapes the 1D array into a 2D array with one row (shape: (1, Ntrials)).
      3. Repeats the reshaped array for each participant, resulting in an array of shape
        (Nparticipants, Ntrials).

    Args:
        data (pandas.DataFrame):
            The experimental data in long format.
        variable_name (str):
            The key corresponding to the variable to be extracted from the DataFrame.

    Returns:
        repeated_data (numpy.ndarray):
            A 2D array containing the repeated variable data with dimensions
            corresponding to (Nparticipants, Ntrials).
    """
    # Extract the variable.
    variable_data = data[variable_name].values

    # Reshape the data from (Ntrials,) to (1, Ntrials).
    variable_data = variable_data.reshape(1, -1)

    # Repeat the single row for each participant.
    repeated_data = np.tile(variable_data, (Nparticipants, 1))

    return repeated_data


# Create dictionary to hold the variables required for the simulation.
variables = {
    "d_phy_p": extract_variable(data, "d_phy_p"),
    "d_phy_m": extract_variable(data, "d_phy_m"),
    "kplus": extract_variable(data, "CSptrials"),
    "kminus": extract_variable(data, "CSmtrials"),
    "rplus": extract_variable(data, "USp"),
    "rminus": extract_variable(data, "USm"),
}

# Set seed for reproducibility.
np.random.seed(34122)


# Simulation function.
def lg_fun(Nparticipants, Ntrials, variables, persd=30, A=1, K=10):
    """
    Simulates the learning process for multiple participants across several trials.

    This function implements a trial-based learning simulation.
    It models how participants update associative strengths, perceive stimulus distances,
    compute similarities, and generate responses according to the non-linear transformation.
    The process is influenced by participant-specific parameters which are randomly generated
    based on group membership. The simulation ultimately returns a long-format DataFrame that
    combines all relevant output variables for each participant and trial.

    Args:
        Nparticipants (int):
            Number of participants to simulate.
        Ntrials (int):
            Number of total trials for each participant.
        variables (dict):
           A dictionary containing simulation input arrays. Keys include:
           - 'kplus'    : Indicator matrix for updating positive associative strength.
           - 'kminus'   : Indicator matrix for updating negative associative strength.
           - 'rplus'    : Reward signal matrix for positive outcomes.
           - 'rminus'   : Reward signal matrix for negative outcomes.
           - 'd_phy_p'  : Physical distance matrix from CS+.
           - 'd_phy_m'  : Physical distance matrix from CS-.
        persd (float):
            Standard deviation for the noise applied when computing perceived distances.
        A (float):
            Lower bound constant for the sigmoid transformation for generating theta.
        K (float):
            Upper bound constant for the sigmoid transformation for generating theta.

    Returns:
        results_df (pandas.DataFrame):
            Long-format DataFrame containing the simulated learning data with columns for each
            output variable, participant-specific parameters, group membership, and stimulus type.
    """
    participants = np.arange(1, Nparticipants + 1)
    trials = np.arange(1, Ntrials + 1)

    # Initialize arrays.
    alpha, lambda_, w0, w1, sigma = [np.zeros(Nparticipants) for _ in range(5)]
    group = np.repeat(np.arange(0, 4), Nparticipants // 4)
    y = np.zeros((Nparticipants, Ntrials))
    d_per_p = np.zeros((Nparticipants, Ntrials))
    d_per_m = np.zeros((Nparticipants, Ntrials))
    theta = np.zeros((Nparticipants, Ntrials))
    s_plus = np.zeros((Nparticipants, Ntrials))
    s_minus = np.zeros((Nparticipants, Ntrials))
    v_plus = np.zeros((Nparticipants, Ntrials + 1))
    v_minus = np.zeros((Nparticipants, Ntrials + 1))
    g = np.zeros((Nparticipants, Ntrials))

    # Iterate through participants.
    for i in range(Nparticipants):

        # Learning rate.
        alpha[i] = np.where(group[i] == 0, 0, np.random.beta(1, 1))

        # Generalization rate.
        lambda_[i] = np.where(
            group[i] == 0,
            0,
            np.where(
                group[i] == 1,
                np.clip(np.random.normal(0.0026, 0.001), 0, 0.0052),
                np.clip(np.random.normal((0.0052 + 0.3022) / 2, 0.1), 0.0052, 0.3022),
            ),
        )

        # Baseline response.
        w0[i] = np.where(group[i] < 2, np.random.normal(0, 5), np.random.normal(-2, 1))

        # Scaling factor.
        w1[i] = np.random.gamma(10, 1)

        # Response noise.
        sigma[i] = np.where(group[i] == 0, 2.5, 0.5)

        # Iterate through trials.
        for j in range(Ntrials):

            # Learning (associative strengths).
            v_plus[i, j + 1] = np.where(
                group[i] != 0,
                np.where(
                    variables["kplus"][i, j] == 1,
                    v_plus[i, j] + alpha[i] * (variables["rplus"][i, j] - v_plus[i, j]),
                    v_plus[i, j],
                ),
                0,
            )
            v_minus[i, j + 1] = np.where(
                group[i] != 0,
                np.where(
                    variables["kminus"][i, j] == 1,
                    v_minus[i, j]
                    + alpha[i] * (variables["rminus"][i, j] - v_minus[i, j]),
                    v_minus[i, j],
                ),
                0,
            )

            # Perceptual distances.
            d_per_p[i, j] = np.maximum(
                0, np.random.normal(variables["d_phy_p"][i, j], persd)
            )
            d_per_m[i, j] = np.maximum(
                0, np.random.normal(variables["d_phy_m"][i, j], persd)
            )

            # Stimulus similarities.
            s_plus[i, j] = np.where(
                v_plus[i, j] > 0 and group[i] > 0,
                np.where(
                    group[i] == 3,
                    np.exp(-lambda_[i] * d_per_p[i, j]),
                    np.exp(-lambda_[i] * variables["d_phy_p"][i, j]),
                ),
                1,
            )
            s_minus[i, j] = np.where(
                abs(v_minus[i, j]) > 0 and group[i] > 0,
                np.where(
                    group[i] == 3,
                    np.exp(-lambda_[i] * d_per_m[i, j]),
                    np.exp(-lambda_[i] * variables["d_phy_m"][i, j]),
                ),
                1,
            )

            # Generalized associative strength.
            g[i, j] = v_plus[i, j] * s_plus[i, j] + v_minus[i, j] * s_minus[i, j]

            # Non linear sigmoid transformation.
            theta[i, j] = A + (K - A) / (1 + np.exp(-(w0[i] + w1[i] * g[i, j])))

            # Final response.
            y[i, j] = np.random.normal(theta[i, j], sigma[i])

    # Melt and combine DataFrames.
    def melt_data(array, var_name):
        df = pd.DataFrame(array, index=participants, columns=trials)
        return pd.melt(
            df.reset_index(), id_vars="index", var_name="trials", value_name=var_name
        ).rename(columns={"index": "participant"})

    # Create a dictionary of DataFrames to melt.
    to_melt = {
        "y": y,
        "g": g,
        "theta": theta,
        "v_plus": v_plus[:, :-1],
        "v_minus": v_minus[:, :-1],
        "s_plus": s_plus,
        "s_minus": s_minus,
        "d_per_p": d_per_p,
        "d_per_m": d_per_m,
        "d_phy_p": variables["d_phy_p"],
        "d_phy_m": variables["d_phy_m"],
        "rplus": variables["rplus"],
    }

    melted_data = {key: melt_data(value, key) for key, value in to_melt.items()}

    # Merge all melted DataFrames on 'participant' and 'trials'.
    results_df = reduce(
        lambda left, right: pd.merge(left, right, on=["participant", "trials"]),
        melted_data.values(),
    )

    # Add additional columns.
    param_df = pd.DataFrame(
        {
            "participant": participants,  # 1 … Nparticipants
            "group": group,
            "alpha": alpha,
            "lambda": lambda_,
            "w0": w0,
            "w1": w1,
            "sigma": sigma,
        }
    )

    results_df = results_df.merge(param_df, on="participant", how="left")
    results_df["stim"] = np.select(
        [results_df["d_phy_p"] == 0, results_df["d_phy_m"] == 0],
        ["CS+", "CS-"],
        default="TS",
    )

    return results_df


# Generate simulated dataset.
result_df = lg_fun(Nparticipants, Ntrials, variables)

# Update group labels.
result_df["group"] = pd.Categorical(
    result_df["group"], categories=[0, 1, 2, 3], ordered=True
)
result_df["group"] = result_df["group"].cat.rename_categories(group_names)

# Merge the stimulation's results with the experimental data's stimulus column.
result_df = result_df.merge(data[["trials", "stimulus"]], on="trials", how="left")

# Set stimulus column as categorical and order the stimuli.
stim_levels = ["CS+", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "CS-"]
result_df["stimulus"] = pd.Categorical(
    result_df["stimulus"], categories=stim_levels, ordered=True
)


# Visualization.


# Learning.
def lr_plot(result_df):
    """
    Visualizes the progression of associative strengths over learning trials for
    different groups in the simulated dataset.

    This function processes the simulated dataset by first filtering out
    generalization trials. It computes the group-level mean associative strengths for
    both the excitatory and inhibitory components across trials, then generates a series
    of subplots, one per group, to display individual participant trajectories, as well
    as the group means.

    Args:
        result_df (DataFrame):
            A long format DataFrame containing the simulated dataset.
    """
    # Filter out generalization trials.
    filetered_df = result_df[result_df["trials"] < 24].copy()

    # Generate columns for groups' mean associative strengths.
    filetered_df["mean_vp"] = filetered_df.groupby(["trials", "group"], observed=True)[
        "v_plus"
    ].transform("mean")
    filetered_df["mean_vm"] = filetered_df.groupby(["trials", "group"], observed=True)[
        "v_minus"
    ].transform("mean")

    # Define the color palette for individual lines.
    individual_line_color = {"v_plus": "#e41a1c", "v_minus": "#377eb8"}

    # Setup the plot.
    groups = filetered_df["group"].sort_values().unique()
    fig, axes = plt.subplots(
        nrows=len(groups), figsize=(18, 3 * len(groups)), sharex=True
    )

    for ax, group in zip(axes, groups):
        group_data = filetered_df[filetered_df["group"] == group]

        # Plot group associative strengths' individual participant lines and points.
        sns.lineplot(
            data=group_data,
            x="trials",
            y="v_plus",
            hue="participant",
            ax=ax,
            palette=[individual_line_color["v_plus"]]
            * group_data["participant"].nunique(),
            legend=None,
            alpha=0.1,
            linewidth=1,
            zorder=0,
        )
        sns.scatterplot(
            data=group_data,
            x="trials",
            y="v_plus",
            style="rplus",
            markers={0: "o", 1: "^"},
            ax=ax,
            legend=None,
            color=individual_line_color["v_plus"],
            alpha=0.1,
            s=30,
            zorder=1,
        )
        sns.lineplot(
            data=group_data,
            x="trials",
            y="v_minus",
            hue="participant",
            ax=ax,
            palette=[individual_line_color["v_minus"]]
            * group_data["participant"].nunique(),
            legend=None,
            alpha=0.3,
            linewidth=1,
            zorder=0,
        )
        sns.scatterplot(
            data=group_data,
            x="trials",
            y="v_minus",
            style="rplus",
            markers={0: "o", 1: "^"},
            ax=ax,
            legend=None,
            color=individual_line_color["v_minus"],
            alpha=0.3,
            s=30,
            zorder=1,
        )

        # Plot group associative strengths' mean lines and points.
        sns.lineplot(
            data=group_data,
            x="trials",
            y="mean_vp",
            ax=ax,
            color=individual_line_color["v_plus"],
            linewidth=10,
            legend=None,
            zorder=2,
        )
        sns.scatterplot(
            data=group_data,
            x="trials",
            y="mean_vp",
            style="rplus",
            markers={0: "o", 1: "^"},
            ax=ax,
            legend=None,
            color="black",
            s=200,
            zorder=3,
        )
        sns.lineplot(
            data=group_data,
            x="trials",
            y="mean_vm",
            ax=ax,
            color=individual_line_color["v_minus"],
            linewidth=10,
            legend=None,
            zorder=2,
        )
        sns.scatterplot(
            data=group_data,
            x="trials",
            y="mean_vm",
            style="rplus",
            markers={0: "o", 1: "^"},
            ax=ax,
            legend=None,
            color="black",
            s=200,
            zorder=3,
        )
        sns.scatterplot(
            data=group_data,
            x="trials",
            y=-1.4,
            ax=ax,
            hue="stim",
            style="stim",
            palette={"CS+": "red", "CS-": "blue"},
            markers={"CS+": "s", "CS-": "s"},
            s=75,
            legend=None,
            zorder=3,
        )

        # Set subplot parameters.
        ax.set_title(f"{group}", weight="bold", fontsize=14)
        ax.set_xlabel("Trials", weight="bold", fontsize=12)
        ax.set_ylabel("")
        ax.set_ylim(-1.5, 1.2)
        ax.set_yticks(np.arange(-1, 1.1, 0.5))
        ax.set_facecolor("whitesmoke")

    # Create custom overall legend.
    custom_lines = [
        plt.Line2D([], [], color="#e41a1c", linewidth=4, label="CS+ strength"),
        plt.Line2D([], [], color="#377eb8", linewidth=4, label="CS- strength"),
        plt.Line2D(
            [],
            [],
            color="black",
            marker="^",
            linestyle="None",
            markersize=12,
            label="Shock",
        ),
        plt.Line2D(
            [],
            [],
            color="black",
            marker="o",
            linestyle="None",
            markersize=12,
            label="Shock absent",
        ),
        plt.Line2D(
            [],
            [],
            color="red",
            marker="s",
            linestyle="None",
            markersize=8,
            label="CS+ trial",
        ),
        plt.Line2D(
            [],
            [],
            color="blue",
            marker="s",
            linestyle="None",
            markersize=8,
            label="CS- trial",
        ),
    ]
    fig.legend(
        handles=custom_lines,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=6,
        fancybox=True,
        shadow=True,
        fontsize=12,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.97, bottom=0.08)
    fig.text(
        0.0015,
        0.5,
        "Associative Strengths",
        va="center",
        weight="bold",
        rotation="vertical",
        fontsize=12,
    )
    plt.savefig(f"{SIM_dir}/Simulation_part_2_Learning.png", dpi=300)


# Plot learning (associative strengths adjustments over learning trials).
lr_plot(result_df)


# Similarity.
def sim_plot(result_df):
    """
    Visualizes the similarity to CS+ across adjusted physical and perceptual distances
    for the different groups in the simulated dataset.

    This function filters the input DataFrame to retain data for Non-Learners or
    participants with relevant associative strengths. For each group, the function
    creates a subplot that displays both the physical and perceptual similarity to CS+.

    Args:
        result_df (DataFrame):
            A long format DataFrame containing the simulated dataset.
    """
    # Filter the DataFrame to keep Non Learners and participants which have relevant
    # associative strengths.
    filtered_df = result_df[
        (result_df["group"] == "Non-Learners")
        | ((result_df["v_plus"] > 0.1) & (result_df["v_minus"] < 0.1))
    ].copy()

    # Create adjusted physical distances from CS+ columns (for plotting).
    filtered_df["d_phy_p_adj"] = filtered_df["d_phy_p"] + 10

    # Create adjusted perceptual distances from CS+ columns (for plotting).
    filtered_df["d_per_p_adj"] = -filtered_df["d_per_p"] - 10

    # Calculate maximum value of perceptual distance from CS+ and add 10 (for plotting).
    max_d = result_df["d_per_p"].max() + 10

    # Get unique groups.
    groups = filtered_df["group"].sort_values().unique()

    # Generate figure.
    fig, axes = plt.subplots(
        nrows=len(groups), figsize=(18, 3 * len(groups)), sharex=True
    )

    # Iterate through the groups.
    for i, group in enumerate(groups):
        ax = axes[i]
        group_data = filtered_df[filtered_df["group"] == group]

        # Create scatter and line plots for physical similarity to CS+.
        ax.plot(
            group_data["d_phy_p_adj"],
            group_data["s_plus"],
            marker="",
            linestyle="-",
            color="grey",
            alpha=0.25,
        )
        sns.scatterplot(
            data=group_data,
            x="d_phy_p_adj",
            y="s_plus",
            hue="stim",
            palette={"CS+": "#e41a1c", "CS-": "#377eb8", "TS": "grey"},
            ax=ax,
            alpha=0.5,
            legend=False,
        )

        # Create scatter and line plots for perceptual similarity to CS+.
        ax.plot(
            group_data["d_per_p_adj"],
            group_data["s_plus"],
            marker="",
            linestyle="-",
            color="grey",
            alpha=0.25,
        )
        sns.scatterplot(
            data=group_data,
            x="d_per_p_adj",
            y="s_plus",
            hue="stim",
            palette={"CS+": "#e41a1c", "CS-": "#377eb8", "TS": "grey"},
            ax=ax,
            alpha=0.5,
        )

        # Set subplot parameters.
        ax.set_title(f"{group}", weight="bold", fontsize=14)
        ax.set_xlim(-max_d, max_d)
        ax.set_yticks(np.arange(0, 1.1, 0.5))
        ax.set_ylim(-0.1, 1.1)
        ax.axvline(0, color="black", linestyle="dashed", linewidth=1)
        ax.set_xlabel(
            "Distance to CS+ (left: perceptual; right: physical)",
            weight="bold",
            fontsize=12,
        )
        ax.set_ylabel("")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{abs(x)}"))
        ax.get_legend().remove()
        ax.set_facecolor("whitesmoke")

    # Create overall legend.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=3,
        fancybox=True,
        shadow=True,
        fontsize=12,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.97, bottom=0.08)
    fig.text(
        0.0015,
        0.5,
        "Similarity to CS+",
        va="center",
        weight="bold",
        rotation="vertical",
        fontsize=12,
    )
    plt.savefig(f"{SIM_dir}/Simulation_part_2_Similarity.png", dpi=300)


# Plot similarity to CS+.
sim_plot(result_df)


# Generalization.


# Overall generalized response.
def ge_plot(result_df):
    """
    Visualizes overall generalized responses for the simulated dataset, focusing on
    generalization trials.

    The function filters out learning trials, computes and plots both overall and
    individual mean responses for each stimulus based on the simulated response ('y').

    Args:
        result_df (DataFrame):
            A long format DataFrame containing the simulated dataset.
    """
    # Filter out learning trials.
    filtered_df = result_df[result_df["trials"] > 24].copy()

    # Calculate overall mean and individual mean responses (simulated US expectancies)
    # grouped by stimulus.
    filtered_df["mean_res"] = filtered_df.groupby("stimulus", observed=True)[
        "y"
    ].transform("mean")
    filtered_df["indi_res"] = filtered_df.groupby(
        ["participant", "stimulus"], observed=True
    )["y"].transform("mean")

    # Create figure.
    plt.figure(figsize=(10, 6))

    # Get figure ax.
    ax = plt.gca()

    # Plot point and lines for overall mean responses.
    sns.lineplot(
        data=filtered_df,
        x="stimulus",
        y="mean_res",
        color="black",
        linewidth=3,
        zorder=2,
    )
    sns.scatterplot(
        data=filtered_df, x="stimulus", y="mean_res", color="black", s=70, zorder=3
    )

    # Plot point and lines for individual mean responses.
    for participant in filtered_df["participant"].unique():
        participant_data = filtered_df[filtered_df["participant"] == participant]
        sns.lineplot(
            data=participant_data,
            x="stimulus",
            y="indi_res",
            color="grey",
            estimator=None,
            alpha=0.2,
            linewidth=1,
            legend=None,
            zorder=0,
        )
        sns.scatterplot(
            data=participant_data,
            x="stimulus",
            y="indi_res",
            color="grey",
            alpha=0.2,
            s=5,
            zorder=1,
        )

    # Set plot parameters.
    ax.set_title("Generalized Response (Whole Dataset)", weight="bold", fontsize=14)
    ax.set_xlabel("Stimulus", weight="bold", fontsize=12)
    ax.set_ylabel("US Expectancy", weight="bold", fontsize=12)
    ax.set_yticks([1, 5, 10])
    ax.set_ylim(-1, 12)
    ax.set_facecolor("whitesmoke")

    # Create custom legend.
    custom_lines = [
        plt.Line2D(
            [],
            [],
            color="black",
            marker="o",
            linewidth=3,
            label="Overall mean response",
        ),
        plt.Line2D(
            [],
            [],
            color="grey",
            marker="o",
            linewidth=3,
            alpha=0.5,
            label="Individual mean response",
        ),
    ]
    ax.legend(
        handles=custom_lines,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=2,
        fancybox=True,
        shadow=True,
        fontsize=12,
    )

    plt.savefig(f"{SIM_dir}/Simulation_part_2_Generalized_Response_Whole_Dataset")


# Plot overall generalization.
ge_plot(result_df)


# Generalized response for each group.
def ge_gr_plot(result_df):
    """
    Visualizes generalized responses for each experimental group across stimuli in the
    simulated dataset.

    The function filters out learning trials, computes both group and group's individuals
    mean responses for each stimulus based on the simulated response, and generates a
    subplot for each group to visualize these mean responses.

    Args:
        result_df (DataFrame):
            A long format DataFrame containing the simulated dataset.
    """
    # Filter out learning trials.
    filtered_df = result_df[result_df["trials"] > 24].copy()

    # Calculate groups mean and individual mean responses (simulated US expectancies)
    # grouped by stimulus.
    filtered_df["mean_res"] = filtered_df.groupby(["group", "stimulus"], observed=True)[
        "y"
    ].transform("mean")
    filtered_df["indi_res"] = filtered_df.groupby(
        ["participant", "stimulus"], observed=True
    )["y"].transform("mean")

    # Get unique groups.
    groups = filtered_df["group"].sort_values().unique()

    # Create the figure.
    fig, axes = plt.subplots(2, 2, figsize=(20, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    # Iterate through groups and axes.
    for ax, group in zip(axes, groups):

        # Retrieve current group data.
        group_data = filtered_df[filtered_df["group"] == group]

        # Plot points and lines for current group mean responses.
        sns.lineplot(
            data=group_data,
            x="stimulus",
            y="mean_res",
            ax=ax,
            color="black",
            linewidth=3,
            zorder=2,
        )
        sns.scatterplot(
            data=group_data,
            x="stimulus",
            y="mean_res",
            ax=ax,
            color="black",
            s=75,
            zorder=3,
        )

        # Plot points and lines for current group individual mean responses.
        sns.lineplot(
            data=group_data,
            x="stimulus",
            y="indi_res",
            hue="participant",
            ax=ax,
            palette="grey",
            legend=None,
            linewidth=1,
            alpha=0.2,
            zorder=0,
        )
        sns.scatterplot(
            data=group_data,
            x="stimulus",
            y="indi_res",
            hue="participant",
            ax=ax,
            palette="grey",
            s=5,
            legend=None,
            zorder=1,
        )

        # Set subplot parameters.
        ax.set_title(f"{group}", weight="bold", fontsize=14)
        ax.set_xlabel("Stimulus", weight="bold", fontsize=12)
        ax.set_ylabel("US Expectancy", weight="bold", fontsize=12)
        ax.set_yticks([1, 5, 10])
        ax.set_ylim(-1, 12)
        ax.set_facecolor("whitesmoke")

    # Create overall legend.
    custom_lines = [
        plt.Line2D(
            [], [], color="black", marker="o", linewidth=3, label="Group mean response"
        ),
        plt.Line2D(
            [],
            [],
            color="grey",
            marker="o",
            linewidth=3,
            alpha=0.5,
            label="Group member mean response",
        ),
    ]
    fig.legend(
        handles=custom_lines,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=2,
        fancybox=True,
        shadow=True,
        fontsize=12,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.savefig(f"{SIM_dir}/Simulation_part_2_Generalized_Response.png", dpi=300)


# Plot group generalization.
ge_gr_plot(result_df)


def prepare_data_for_pymc(
    L,
    sim_dataset=result_df,
    variables=variables,
    participants=Nparticipants,
    lr_trials=Nactrials,
    trials=Ntrials,
):
    """
    Prepare input data for PyMC analysis in the context of differential conditioning.

     Args:
        L (int):
            Specifies the type of data to handle:
            - 1: Use only generalization trials.
            - 2: Use the full dataset, including learning trials.
        sim_dataset (DataFrame):
            The DataFrame containing the simulated dataset.
        variables (dict):
            A dictionary containing additional data arrays like:
            - Reinforcement data 'rplus', 'rminus'.
            - Indicator data 'kplus', 'kminus'.
        participants (int):
            Number of participants.
        lr_trials (int):
            Number of learning trials.
        trials (int):
            Total number of trials.

    Returns:
        data (dict):
            A dictionary with the following keys:
            - 'Nparticipants' (int): Number of participants.
            - 'Ntrials' (int): Number of trials.
            - 'Nactrials' (int): Number of active trials (set to 24).
            - 'd_p_per' (ndarray): Perceptual distances for CS+ for each
              participant/trial.
            - 'd_p_phy' (ndarray): Physical distances for CS+ for each
              participant/trial.
            - 'd_m_per' (ndarray): Perceptual distances for CS- for each
              participant/trial.
            - 'd_m_phy' (ndarray): Physical distances for CS- for each
              participant/trial.
            - 'y' (ndarray): US expectancy data.
            - 'r_plus' (ndarray, optional): Reinforcement data for CS+ (if `L == 2`).
            - 'r_minus' (ndarray, optional): Reinforcement data for CS- (if `L == 2`).
            - 'k_plus' (ndarray, optional): CS+ indicator data (if `L == 2`).
            - 'k_minus' (ndarray, optional): CS- indicator data (if `L == 2`).
    """
    # Reshape data to ensure it's in participant x trials format.
    y_matrix = sim_dataset["y"].values.reshape(participants, trials, order="F")
    d_p_per_matrix = sim_dataset["d_per_p"].values.reshape(
        participants, trials, order="F"
    )
    d_m_per_matrix = sim_dataset["d_per_m"].values.reshape(
        participants, trials, order="F"
    )
    d_p_phy_matrix = sim_dataset["d_phy_p"].values.reshape(
        participants, trials, order="F"
    )
    d_m_phy_matrix = sim_dataset["d_phy_m"].values.reshape(
        participants, trials, order="F"
    )

    # Select data based on L.
    if L == 1:
        selected_trials = slice(Nactrials, Ntrials + 1)
    elif L == 2:
        selected_trials = slice(None)

    # Construct final dictionary.
    data = {
        "Nparticipants": participants,
        "Ntrials": trials,
        "Nactrials": lr_trials,
        "y": y_matrix[:, selected_trials],
        "d_p_per": d_p_per_matrix[:, selected_trials],
        "d_m_per": d_m_per_matrix[:, selected_trials],
        "d_p_phy": d_p_phy_matrix[:, selected_trials],
        "d_m_phy": d_m_phy_matrix[:, selected_trials],
        "r_plus": np.array(variables["rplus"]),
        "r_minus": np.array(variables["rminus"]),
        "k_plus": np.array(variables["kplus"]),
        "k_minus": np.array(variables["kminus"]),
    }

    return data


# Create directory to store input data for PyMC models.
os.makedirs("../PYMC_input_data", exist_ok=True)

# PyMC input data with an assumption of continuous learning.
pickle.dump(
    prepare_data_for_pymc(2), open("../PYMC_input_data/Sim_PYMCinput_CLG.pkl", "wb")
)
# PyMC input data without learning trials.
pickle.dump(prepare_data_for_pymc(1), open("../PYMC_input_data/Sim_PYMCinput_G.pkl", "wb"))


if __name__ == "__main__":

    '''# Sample and save fitting results for the simulation.

    # Sample with full model: CLG2D.
    Result_Sim_CLG2D = sampling_fun(
        pickle.load(open("PYMC_input_data/Sim_PYMCinput_CLG.pkl", "rb")),
        "models_definitions",
        "model_2v_LG2D",
    )
    # Save results.
    az.to_netcdf(Result_Sim_CLG2D, "Fitting_results/model_2v_LG2D/Result_Sim_CLG2D.nc")
    # Sample with simplified model: LGPHY.
    Result_Sim_CLGPHY = sampling_fun(
        pickle.load(open("PYMC_input_data/Sim_PYMCinput_CLG.pkl", "rb")),
        "models_definitions",
        "model_2v_LGPHY",
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
    )
    # Save results.
    az.to_netcdf(Result_Sim_G2D, "Fitting_results/model_2v_G2D/Result_Sim_G2D.nc")'''

# Load simulation fitting results.
results = {
    "lg2d": ReshapedInferenceData(
        az.from_netcdf("../Fitting_results/model_2v_LG2D/Result_Sim_CLG2D.nc")
    ),
    "lgphy": ReshapedInferenceData(
        az.from_netcdf("../Fitting_results/model_2v_LGPHY/Result_Sim_CLGPHY.nc")
    ),
    "g2d": ReshapedInferenceData(
        az.from_netcdf("../Fitting_results/model_2v_G2D/Result_Sim_G2D.nc")
    ),
}

# Define plotting parameters.
group_names = [
    "Non-Learners",
    "Overgeneralizers",
    "Physical Generalizers",
    "Perceptual Generalizers",
]
color_groups = ["#CC79A7", "#F0E442", "#56B4E9", "#D55E00"]
palette = dict(zip(group_names, color_groups))

# Plot group allocation for the full model LG2D with the simulated dataset.
plot_gp_allocation(
    gp_allocation(results["lg2d"].posterior["gp"]),
    "Simulation part 2 LG2D",
    group_names,
    color_groups,
    SIM_dir,
)

# Create lambda generalization rate long-format dataframe for plotting.
lambda_df = pd.DataFrame(
    {
        # Number of participants in the simulated dataset.
        "participant": np.arange(1, 201),
        # Posterior participants median (50th percentile) of alpha from the LG2D model.
        "lg2d": np.nanpercentile(results["lg2d"].posterior["lambda"], 50, axis=0),
        # Posterior participants median (50th percentile) of alpha from the LGPHY model.
        "lgphy": np.nanpercentile(results["lgphy"].posterior["lambda"], 50, axis=0),
        # Posterior participants median (50th percentile) of alpha from the G2D model.
        "g2d": np.nanpercentile(results["g2d"].posterior["lambda"], 50, axis=0),
    }
)

# Merge the lambda dataframe with the simulated data to create ground truth column
# and group column.
lambda_df = lambda_df.merge(
    result_df[["participant", "lambda", "group"]], on="participant", how="left"
).rename(columns={"lambda": "true"})

# Create alpha learning rate long-format dataframe for plotting.
alpha_df = pd.DataFrame(
    {
        # Number of participants in the simulated dataset.
        "participant": np.arange(1, 201),
        # Posterior participants median (50th percentile) of alpha from the LG2D model.
        "lg2d": np.nanpercentile(results["lg2d"].posterior["alpha"], 50, axis=0),
        # Posterior participants median (50th percentile) of alpha from the LGPHY model.
        "lgphy": np.nanpercentile(results["lgphy"].posterior["alpha"], 50, axis=0),
    }
)

# Merge the alpha dataframe with the simulated data to create ground truth column
# and group column.
alpha_df = alpha_df.merge(
    result_df[["participant", "alpha", "group"]], on="participant", how="left"
).rename(columns={"alpha": "true"})

# Recovery.


# Lambda recovery subplot generator.
def recovery_plot(df, yvar, subtitle, palette):
    """
    Visualizes parameter recovery for a specific model panel.

    This function creates a scatter plot comparing simulated ("true") versus inferred
    values of the generalization rate (lambda) for a given model, color-coded by group.

    Args:
        df (DataFrame):
            DataFrame containing the columns for recovery analysis.
        yvar (str):
            Name of the column in df holding the inferred lambda values to plot.
        subtitle (str):
            Subtitle for the subplot, displayed as the panel title.
        palette (dict):
            Mapping from group names to colors.

    Returns:
        ax (matplotlib.axes):
            The axes object with the recovery plot.
    """
    # Plot lambda median points.
    ax = sns.scatterplot(
        data=df,
        x="true",
        y=yvar,
        hue="group",
        palette=palette,
        s=40,
        edgecolor="w",
        alpha=0.9,
        legend=False,
    )

    # Plot lambda ground truth line.
    sns.lineplot(data=df, x="true", y="true", ax=ax, linewidth=1, color="black")

    # Set subplot parameters.
    ax.set_xlabel(f"Simulated $\\mathbf{{\\lambda}}$", weight="bold", fontsize=10)
    ax.set_ylabel(f"Inferred $\\mathbf{{\\lambda}}$", weight="bold", fontsize=10)
    ax.set_title(subtitle, weight="bold", fontsize=12)
    ax.set_facecolor("whitesmoke")
    return ax


# All models lambda recovery.
def all_recovery_plot(lambda_df, palette):
    """
    Generates a figure displaying lambda recovery across different model specifications.

    This function arranges a grid and invokes `recovery_plot` to create three panels:
    - Data generating model (LG2D)
    - Simplified model (LGPHY)
    - Simplified model (G2D) with a zoomed inset

    Args:
        lambda_df (DataFrame):
            DataFrame containing the columns for recovery analysis.
        palette (dict):
            Mapping from group names to colors.
    """
    # Create figure.
    fig = plt.figure(figsize=(12, 8))

    # Generate subplots grid.
    outer = GridSpec(
        2, 3, width_ratios=[0.2, 1, 1], height_ratios=[1, 1], wspace=0.3, hspace=0.4
    )

    # Spacer.
    ax = fig.add_subplot(outer[0, 0])
    ax.axis("off")

    # Generate LG2D subplot.
    axA = fig.add_subplot(outer[0, 1])
    recovery_plot(lambda_df, "lg2d", "Data generating model LG2D", palette)

    # Spacer.
    ax = fig.add_subplot(outer[0, 2])
    ax.axis("off")
    # Spacer.
    ax = fig.add_subplot(outer[1, 0])
    ax.axis("off")

    # Generate LGPHY subplot.
    axB = fig.add_subplot(outer[1, 1])
    recovery_plot(lambda_df, "lgphy", "Simplified model LGPHY", palette)

    # Generate subgrid for g2d plots.
    sub = outer[1, 2].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.6)

    # Generate G2D subplot.
    axC = fig.add_subplot(sub[0, 0])
    recovery_plot(lambda_df, "g2d", "Simplified model G2D", palette)

    # Generate zoomed in G2D subplot.
    axZ = fig.add_subplot(sub[1, 0])
    recovery_plot(lambda_df, "g2d", "", palette)
    axZ.set_xlabel("")
    axZ.set_ylabel("")
    axZ.set_ylim(0.0024, 0.0027)

    # Generate overall legend.
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor=col,
            markeredgecolor="w",
            markersize=8,
        )
        for col in palette.values()
    ]
    labels = list(palette.keys())
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.subplots_adjust(bottom=0.15)

    # Save figure.
    fig.savefig(
        f"{SIM_dir}/Simulation_part_2_lambda_recovery.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


# Lambda vs alpha LG2D recovery.
def vs_recovery_plot(lambda_df, alpha_df, palette):
    """
    Visualizes side-by-side recovery of lambda and alpha for the LG2D model.

    This function produces a two-panel figure:
      - Left panel: simulated vs inferred generalization rate (lambda)
      - Right panel: simulated vs inferred learning rate (alpha)

    Args:
        lambda_df (DataFrame):
            DataFrame containing the columns for lambda recovery.
        alpha_df (DataFrame):
            DataFrame containing the columns for alpha recovery.
        palette (dict):
            Mapping from group names to colors.
    """
    # Generate figure.
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=False)

    # Plot lambda recovery.
    ax = axes[0]
    # Plot lambda ground truth line.
    sns.lineplot(x="true", y="true", data=lambda_df, ax=ax, linewidth=1, color="black")
    # Plot lambda median points.
    sns.scatterplot(
        x="true",
        y="lg2d",
        hue="group",
        palette=palette,
        data=lambda_df,
        ax=ax,
        s=40,
        edgecolor="w",
        alpha=0.9,
    )

    # Set subplot parameters.
    ax.set_xlabel(f"Simulated $\\mathbf{{\\lambda}}$", weight="bold", fontsize=10)
    ax.set_ylabel(f"Inferred $\\mathbf{{\\lambda}}$", weight="bold", fontsize=10)
    ax.set_title("LG2D model", weight="bold", fontsize=12)
    ax.set_facecolor("whitesmoke")
    ax.get_legend().remove()

    # Plot alpha recovery.
    ax = axes[1]
    # Plot alpha ground truth line.
    sns.lineplot(x="true", y="true", data=alpha_df, ax=ax, linewidth=1, color="black")
    # Plot alpha median point.
    sns.scatterplot(
        x="true",
        y="lg2d",
        hue="group",
        palette=palette,
        data=alpha_df,
        ax=ax,
        s=40,
        edgecolor="w",
        alpha=0.9,
    )

    # Set subplot parameters.
    ax.set_xlabel(f"Simulated $\\mathbf{{\\alpha}}$", weight="bold", fontsize=10)
    ax.set_ylabel(f"Inferred $\\mathbf{{\\alpha}}$", weight="bold", fontsize=10)
    ax.set_title("LG2D model", weight="bold", fontsize=12)
    ax.set_facecolor("whitesmoke")
    ax.get_legend().remove()

    # Generate overall legend.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(palette), frameon=False)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(
        f"{SIM_dir}/Simulation_part_2_lambda_vs_alpha_recovery_LG2D.png", dpi=300
    )
    plt.close()


# Plot models recovery of lambda generalization rate.
fig1 = all_recovery_plot(lambda_df, palette)

# Plot lambda vs alpha recovery for LG2D model.
fig2 = vs_recovery_plot(lambda_df, alpha_df, palette)
