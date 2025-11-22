"""
Module: response_patterns.py

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
This Python module is an independent implementation that is largely a
conversion of the original R code to Python, designed to process and analyze
response patterns from Bayesian modeling results. It facilitates the
visualization of response dynamics, generalization, learning and
perception patterns, similarity simulations and perceptual assessments across
participants grouped by their response characteristics in the psychological studies.

The key functionalities of this module include:
- Integrating experimental data with group allocations and calculating statistics
  for US expectancy, perceived stimulus size, and simulus distance from CS+.
- Visualizing generalization overall and grouped data US expectancy across stimuli.
- Visualizing groups observed US expectancies for CS+ and CS- across trials.
- Computing and visualizing groups learning processes across trials in respect to
  CS+ and CS- by integrating the alpha learning rate parameter from the studies
  into the experimental data.
- Analyzing and visualizing lambda generalization rates and perceived distances
  from CS+ across all stimuli.
- Computing, analyzing and visualzing groups similarity metrics to CS+ across trials
  based on perceptual and physical distances.

### Functions:
- rp_process_data: Integrates experimental data with group allocations and calculates
  statistics for key variables.
- ge_plot: Visualizes generalization trials data, showing individual and overall observed
  mean US expectancy across stimuli.
- ge_gr_plot: Plots generalization trials data, showing individual and overall observed
  mean US expectancy across stimuli for each dominant group.
- lr_gr_plot: Visualizes observed US expectancy across trials for CS+ and CS-, showing both
  group averages and individual trajectories.
- v_gr_plot: Visualizes associative strength adjustments over trials for both CS+ and CS-,
  showing both group averages and individual trajectories.
- lambda_plot: Displays histograms of the lambda generalization rate from the posterior
  distributions.
- per_gr_mean: Helper function to plot the mean perceived stimulus distance from CS+ for
  a specific group within an experiment.
- per_gr_sd: Helper function to plot the standard deviation of perceived stimulus distance
  from CS+ for a specific group within an experiment.
- per_gr_plot: Arranges and displays perception data for groups within an experiment, showing
  both mean and standard deviation of perceived stimulus distance from CS+ across all stimuli.
- sim_gr_plot: Calculates and plots similarity to CS+ metrics based on generalization rate
  influenced by perceptual or physical distances to CS+ stimuli, differentiated by dominant
  participant groups.
"""

import arviz as az
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.ticker as ticker
from group_allocation import gp_allocation
from data_utils import ReshapedInferenceData

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
results = {
    "s1": {"CLG2D": ReshapedInferenceData(Result_Study1_CLG2D)},
    "s2": {"CLG2D": ReshapedInferenceData(Result_Study2_CLG2D)},
}

# Load long format experiments data.
data_ = {
    "s1": pickle.load(open("../Preprocessed_data/Data_s1.pkl", "rb")),
    "s2": pickle.load(open("../Preprocessed_data/Data_s2.pkl", "rb")),
}

# Define plotting parameters.
experiments = [
    "Experiment 1 Simple Conditioning",
    "Experiment 2 Differential Conditioning",
]
stimulus_levels = {
    "s1": ["S4", "S5", "S6", "CS+", "S8", "S9", "S10"],
    "s2": ["CS+", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "CS-"],
}

# Extract group indicators samples of both studies from the posterior distributions.
gp_samples = [
    results["s1"]["CLG2D"].posterior["gp"],
    results["s2"]["CLG2D"].posterior["gp"],
]

# Create a directory to save response patterns plots.
RP_dir = "../Plots/Response_Patterns/"
os.makedirs(RP_dir, exist_ok=True)


# Process experiments data.
def rp_process_data(data_, gps_processed, stimulus_levels):
    """
    Integrates experimental data with group allocations and calculates
    statistics: means and standard deviations for US expectancy,
    perceived stimulus size and perceived stimulus distance from CS+ while
    grouping data in various ways.

    Args:
        data_ (dict):
            Contains DataFrames for both experiments.
        gps_processed (list of DataFrames):
            List of processed group information data per participant for both studies.
        stimulus_levels (dict):
            Specifies the order of stimuli for each study.

    Returns:
        processed_data, data_ge (tuple):
            A tuple containing two lists, one for processed data and one for
            generalization trials specific data across both experiments.
    """
    # Initialize lists shapes.
    processed_data = [None] * 2
    data_ge = [None] * 2

    # Iterate through the number of total experiments.
    for i, key in enumerate(sorted(data_.keys())):

        processed_data[i] = data_[f"{key}"].copy()
        processed_data[i] = processed_data[i].rename(
            columns={"participant": "Participant_Num"}
        )

        # Merge the data and the group allocation dataframes.
        processed_data[i] = processed_data[i].merge(
            gps_processed[i][["Participant_Num", "Group_Name"]],
            on="Participant_Num",
            how="left",
        )

        # Convert the Group Name and stimulus columns in categorical columns.
        group_order = [
            "Non-Learners",
            "Overgeneralizers",
            "Physical Generalizers",
            "Perceptual Generalizers",
            "Unknown",
        ]

        processed_data[i]["Group_Name"] = pd.Categorical(
            processed_data[i]["Group_Name"], categories=group_order, ordered=True
        )

        processed_data[i]["stimulus"] = pd.Categorical(
            processed_data[i]["stimulus"],
            categories=stimulus_levels[f"{key}"],
            ordered=True,
        )

        # Compute statistics and create columns.

        # Column for mean of individual perceived stimulus size.
        processed_data[i]["mean_per_indi"] = (
            processed_data[i]
            .groupby(["stimulus", "Participant_Num"], observed=True)["Per_size"]
            .transform("mean")
        )
        # Column for sd of individual perceived stimulus size.
        processed_data[i]["sd_per_indi"] = (
            processed_data[i]
            .groupby(["stimulus", "Participant_Num"], observed=True)["Per_size"]
            .transform("std")
        )
        # Column for mean of group stimulus US expectancy.
        processed_data[i]["mean_us_category"] = (
            processed_data[i]
            .groupby(["stimulus", "Group_Name"], observed=True)["US_expect"]
            .transform("mean")
        )
        # Column for mean of group perceived stimulus size.
        processed_data[i]["mean_per_group"] = (
            processed_data[i]
            .groupby(["stimulus_phy", "Group_Name"], observed=True)["Per_size"]
            .transform("mean")
        )
        # Column for sd of group perceived stimulus size.
        processed_data[i]["sd_per_group"] = (
            processed_data[i]
            .groupby(["stimulus_phy", "Group_Name"], observed=True)["Per_size"]
            .transform("std")
        )
        # Column for mean of group stimulus US expectancy.
        processed_data[i]["mean_us_group"] = (
            processed_data[i]
            .groupby(["stimulus"], observed=True)["US_expect"]
            .transform("mean")
        )
        # Column for mean of by trial group stimulus US expectancy.
        processed_data[i]["mean_us_trial"] = (
            processed_data[i]
            .groupby(["trials", "Group_Name", "stimulus"], observed=True)["US_expect"]
            .transform("mean")
        )
        # Column for mean of trial stimulus US expectancy.
        processed_data[i]["mean_us_trial_all"] = (
            processed_data[i]
            .groupby(["trials", "stimulus"], observed=True)["US_expect"]
            .transform("mean")
        )

        # Initialize the dataframe for generalization trials specific data.
        trial_ranges = [list(range(15, 189)), list(range(25, 181))]
        data_ge[i] = processed_data[i][
            processed_data[i]["trials"].isin(trial_ranges[i])
        ].copy()

        # Compute statistics and create columns for generalization trials dataframe.

        # Column for mean of group stimulus US expectancy.
        data_ge[i]["mean_geus_category"] = (
            data_ge[i]
            .groupby(["stimulus", "Group_Name"], observed=True)["US_expect"]
            .transform("mean")
        )
        # Column for mean of individual stimulus US expectancy.
        data_ge[i]["mean_geus"] = (
            data_ge[i]
            .groupby(["stimulus", "Participant_Num"], observed=True)["US_expect"]
            .transform("mean")
        )
        # Column for mean of individual perceived stimulus size.
        data_ge[i]["mean_per"] = (
            data_ge[i]
            .groupby(["stimulus", "Participant_Num"], observed=True)["Per_size"]
            .transform("mean")
        )
        # Column for mean of overall stimulus US expectancy.
        data_ge[i]["mean_geus_all"] = (
            data_ge[i]
            .groupby(["stimulus"], observed=True)["US_expect"]
            .transform("mean")
        )
        # Column for sd of overall stimulus US expectancy.
        data_ge[i]["sd_geus_all"] = (
            data_ge[i]
            .groupby(["stimulus"], observed=True)["US_expect"]
            .transform("std")
        )

    for j, df in enumerate(processed_data):
        if j == 0:
            # Compute statistics and create columns for experiment 1 dataframe.

            # Column for mean of group perceived stimulus distance from CS+.
            df["mean_disp_group"] = df.groupby(
                ["stimulus", "Group_Name"], observed=True
            )["dper"].transform("mean")
            # Column for sd of group perceived stimulus distance from CS+.
            df["sd_disp_group"] = df.groupby(["stimulus", "Group_Name"], observed=True)[
                "dper"
            ].transform("std")
            # Column for mean of individual perceived stimulus distance from CS+.
            df["mean_disp_indi"] = df.groupby(
                ["stimulus", "Participant_Num"], observed=True
            )["dper"].transform("mean")
            # Column for sd of individual perceived stimulus distance from CS+.
            df["sd_disp_indi"] = df.groupby(
                ["stimulus", "Participant_Num"], observed=True
            )["dper"].transform("std")
        else:
            # Compute statistics and create columns for experiment 2 dataframe.

            # Column for mean of group perceived stimulus distance from CS+.
            df["mean_disp_group"] = df.groupby(
                ["stimulus", "Group_Name"], observed=True
            )["d_per_p"].transform("mean")
            # Column for sd of group perceived stimulus distance from CS+.
            df["sd_disp_group"] = df.groupby(["stimulus", "Group_Name"], observed=True)[
                "d_per_p"
            ].transform("std")
            # Column for mean of individual perceived stimulus distance from CS+.
            df["mean_disp_indi"] = df.groupby(
                ["stimulus", "Participant_Num"], observed=True
            )["d_per_p"].transform("mean")
            # Column for sd of individual perceived stimulus distance from CS+.
            df["sd_disp_indi"] = df.groupby(
                ["stimulus", "Participant_Num"], observed=True
            )["d_per_p"].transform("std")

    return processed_data, data_ge


# Load and process data for the two experiments.
processed_data, data_ge = rp_process_data(
    data_,
    [gp_allocation(gp_samples[0]), gp_allocation(gp_samples[1])],
    stimulus_levels,
)


# Generalization.


# All data - generalization pattern.
def ge_plot(data_ge, experiments, output_dir):
    """
    Plots generalization trials data across all stimuli, showing individual and overall
    observed mean US expectancy.

    Args:
        data_ge (list):
            List of DataFrames with generalization trial data for both experiments.
        experiments (list):
            List of the experiments names for plotting.
        output_dir (str):
            Directory where the plots will be saved.
    """
    # Create the figure.
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Iterate through the experiments list of DataFrames.
    for idx, df in enumerate(data_ge):

        # Create lineplot for the overall US expectancy mean across all stimuli.
        sns.lineplot(
            data=df,
            x="stimulus",
            y="mean_geus_all",
            ax=axs[idx],
            marker="o",
            color="black",
        )

        # Create lineplots for US expectancy mean across stimuli for each participant.
        for participant in df["Participant_Num"].unique():
            sns.lineplot(
                data=df[df["Participant_Num"] == participant],
                x="stimulus",
                y="mean_geus",
                ax=axs[idx],
                color="gray",
                alpha=0.1,
            )

        # Set subplot parameters.
        axs[idx].set_title(experiments[idx], weight="bold", fontsize=12)
        axs[idx].set_ylim([0.5, 10.5])
        axs[idx].set_ylabel(
            "US Expectancy (observed data scale: 1-10)", weight="bold", fontsize=10
        )
        axs[idx].set_xlabel("Stimulus", weight="bold", fontsize=10)
        axs[idx].set_yticks([1, 5, 10])
        axs[idx].set_facecolor("whitesmoke")

    # Create overall legend.
    overall_line = plt.Line2D(
        [0], [0], color="black", marker="o", label="Observed data overall mean"
    )
    individual_line = plt.Line2D(
        [0], [0], color="gray", label="Observed data individuals means"
    )
    fig.legend(
        handles=[overall_line, individual_line],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        fancybox=True,
        shadow=True,
        ncol=2,
        fontsize=10,
    )

    fig.subplots_adjust(bottom=0.2)
    fig.savefig(
        f"{output_dir}/Response_Patterns_Overall_Generalization_Pattern.png", dpi=300
    )


# Plot both experiments overall generalization pattern.
ge_plot(data_ge, experiments, RP_dir)


# Groups - generalization pattern.
def ge_gr_plot(data_ge, experiments, output_dir):
    """
    Plots generalization trials data grouped by participans dominant groups, including
    mean and individual US expectancies for each stimulus.

    Args:
        data_ge (list):
            List of DataFrames with generalization trial data for both experiments.
        experiments (list):
            List of the experiments names for plotting.
        output_dir (str):
            Directory where the plots will be saved.
    """
    # Create the figure.
    fig, axs = plt.subplots(2, 4, figsize=(26, 12), sharey=True)

    # Iterate through experiments generalization trials DataFrames.
    for row, df in enumerate(data_ge):

        # Get the unique dominant groups.
        df = df.loc[df["Group_Name"] != "Unknown"]
        groups = df["Group_Name"].sort_values().unique()

        # Iterate through the groups.
        for col, group in enumerate(groups):

            # Retrieve current group specific data in the experiment DataFrame.
            group_df = df[df["Group_Name"] == group]

            # Create lineplot for mean of current group stimulus US expectancy.
            sns.lineplot(
                data=group_df,
                x="stimulus",
                y="mean_geus_category",
                ax=axs[row, col],
                marker="o",
                color="black",
            )

            # Create lineplot for mean of stimulus US expectancy for each participant
            # allocated to the current group.
            for participant in group_df["Participant_Num"].unique():
                sns.lineplot(
                    data=group_df[group_df["Participant_Num"] == participant],
                    x="stimulus",
                    y="mean_geus",
                    ax=axs[row, col],
                    color="gray",
                    alpha=0.1,
                )

            # Set subplot parameters.
            axs[row, col].set_title(f"{group}", weight="bold", fontsize=12)
            axs[row, col].set_ylim(0.5, 10.5)
            axs[row, col].set_yticks([1, 5, 10])
            axs[row, col].set_xlabel("Stimulus", weight="bold", fontsize=10)
            axs[row, col].set_ylabel(
                "US Expectancy (observed data scale: 1-10)", weight="bold", fontsize=10
            )
            axs[row, col].set_facecolor("whitesmoke")

    # Create overall legend.
    overall_line = plt.Line2D(
        [], [], color="black", marker="o", label="Observed data group mean"
    )
    individual_line = plt.Line2D(
        [], [], color="gray", label="Observed data group individuals means"
    )
    fig.legend(
        handles=[overall_line, individual_line],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        fontsize=14,
        fancybox=True,
        shadow=True,
        ncol=2,
    )

    # Set overall titles.
    fig.text(
        0.5, 0.92, experiments[0], ha="center", va="center", fontsize=14, weight="bold"
    )
    fig.text(
        0.5, 0.47, experiments[1], ha="center", va="center", fontsize=14, weight="bold"
    )

    plt.subplots_adjust(hspace=0.5, top=0.87)
    fig.savefig(
        f"{output_dir}/Response_Patterns_Group_Generalization_Pattern.png", dpi=300
    )


# Plot both experiments groups generalization pattern.
ge_gr_plot(data_ge, experiments, RP_dir)


# Learning.


def lr_gr_plot(processed_data, experiments, output_dir):
    """
    Plots observed US expectancy across trials for CS+ and CS- showing both group
    averages and individual trajectories.

    Args:
        processed_data (list):
            List of processed DataFrames for both experiments.
        experiments (list):
            List of the experiments names for plotting.
        output_dir (str):
            Directory where the plots will be saved.
    """
    # Define color palette for plotting.
    color_palette = {
        "CS+": "#CC79A7",
        "CS-": "#56B4E9",
        "S4": "gray",
        "S5": "gray",
        "S6": "gray",
        "S8": "gray",
        "S9": "gray",
        "S10": "gray",
        "S2": "gray",
        "S7": "gray",
        "S3": "gray",
    }

    # Create the figure.
    fig, axs = plt.subplots(4, 2, figsize=(20, 12), sharey=True, sharex=True)

    # Iterate through experiments processed data.
    for idx, df in enumerate(processed_data):

        # Setup experiment specific US stimuli and trial range.
        valid_stimuli = ["CS+"] if idx == 0 else ["CS+", "CS-"]
        trial_range = range(1, [188, 180][idx] + 1)

        # Filter the experiment processed data based on the US stimuli (CS+, CS-)
        # and trial range.
        df_filtered = df[
            (df["stimulus"].isin(valid_stimuli)) & (df["trials"].isin(trial_range))
        ].copy()
        df_filtered = df_filtered.loc[df_filtered["Group_Name"] != "Unknown"]

        # Iterate through experiments processed data grouped by participants groups.
        for group_idx, (name, group_df) in enumerate(
            df_filtered.groupby("Group_Name", observed=True)
        ):

            # Create ax for the subplot.
            ax = axs[group_idx, idx]

            # Create a lineplot for the mean of by trial stimulus US expectancy
            # for the current group.
            sns.lineplot(
                data=group_df,
                x="trials",
                y="mean_us_trial",
                hue="stimulus",
                palette=color_palette,
                ax=ax,
                markers="o",
                linewidth=1,
                legend=False,
            )

            # Create lineplots for by trial stimulus US expectancy of each participant
            # allocated in the current group.
            for participant in group_df["Participant_Num"].unique():
                participant_data = group_df[group_df["Participant_Num"] == participant]
                sns.lineplot(
                    data=participant_data,
                    x="trials",
                    y="US_expect",
                    hue="stimulus",
                    palette=color_palette,
                    ax=ax,
                    alpha=0.1,
                    markers="o",
                    legend=False,
                )

            # Set subplot parameters.
            ax.set_title(f"{name}", weight="bold", fontsize=12)
            ax.set_ylim(1, 10)
            ax.set_ylim(0.5, 10.5)
            ax.set_yticks([1, 5, 10])
            ax.set_xlabel("Trials", weight="bold", fontsize=12)
            ax.set_ylabel("")
            ax.set_facecolor("whitesmoke")

    # Add overall y-axis label.
    fig.text(
        0.0015,
        0.5,
        "US Expectancy (observed data scale: 1-10)",
        va="center",
        weight="bold",
        rotation="vertical",
        fontsize=12,
    )

    # Add overall titles.
    fig.text(0.17, 0.99, experiments[0], va="top", fontsize=14, weight="bold")
    fig.text(0.63, 0.99, experiments[1], va="top", fontsize=14, weight="bold")

    # Create overall legend.
    custom_handles = [
        plt.Line2D([], [], color="#CC79A7", label="Observed response to CS+"),
        plt.Line2D([], [], color="#56B4E9", label="Observed response to CS-"),
    ]
    fig.legend(
        handles=custom_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.003),
        ncol=2,
        fancybox=True,
        shadow=True,
        fontsize=12,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.06)
    fig.savefig(f"{output_dir}/Response_Patterns_Learning.png", dpi=300)


# Plot both experiments groups observed learning in respect to US stimuli (CS+, CS-).
lr_gr_plot(processed_data, experiments, RP_dir)

# Load PyMC input data used to run the models of both studies.
pymc1 = pickle.load(open("../PYMC_input_data/Data1_PYMCinput_CLG.pkl", "rb"))
pymc2 = pickle.load(open("../PYMC_input_data/Data2_PYMCinput_CLG.pkl", "rb"))

# Extract and compute the median of alpha learning rate samples from the posterior distributions
# of both studies.
alpha = [
    np.nanpercentile(results["s1"]["CLG2D"].posterior["alpha"], 50, axis=0),
    np.nanpercentile(results["s2"]["CLG2D"].posterior["alpha"], 50, axis=0),
]

# Initialize matrices for the associative strengths of both studies.
v1 = np.zeros((pymc1["r"].shape[0], pymc1["r"].shape[1] + 1))
v2_plus = np.zeros((pymc2["r_plus"].shape[0], pymc2["r_plus"].shape[1] + 1))
v2_minus = np.zeros((pymc2["r_minus"].shape[0], pymc2["r_minus"].shape[1] + 1))

# Fill up the matrix of the first study by computing estimated excitatory associative strength
# using learning rate median samples.
for i in range(pymc1["r"].shape[0]):
    for j in range(1, pymc1["r"].shape[1] + 1):
        k = pymc1["k"][i, j - 1]
        r = pymc1["r"][i, j - 1]
        v1[i, j] = (
            v1[i, j - 1] + alpha[0][i] * (r - v1[i, j - 1]) if k == 1 else v1[i, j - 1]
        )

# Fill up the matrices of the second study by computing estimated excitatory and inhibitory
# associative strengths using learning rate median samples.
for i in range(pymc2["r_plus"].shape[0]):
    for j in range(1, pymc2["r_plus"].shape[1] + 1):
        kp = pymc2["k_plus"][i, j - 1]
        rp = pymc2["r_plus"][i, j - 1]
        km = pymc2["k_minus"][i, j - 1]
        rm = pymc2["r_minus"][i, j - 1]
        v2_plus[i, j] = (
            v2_plus[i, j - 1] + alpha[1][i] * (rp - v2_plus[i, j - 1])
            if kp == 1
            else v2_plus[i, j - 1]
        )
        v2_minus[i, j] = (
            v2_minus[i, j - 1] + alpha[1][i] * (rm - v2_minus[i, j - 1])
            if km == 1
            else v2_minus[i, j - 1]
        )

# Convert associative strengths matrices to wide format dataframes.
v1 = pd.DataFrame(
    v1[:, 1:], index=np.arange(1, len(v1) + 1), columns=np.arange(1, v1.shape[1])
)
v2_plus = pd.DataFrame(
    v2_plus[:, 1:],
    index=np.arange(1, len(v2_plus) + 1),
    columns=np.arange(1, v2_plus.shape[1]),
)
v2_minus = pd.DataFrame(
    v2_minus[:, 1:],
    index=np.arange(1, len(v2_minus) + 1),
    columns=np.arange(1, v2_minus.shape[1]),
)

# Reshape the dataframes from wide format to long format.
v1 = v1.reset_index().melt(id_vars="index", var_name="trials", value_name="vp")
v1.columns = ["Participant_Num", "trials", "vp"]
v2_plus = v2_plus.reset_index().melt(
    id_vars="index", var_name="trials", value_name="vp"
)
v2_plus.columns = ["Participant_Num", "trials", "vp"]
v2_minus = v2_minus.reset_index().melt(
    id_vars="index", var_name="trials", value_name="vm"
)
v2_minus.columns = ["Participant_Num", "trials", "vm"]

# Add the assocaitive strengths columns to the processed data for both studies.
processed_data[0] = processed_data[0].merge(v1, on=["Participant_Num", "trials"])
processed_data[1] = processed_data[1].merge(v2_plus, on=["Participant_Num", "trials"])
processed_data[1] = processed_data[1].merge(
    v2_minus, on=["Participant_Num", "trials"], suffixes=("_plus", "_minus")
)

# Create column for the mean of the group excitatory associative strength.
processed_data[0]["mean_vp"] = (
    processed_data[0]
    .groupby(["Group_Name", "trials"], observed=True)["vp"]
    .transform("mean")
)
# Create column for the mean of the group excitatory associative strength (second study).
processed_data[1]["mean_vp"] = (
    processed_data[1]
    .groupby(["Group_Name", "trials"], observed=True)["vp"]
    .transform("mean")
)
# Create column for the mean of the group inhibitory associative strength (second study).
processed_data[1]["mean_vm"] = (
    processed_data[1]
    .groupby(["Group_Name", "trials"], observed=True)["vm"]
    .transform("mean")
)


# Simulated Learning.


def v_gr_plot(processed_data, experiments, output_dir):
    """
    Visualizes simulated estimated associative strength adjustments over trials
    for both CS+ and CS-, showing both group averages and individual trajectories.

    Args:
        processed_data (list):
            List of processed data frames for each experiment.
        experiments (list):
            List of the experiments names for plotting.
        output_dir (str):
            Directory where the plots will be saved.
    """
    # Create figure
    fig, axs = plt.subplots(4, 2, figsize=(20, 12), sharey=True, sharex=True)

    # Iterate through experiments processed data.
    for col, df in enumerate(processed_data):

        # Retriwve unique dominant groups.
        df = df.loc[df["Group_Name"] != "Unknown"]
        groups = df["Group_Name"].sort_values().unique()

        # Iterate through the groups.
        for idx, group in enumerate(groups):

            # Filter current experiment processed data by the current group.
            group_df = df[df["Group_Name"] == group]

            # Create lineplot for the mean of estimated excitatory associative strength
            # for the current group.
            sns.lineplot(
                data=group_df,
                x="trials",
                y="mean_vp",
                ax=axs[idx, col],
                color="#e41a1c",
                markers="o",
                linewidth=1,
                legend=False,
            )

            # Create lineplots for the mean of estimated excitatory associative strength 
            # for each participant allocated in the current group.
            sns.lineplot(
                data=group_df,
                x="trials",
                y="vp",
                hue="Participant_Num",
                palette=["#e41a1c" for _ in group_df["Participant_Num"].unique()],
                markers="o",
                ax=axs[idx, col],
                alpha=0.1,
                legend=False,
                errorbar=None,
            )

            # Plot for 'vm' if it exists.
            if "mean_vm" in df.columns:

                # Create lineplot for the mean of estimated inhibitory associative strength
                # for the current group.
                sns.lineplot(
                    data=group_df,
                    x="trials",
                    y="mean_vm",
                    ax=axs[idx, col],
                    color="#377eb8",
                    markers="o",
                    linewidth=1,
                    legend=False,
                )

                # Create lineplots for the mean of estimated inhibitory associative strength 
                # for each participant allocated in the current group.
                sns.lineplot(
                    data=group_df,
                    x="trials",
                    y="vm",
                    hue="Participant_Num",
                    palette=["#377eb8" for _ in group_df["Participant_Num"].unique()],
                    markers="o",
                    ax=axs[idx, col],
                    alpha=0.1,
                    legend=False,
                    errorbar=None,
                )

            # Set subplot parameters.
            axs[idx, col].set_title(f"{group} Learning", weight="bold", fontsize=12)
            axs[idx, col].set_xlabel("Trials", weight="bold", fontsize=12)
            axs[idx, col].set_ylabel("")
            axs[idx, col].set_ylim(-1, 1)
            axs[idx, col].set_facecolor("whitesmoke")

    # Create overall legend.
    custom_handles = [
        plt.Line2D([], [], color="#e41a1c", label="CS+ strength"),
        plt.Line2D([], [], color="#377eb8", label="CS- strength"),
    ]
    fig.legend(
        handles=custom_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.003),
        ncol=2,
        fancybox=True,
        shadow=True,
        fontsize=12,
    )

    # Add overall y-axis label.
    fig.text(
        0.0015,
        0.5,
        "Associative Strengths",
        va="center",
        weight="bold",
        rotation="vertical",
        fontsize=12,
    )

    # Add overall titles.
    fig.text(0.17, 0.99, experiments[0], va="top", fontsize=14, weight="bold")
    fig.text(0.63, 0.99, experiments[1], va="top", fontsize=14, weight="bold")

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.06)
    fig.savefig(f"{output_dir}/Response_Patterns_Associative_Strengths.png", dpi=300)


# Plot both experiments groups simulated learning in respect to US stimuli (CS+, CS-).
v_gr_plot([processed_data[0], processed_data[1]], experiments, RP_dir)


# Lambda generalization rate.


def lambda_plot(lambda_values, experiments, output_dir):
    """
    Displays histograms of the lambda generalization rate from the posterior
    distributions, offering insights into the spread and concentration of these
    parameter distribution under different experimental conditions.

    Args:
        lambda_values (dict):
            Lambda values for both studies.
        experiments (list):
            List of the experiments names connected to the studies for plotting.
        output_dir (str):
            Directory where the plots will be saved.
    """
    # Convert the lambda arrays to a DataFrame.
    lambdaposterior = pd.DataFrame(lambda_values)

    # Create the figure.
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Initialize subplots parameters.
    bin_settings = [(0.0052, 0.01, 0.1, 0.2, 0.3), (0, 0.0052, 0.01)]
    limits = [0.3, 0.01]

    # Iterate through subplots axes.
    for i, ax in enumerate(axs.flat):

        # Retrieve study specific lambda generalization rate posterior samples
        # based on current subplot ax.
        exp_col = "study1" if i % 2 == 0 else "study2"
        limit = limits[i // 2]
        data_to_plot = lambdaposterior[lambdaposterior[exp_col] <= limit]

        # Histogram plot of the lambda generalization rate for the current subplot.
        sns.histplot(
            data_to_plot[exp_col],
            bins=6,
            kde=False,
            color="red",
            ax=ax,
            line_kws={"edgecolor": "black"},
            alpha=0.5,
        )

        # Set subplot parameters.
        ax.axvline(x=0.0052, color="black", linestyle="dashed", linewidth=1.5)
        ax.set_title("")
        x_range = limit if i < 2 else 0.01
        x_pad = 0.05 * x_range
        y_max = ax.get_ylim()[1]
        ax.set_xlim([-x_pad, x_range + x_pad])
        ax.set_ylim([-0.05 * y_max, 1.05 * y_max])
        ax.set_xlabel("\u03bb posterior", weight="bold", fontsize=12)
        ax.set_ylabel("Posterior samples", weight="bold", fontsize=12)
        ax.set_xticks(bin_settings[0] if i < 2 else bin_settings[1])
        ax.set_facecolor("whitesmoke")

    # Add overall titles.
    fig.text(0.13, 0.99, experiments[0], va="top", fontsize=14, weight="bold")
    fig.text(0.6, 0.99, experiments[1], va="top", fontsize=14, weight="bold")

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    fig.savefig(f"{output_dir}/Response_Patterns_Lambda_Posterior.png", dpi=300)


# Retrieve subsets of the posterior distributions lambda generalization rates samples
# for both studies for plotting.
selected_indices1 = np.random.choice(
    results["s1"]["CLG2D"].posterior["lambda"].shape[0],
    5000,
    replace=False,
)
selected_indices2 = np.random.choice(
    results["s2"]["CLG2D"].posterior["lambda"].shape[0],
    5000,
    replace=False,
)

# Extract lambda generalization rates samples of both studies from the posterior distributions.
lambda_values = {
    "study1": results["s1"]["CLG2D"]
    .posterior["lambda"][selected_indices1, :]
    .flatten(),
    "study2": results["s2"]["CLG2D"]
    .posterior["lambda"][selected_indices2, :]
    .flatten(),
}

# Plot lambda generalization rate from the posterior for both sudies.
lambda_plot(lambda_values, experiments, RP_dir)


# Perception.


def per_gr_mean_plot(processed_data, group, ax, Ntrials):
    """
    Helper function to plot the mean perceived stimulus distance from CS+
    for a specific group within an experiment.

    Args:
        processed_data (DataFrame):
            Data containing group, stimulus and dispersion information.
        group (str):
            The name of the group to filter data by.
        ax (AxesSubplot):
            Matplotlib subplot object to plot the data.
        Ntrials (int):
            Total number of trials in the experiment.
    """
    # Retrieve group specific data.
    group_data = processed_data[processed_data["Group_Name"] == group]

    # Create lineplot for the group specific mean of the perceived stimulus distance
    # from CS+.
    sns.lineplot(
        data=group_data,
        x="stimulus",
        y="mean_disp_group",
        ax=ax,
        marker="o",
        color="skyblue",
        legend=False,
    )

    # Create lineplots for the mean of the perceived stimulus distance from CS+ of
    # each participant allocated to the specific group.
    sns.lineplot(
        data=group_data,
        x="stimulus",
        y="mean_disp_indi",
        hue="Participant_Num",
        palette=["skyblue" for _ in group_data["Participant_Num"].unique()],
        ax=ax,
        marker="o",
        alpha=0.1,
        legend=False,
    )

    # Set subplot parameters.
    ax.set_title(f"{group}", weight="bold", fontsize=12)
    ax.set_ylabel("Distance to CS+", weight="bold", fontsize=12)
    ax.set_xlabel("Stimulus", weight="bold", fontsize=12)
    ax.set_facecolor("whitesmoke")

    # Add group specific proportion annotation.
    ax.annotate(
        f"N = {len(group_data['Participant_Num'].unique())}, Proportion = {len(group_data['Participant_Num'].unique()) / Ntrials:.2f}",
        (0.5, 0.02),
        xycoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=12,
    )


def per_gr_sd_plot(processed_data, group, ax, Ntrials):
    """
    Helper function to plot the standard deviation of perceived stimulus distance
    from CS+ for a specific group within an experiment.

    Args:
        processed_data (DataFrame):
            Data containing group, stimulus and dispersion information.
        group (str):
            The name of the group to filter data by.
        ax (AxesSubplot):
            Matplotlib subplot object to plot the data.
        Ntrials (int):
            Total number of trials in the experiment.
    """
    # Retrieve group specific data.
    group_data = processed_data[processed_data["Group_Name"] == group]

    # Create lineplot for the group specific standard deviation of the perceived
    # stimulus distance from CS+.
    sns.lineplot(
        data=group_data,
        x="stimulus",
        y="sd_disp_group",
        ax=ax,
        marker="o",
        color="orange",
        legend=False,
    )

    # Create lineplots for the standard deviation of the perceived stimulus distance
    # from CS+ of each participant allocated to the specific group.
    sns.lineplot(
        data=group_data,
        x="stimulus",
        y="sd_disp_indi",
        hue="Participant_Num",
        palette=["orange" for _ in group_data["Participant_Num"].unique()],
        ax=ax,
        marker="o",
        alpha=0.1,
        legend=False,
    )

    # Set subplot parameters.
    ax.set_title(f"{group}", weight="bold", fontsize=12)
    ax.set_ylabel("Distance to CS+ (sd)", weight="bold", fontsize=12)
    ax.set_xlabel("Stimulus", weight="bold", fontsize=12)
    ax.set_facecolor("whitesmoke")

    # Add group specific proportion annotation.
    ax.annotate(
        f"N = {len(group_data['Participant_Num'].unique())}, Proportion = {len(group_data['Participant_Num'].unique()) / Ntrials:.2f}",
        (0.5, 0.02),
        xycoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=12,
    )


def per_gr_plot(processed_data, experiment, Ntrials, output_dir):
    """
    Arranges and displays perception data for groups within an experiment, including
    both mean and standard deviation of perceived stimulus distance from CS+ across
    all stimuli.

    Args:
        processed_data (DataFrame):
            Data containing perceptions across trials.
        experiment (str):
            The name of the experiment for the plot title.
        Ntrials (int):
            Total number of trials in the experiment.
        output_dir (str):
            Directory where the plots will be saved.
    """
    # Create the figure.
    fig, axs = plt.subplots(2, 4, figsize=(24, 10), sharey="row", sharex="col")

    # Retrieve unique dominant groups names.
    processed_data = processed_data.loc[processed_data["Group_Name"] != "Unknown"]
    groups = processed_data["Group_Name"].sort_values().unique()

    # Iterate trough the unique groups.
    for i, group in enumerate(groups):

        # Create subplots for the current group overall and individual mean of the
        # perceived stimulus distance from CS+.
        per_gr_mean_plot(processed_data, group, axs[0, i], Ntrials)

        # Create subplots for the current group overall and individual standard deviation
        # of the perceived stimulus distance from CS+.
        per_gr_sd_plot(processed_data, group, axs[1, i], Ntrials)

    # Create overall legend.
    custom_handles = [
        plt.Line2D(
            [], [], color="skyblue", marker="o", label="Perception Mean Dispersion "
        ),
        plt.Line2D(
            [],
            [],
            color="orange",
            marker="o",
            label="Perception Standard Deviation Dispersion",
        ),
    ]
    fig.legend(
        handles=custom_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        fontsize=14,
        fancybox=True,
        shadow=True,
        ncol=2,
    )

    plt.suptitle(experiment, weight="bold", fontsize=14)
    fig.savefig(
        f"{output_dir}/Response_Patterns_Perception_{experiment.replace(" ", "_")}.png",
        dpi=300,
    )


# Plot experiment 1 groups overall and individual means and standard deviations of the
# perceived stimulus distance from CS+.
per_gr_plot(processed_data[0], experiments[0], 188, RP_dir)
# Plot experiment 2 groups overall and individual means and standard deviations of the
# perceived stimulus distance from CS+.
per_gr_plot(processed_data[1], experiments[1], 180, RP_dir)


# Similarity simulation.

# Rename columns for the experiment 1 for easier computations.
processed_data[0] = processed_data[0].rename(
    columns={"dphy": "d_phy_p", "dper": "d_per_p"}
)

# Retrieve lambda generalization rate samples from the posterior distributions
# of both studies, create a DataFrame for each study and fill it with the
# 50% quantile of the lambda samples.
m_lambda = {
    "s1": pd.DataFrame(
        {
            "Participant_Num": [i for i in range(1, 41)],
            "lambda": np.nanpercentile(
                results["s1"]["CLG2D"].posterior["lambda"], 50, axis=0
            ),
        }
    ),
    "s2": pd.DataFrame(
        {
            "Participant_Num": [i for i in range(1, 41)],
            "lambda": np.nanpercentile(
                results["s2"]["CLG2D"].posterior["lambda"], 50, axis=0
            ),
        }
    ),
}

# Iterate through both experiments processed data.
for i, df in enumerate(processed_data):

    # Merge the study lambda 50% quantile DataFrame with the equivalent experiment
    # processed data DataFrame.
    processed_data[i] = df.merge(m_lambda[f"s{i+1}"], on="Participant_Num", how="left")


# Plot similarity to CS+ .
def sim_gr_plot(processed_data, experiments, output_dir):
    """
    Calculates and plots estimated similarity metrics based on median generalization 
    rate samples influenced by perceptual and physical distances to CS+ stimuli,
    differentiated by dominant participants groups.

    Args:
        data (list):
            List of DataFrames for both experiments containing associative strength
            and distance information.
        experiments (list):
            Names of the experiments to annotate the plots.
        output_dir (str):
            Directory where the plots will be saved.
    """
    # Create the figure.
    fig, axs = plt.subplots(4, 2, figsize=(20, 12), sharey=True, sharex=True)

    # Flatten the axes array for easier handling.
    axs = axs.flatten() if len(processed_data) == 1 else axs.T.flatten()

    # Iterate through the experiments processed data DataFrames.
    for exp_index, df in enumerate(processed_data):

        df = df.copy()
        df = df.loc[df["Group_Name"] != "Unknown"]
        groups = df["Group_Name"].sort_values().unique()

        # Compute estimated similarity to CS+ for the current experiment and add 
        # it to the processed data DataFrame.
        df["similarity"] = np.where(
            df["Group_Name"] == "Non-Learners",
            1,
            np.where(
                df["Group_Name"] == "Perceptual Generalizers",
                np.exp(-df["lambda"] * df["d_per_p"]),
                np.exp(-df["lambda"] * df["d_phy_p"]),
            ),
        )

        # Create a stim column for the processed data DataFrame of the current
        # experiment which keeps only relevant stimuli 'types'.
        df["stim"] = np.where(
            df["stimulus"] == "CS+",
            "CS+",
            np.where(df["stimulus"] == "CS-", "CS-", "TS"),
        )

        # Retrieve unique dominant groups names.
        groups = df["Group_Name"].sort_values().unique()

        # Iterate through unique groups.
        for i, group in enumerate(groups):

            # Filter the current experiment processed data by the current group.
            group_data = df[df["Group_Name"] == group]

            # Setup subplot ax.
            ax = axs[exp_index * 4 + i]

            # Create scatter and line plots for physical stimulus distances to CS+.
            sns.lineplot(
                x=group_data["d_phy_p"] + 10,
                y="similarity",
                hue="stim",
                palette={"CS+": "grey", "CS-": "grey", "TS": "grey"},
                data=group_data,
                ax=ax,
                alpha=0.25,
                legend=False,
            )
            sns.scatterplot(
                x=group_data["d_phy_p"] + 10,
                y="similarity",
                hue="stim",
                data=group_data,
                ax=ax,
                palette={"CS+": "#e41a1c", "CS-": "#377eb8", "TS": "grey"},
                alpha=0.5,
                legend=False,
            )

            # Create scatter and line plots for perceived stimulus similarities to CS+.
            sns.lineplot(
                x=-group_data["d_per_p"] - 10,
                y="similarity",
                hue="stim",
                palette={"CS+": "grey", "CS-": "grey", "TS": "grey"},
                data=group_data,
                ax=ax,
                alpha=0.25,
                legend=False,
            )
            sns.scatterplot(
                x=-group_data["d_per_p"] - 10,
                y="similarity",
                hue="stim",
                data=group_data,
                ax=ax,
                palette={"CS+": "#e41a1c", "CS-": "#377eb8", "TS": "grey"},
                alpha=0.5,
                legend=False,
            )

            # Set subplot parameters.
            ax.axvline(0, color="black", linestyle="dashed", linewidth=1)
            ax.set_title(f"Similarity to CS+ for {group}", fontsize=12, weight="bold")
            ax.set_xlabel(
                "Distance to CS+ (left: perceptual; right: physical)",
                fontsize=12,
                weight="bold",
            )
            ax.set_ylabel("")
            ax.set_xlim([-150, 150])
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, pos: f"{abs(x)}")
            )
            ax.set_facecolor("whitesmoke")

    # Create overall legend.
    custom_handles = [
        plt.Line2D(
            [],
            [],
            color="#e41a1c",
            marker="o",
            markersize=10,
            label="CS+",
            linestyle="None",
        ),
        plt.Line2D(
            [],
            [],
            color="#377eb8",
            marker="o",
            markersize=10,
            label="CS-",
            linestyle="None",
        ),
        plt.Line2D(
            [],
            [],
            color="grey",
            marker="o",
            markersize=10,
            label="TS",
            linestyle="None",
        ),
    ]
    fig.legend(
        handles=custom_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.003),
        ncol=3,
        fancybox=True,
        shadow=True,
        fontsize=12,
    )

    # Add overall y-axis label.
    fig.text(
        0.0015,
        0.5,
        "Similarity to CS+",
        va="center",
        weight="bold",
        rotation="vertical",
        fontsize=12,
    )

    # Add overall titles.
    fig.text(0.17, 0.99, experiments[0], va="top", fontsize=14, weight="bold")
    fig.text(0.63, 0.99, experiments[1], va="top", fontsize=14, weight="bold")

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.06)
    fig.savefig(f"{output_dir}/Response_Patterns_Similarity_Simulation.png", dpi=300)


# Plot similarity to CS+ for each dominant group for both studies.
sim_gr_plot(processed_data, experiments, RP_dir)

