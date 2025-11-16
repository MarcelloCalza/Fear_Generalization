"""
Module: data_visualization.py

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
conversion of the original R code to Python, it is designed to handle visualization
tasks, regarding the data acquired during the experiments, that involve analyzing
learning outcomes, stimulus generalization and perceptual accuracy of participants.
It employs statistical plotting techniques to represent complex behavioral data in a
comprehensible manner, facilitating deeper insights into the data collected from
the experiments.

The key functionalities of this module include:
- Visualization of learning processes by representing the association between
  stimuli and expected outcomes.
- Detailed analysis of generalization across different stimuli to understand
  how responses generalize beyond conditioned stimuli.
- Examination of perception accuracy by comparing physical and perceived sizes
  of stimuli across participants.

### Functions:
- dv_lr_gr_plot: Plots overall learning responses of a given experiment.
- dv_lr_indi_plot: Displays learning curves for individual participants of
  a given experiment.
- dv_ge_gr_plot: Visualizes overall generalization responses for a given experiment.
- dv_ge_indi_plot: Shows individual participant responses to generalization trials
  of a given experiment.
- dv_pe_gr_plot: Creates plots to visualize overall perceived sizes of stimuli
  of a given experiment.
- dv_pe_indi_plot: Displays individual perceptions of stimuli sizes against actual sizes
  of a given experiment.
- dv_cor_plot: Analyzes the correlation between the physical and perceived sizes
  of stimuli for each participant of a given experiment, highlighting the perceptual
  accuracy and biases present.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
from scipy.stats import pearsonr
import os

# Create a directory to save the plots.
os.makedirs("../Plots", exist_ok=True)

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
Nactrials = [14, 24]
stimulus_levels = {
    "s1": ["S4", "S5", "S6", "CS+", "S8", "S9", "S10"],
    "s2": ["CS+", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "CS-"],
}

# Create a directory to save data visualization plots.
DV_dir = "../Plots/Data Visualization/"
os.makedirs(DV_dir, exist_ok=True)


# Learning.


# Overall Learning.
def dv_lr_gr_plot(data_i, Nactrials, experiment, output_dir):
    """
    Visualizes learning curves for a specific experiment, displaying
    overall US expectancy across learning trials.

    This function processes the experimental data to visualize learning across
    learning trials specifically focusing on conditioned stimuli (CS+ and CS-).
    It computes and plots the overall and individual mean US expectancies.
    This visualization aids in comparing excitatory and inhibitory learning effects
    across the different conditioning protocols.

    Args:
        data_i (DataFrame):
            Long format DataFrame containing the data of the experiment.
        Nactrials (int):
            Number of learning trials of the experiment.
        experiment (str):
            Name of the experiment to handle, affects plot title.
        output_dir (str):
            Directory where the plots will be saved.
    """
    # Filter the data by the learning trials and the relevant stimuli.
    filtered_data = data_i[
        (data_i["trials"] <= Nactrials) & (data_i["stimulus"].isin(["CS+", "CS-"]))
    ].copy()

    # Create a column for overall means and standard deviations of stimulus
    # US expectancy across trials for plotting.
    filtered_data["mean_ac"] = filtered_data.groupby(
        ["trials", "stimulus"], observed=True
    )["US_expect"].transform("mean")
    filtered_data["sd_ac"] = filtered_data.groupby(
        ["trials", "stimulus"], observed=True
    )["US_expect"].transform("std")

    # Create a column for individual means and standard deviations of stimulus
    # US expectancy across trials for plotting.
    filtered_data["mean_ac_indi"] = filtered_data.groupby(
        ["trials", "stimulus", "participant"], observed=True
    )["US_expect"].transform("mean")
    filtered_data["sd_ac_indi"] = filtered_data.groupby(
        ["trials", "stimulus", "participant"], observed=True
    )["US_expect"].transform("std")

    # Create the figure.
    plt.figure(figsize=(9, 6))

    # Generate lineplot for the overall mean of the stimulus US expectancy
    # across trials.
    sns.lineplot(
        x="trials",
        y="mean_ac",
        hue="stimulus",
        data=filtered_data,
        marker="o",
        palette={"CS+": "red", "CS-": "#56B4E9"},
        legend=None,
        zorder=1,
    )

    # Generate lineplots for stimulus US expectancy mean across trials
    # for each participant.
    for participant in filtered_data["participant"].unique():
        participant_data = filtered_data[filtered_data["participant"] == participant]
        sns.lineplot(
            x="trials",
            y="mean_ac_indi",
            hue="stimulus",
            data=participant_data,
            palette={"CS+": "red", "CS-": "#56B4E9"},
            legend=None,
            alpha=0.05,
            zorder=0,
        )

    # Retrieve plot ax.
    ax = plt.gca()

    # Add title.
    plt.title(f"Acquisition: {experiment}", fontsize=14, weight="bold")

    # Set plot parameters.
    plt.xlabel("Trials", fontsize=12, weight="bold")
    plt.ylabel("US expectancy (1 - 10)", fontsize=12, weight="bold")
    plt.ylim(0, 10.5)
    plt.yticks([1, 5, 10])
    plt.xticks(range(1, Nactrials + 1, 2))
    ax.set_facecolor("whitesmoke")

    # Create legend.
    legend_handles = []
    if "CS+" in filtered_data["stimulus"].unique():
        legend_handles.append(
            plt.Line2D(
                [],
                [],
                color="red",
                marker="o",
                label="Excitatory learning",
                linestyle="-",
            )
        )
    if "CS-" in filtered_data["stimulus"].unique():
        legend_handles.append(
            plt.Line2D(
                [],
                [],
                color="#56B4E9",
                marker="o",
                label="Inhibitory learning",
                linestyle="-",
            )
        )
    plt.legend(title="", handles=legend_handles, loc="upper right", fontsize=12)

    # Save the plot.
    plt.savefig(
        f"{output_dir}/Learning_Overall_{experiment.replace(" ", "_")}.png", dpi=300
    )
    plt.close()


# Plot overall learning for experiment 1: simple conditioning.
dv_lr_gr_plot(data_["s1"], Nactrials[0], experiments[0], DV_dir)
# Plot overall learning for experiment 2: differential conditioning.
dv_lr_gr_plot(data_["s2"], Nactrials[1], experiments[1], DV_dir)


# Individual Learning.
def dv_lr_indi_plot(data_i, Nactrials, experiment, output_dir):
    """
    Visualizes individual learning curves for a specific experiment, displaying US
    expectancy across learning trials for each participant.

    This function filters the experimental data for conditioned stimuli (CS+ and CS-)
    and learning trials, then plots individual learning curves for each participant.
    It creates a grid of subplots where each subplot represents a participant's
    learning curve over the learning trials. This detailed plot helps in analyzing
    the variance in learning responses across individuals within the same experimental
    conditions.

    Args:
        data_i (DataFrame):
            Long format DataFrame containing the data of the experiment.
        Nactrials (int):
            Number of learning trials of the experiment.
        experiment (str):
            Name of the experiment to handle, affects plot title.
        output_dir (str):
            Directory where the plots will be saved.
    """
    # Filter the data by the learning trials and the relevant stimuli.
    filtered_data = data_i[
        (data_i["trials"] <= Nactrials) & (data_i["stimulus"].isin(["CS+", "CS-"]))
    ].copy()

    # Create a column for individual means of stimulus US expectancy across trials
    # for plotting.
    filtered_data["mean_ac"] = filtered_data.groupby(
        ["participant", "trials", "stimulus"], observed=True
    )["US_expect"].transform("mean")

    # Create the figure and axes.
    participants = filtered_data["participant"].sort_values().unique()
    num_participants = len(participants)
    cols = 5
    rows = (num_participants // 5) + (num_participants % 5 > 0)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 18), sharex=True, sharey=True)
    axes = axes.flatten()

    # Generate lineplots for the mean of stimulus US expectancy across trials
    # for each participant.
    for i, participant in enumerate(participants):
        ax = axes[i]
        participant_data = filtered_data[filtered_data["participant"] == participant]
        for stimulus, group_data in participant_data.groupby("stimulus"):
            ax.scatter(
                group_data["trials"],
                group_data["mean_ac"],
                color={"CS+": "red", "CS-": "#56B4E9"}[stimulus],
            )
            ax.plot(
                group_data["trials"],
                group_data["mean_ac"],
                color={"CS+": "red", "CS-": "#56B4E9"}[stimulus],
            )

        # Set subplot parameters.
        ax.set_title(f"Participant {participant}", weight="bold", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_ylim(0, 10.5)
        ax.set_xticks(range(1, Nactrials + 1, 6))
        ax.set_facecolor("whitesmoke")

    # Create overall x-label and y-label for the whole figure.
    fig.text(0.5, 0.03, "Trials", ha="center", va="center", weight="bold", fontsize=12)
    fig.text(
        0.006,
        0.5,
        "US Expectancy (1-10)",
        ha="center",
        va="center",
        rotation="vertical",
        weight="bold",
        fontsize=12,
    )

    # Create overall legend.
    legend_handles = []
    if "CS+" in filtered_data["stimulus"].unique():
        legend_handles.append(
            plt.Line2D(
                [],
                [],
                color="red",
                marker="o",
                label="Excitatory learning",
                linestyle="-",
            )
        )
    if "CS-" in filtered_data["stimulus"].unique():
        legend_handles.append(
            plt.Line2D(
                [],
                [],
                color="#56B4E9",
                marker="o",
                label="Inhibitory learning",
                linestyle="-",
            )
        )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=2,
        fancybox=True,
        shadow=True,
        fontsize=12,
    )

    # Add title and adjust layout.
    plt.suptitle(f"Acquisition: {experiment}", weight="bold", fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05)
    plt.subplots_adjust(hspace=0.3)

    # Save the plot.
    plt.savefig(
        f"{output_dir}/Learning_Individual_{experiment.replace(" ", "_")}.png", dpi=300
    )
    plt.close(fig)


# Plot individual learning for experiment 1: simple conditioning.
dv_lr_indi_plot(data_["s1"], Nactrials[0], experiments[0], DV_dir)
# Plot individual learning for experiment 2: differential conditioning.
dv_lr_indi_plot(data_["s2"], Nactrials[1], experiments[1], DV_dir)


# Generalization.


# Overall Generalization.
def dv_ge_gr_plot(data_i, Nactrials, stimulus_level, experiment, output_dir):
    """
    Visualizes overall and individual generalization responses for a specifiec
    experiment, highlighting variations in US expectancy across different stimuli.

    This function filters the experimental data to focus on generalization trials
    beyond the specified number of initial learning trials. It calculates and plots
    the mean US expectancy for each stimulus both on an individual participant basis
    and aggregated across all participants. The function ensures that stimuli are
    ordered according to a predefined hierarchy, facilitating direct comparisons
    of generalization responses to different stimuli levels.

    Args:
        data_i (DataFrame):
            Long format DataFrame containing the data of the experiment.
        Nactrials (int):
            Number of learning trials of the experiment.
        stimulus_level (list):
            Predefined order of stimuli for plotting.
        experiment (str):
            Name of the experiment to handle, affects plot title.
        output_dir (str):
            Directory where the plots will be saved.
    """
    # Filter the data to include only generalization trials.
    filtered_data = data_i[data_i["trials"] > Nactrials].copy()

    # Create a column for overall means and standard deviations of stimulus
    # US expectancy across stimuli for plotting.
    filtered_data["mean_ge"] = filtered_data.groupby(["stimulus"], observed=True)[
        "US_expect"
    ].transform("mean")
    filtered_data["sd_ge"] = filtered_data.groupby(["stimulus"], observed=True)[
        "US_expect"
    ].transform("std")

    # Create a column for individual means and standard deviations of stimulus
    # US expectancy across stimuli for plotting.
    filtered_data["mean_ge_indi"] = filtered_data.groupby(
        ["stimulus", "participant"], observed=True
    )["US_expect"].transform("mean")
    filtered_data["sd_ge_indi"] = filtered_data.groupby(
        ["stimulus", "participant"], observed=True
    )["US_expect"].transform("std")

    # Order the stimuli based on the stimulus_level parameter for plotting.
    filtered_data["stimulus"] = pd.Categorical(
        filtered_data["stimulus"], categories=stimulus_level, ordered=True
    )

    # Create the figure.
    plt.figure(figsize=(9, 6))

    # Generate lineplots for the individual/participant mean stimulus US expectancy.
    for participant in filtered_data["participant"].unique():
        participant_data = filtered_data[filtered_data["participant"] == participant]
        sns.lineplot(
            data=participant_data,
            x="stimulus",
            y="mean_ge_indi",
            color="grey",
            alpha=0.2,
            legend=None,
            zorder=0,
        )

    # Generate lineplot for the overall mean stimulus US expectancy.
    sns.lineplot(
        data=filtered_data,
        x="stimulus",
        y="mean_ge",
        marker="o",
        color="black",
        legend=None,
        zorder=1,
    )

    # Retrieve plot ax.
    ax = plt.gca()

    # Add title.
    plt.title(f"Generalization: {experiment}", weight="bold", fontsize=14)

    # Set plot parameters.
    plt.xlabel("Stimulus", weight="bold", fontsize=12)
    plt.ylabel("US expectancy (1 - 10)", weight="bold", fontsize=12)
    plt.ylim(0, 10.5)
    plt.yticks([1, 5, 10])
    ax.set_facecolor("whitesmoke")

    # Create legend.
    legend_handles = []
    legend_handles.append(
        plt.Line2D(
            [],
            [],
            color="black",
            marker="o",
            label="Overall mean response",
            linestyle="-",
        )
    )
    legend_handles.append(
        plt.Line2D(
            [],
            [],
            color="grey",
            alpha=0.5,
            label="Individual mean response",
            linestyle="-",
        )
    )
    plt.legend(title="", handles=legend_handles, loc="upper right", fontsize=12)

    # Save the plot.
    plt.savefig(
        f"{output_dir}/Generalization_Overall_{experiment.replace(" ", "_")}.png",
        dpi=300,
    )
    plt.close()


# Plot individual generalization for experiment 1: simple conditioning.
dv_ge_gr_plot(data_["s1"], Nactrials[0], stimulus_levels["s1"], experiments[0], DV_dir)
# Plot individual generalization for experiment 2: differential conditioning.
dv_ge_gr_plot(data_["s2"], Nactrials[1], stimulus_levels["s2"], experiments[1], DV_dir)


# Individual Generalization.
def dv_ge_indi_plot(data_i, Nactrials, stimulus_level, experiment, output_dir):
    """
    Visualizes individual generalization responses across different stimuli for
    a specific experiment, showing variations in US expectancy.

    This function filters data to focus only on generalization trials after the initial
    learning phase. It calculates the mean US expectancy for each stimulus on an
    individual participant basis, plotting these means along with standard deviations
    as error bars for clarity. The stimuli are ordered according to a predefined
    hierarchy to facilitate comparison. This visualization provides a detailed view
    of each participant's response to different stimuli levels, highlighting individual
    differences in generalization behavior.

    Args:
        data_i (DataFrame):
            Long format DataFrame containing the data of the experiment.
        Nactrials (int):
            Number of learning trials of the experiment.
        stimulus_level (list):
            Predefined order of stimuli for plotting.
        experiment (str):
            Name of the experiment to handle, affects plot title.
        output_dir (str):
            Directory where the plots will be saved.
    """
    # Filter the data to include only generalization trials.
    filtered_data = data_i[data_i["trials"] > Nactrials].copy()

    # Create a column for individual means of stimulus US expectancy across stimuli
    # for plotting.
    filtered_data["mean_ge"] = filtered_data.groupby(
        ["stimulus", "participant"], observed=True
    )["US_expect"].transform("mean")
    filtered_data["sd_ge"] = filtered_data.groupby(
        ["stimulus", "participant"], observed=True
    )["US_expect"].transform("std")

    # Order the stimuli based on the stimulus_level parameter for plotting.
    filtered_data["stimulus"] = pd.Categorical(
        filtered_data["stimulus"], categories=stimulus_level, ordered=True
    )

    # Create the figure and axes.
    participants = filtered_data["participant"].sort_values().unique()
    num_participants = len(participants)
    cols = 5
    rows = (num_participants // 5) + (num_participants % 5 > 0)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 18), sharex=True, sharey=True)
    axes = axes.flatten()

    # Generate lineplots and errorbars for the mean of stimulus US expectancy across
    # trials for each participant.
    for i, participant in enumerate(participants):
        ax = axes[i]
        participant_data = filtered_data[filtered_data["participant"] == participant]
        sns.lineplot(
            data=participant_data,
            x="stimulus",
            y="mean_ge",
            marker="o",
            color="black",
            legend=None,
            ax=ax,
            zorder=1,
        )
        ax.errorbar(
            participant_data["stimulus"],
            participant_data["mean_ge"],
            yerr=[participant_data["sd_ge"].values, participant_data["sd_ge"].values],
            color="gray",
            fmt="o",
            zorder=0,
            capsize=5,
            alpha=0.3,
        )

        # Set subplot parameters.
        ax.set_title(f"Participant {participant}", weight="bold", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_ylim(-1.5, 12.5)
        ax.set_yticks([1, 5, 10])
        ax.set_facecolor("whitesmoke")

    # Create overall x-label and y-label for the whole figure.
    fig.text(
        0.5, 0.03, "Stimulus", ha="center", va="center", weight="bold", fontsize=12
    )
    fig.text(
        0.006,
        0.5,
        "US Expectancy (1-10)",
        ha="center",
        va="center",
        rotation="vertical",
        weight="bold",
        fontsize=12,
    )

    # Create overall legend.
    legend_handles = []
    legend_handles.append(
        plt.Line2D(
            [],
            [],
            color="black",
            marker="o",
            label="Overall individual response",
            linestyle="-",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=1,
        fancybox=True,
        shadow=True,
        fontsize=12,
    )

    # Add title and adjust layout.
    plt.suptitle(f"Generalization: {experiment}", weight="bold", fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05)
    plt.subplots_adjust(hspace=0.3)

    # Save the plot.
    plt.savefig(
        f"{output_dir}/Generalization_Individual_{experiment.replace(" ", "_")}.png",
        dpi=300,
    )
    plt.close(fig)


# Plot individual generalization for experiment 1: simple conditioning.
dv_ge_indi_plot(
    data_["s1"], Nactrials[0], stimulus_levels["s1"], experiments[0], DV_dir
)
# Plot individual generalization for experiment 2: differential conditioning.
dv_ge_indi_plot(
    data_["s2"], Nactrials[1], stimulus_levels["s2"], experiments[1], DV_dir
)


# Perception.


# Overall Perception.
def dv_pe_gr_plot(data_i, experiment, output_dir):
    """
    Visualizes overall perceived sizes across different stimuli for a specific
    experiment, displaying variations in perception through violin and scatter plots.

    This function sorts and visualizes the perceived size data for each stimulus,
    providing an aggregate view that helps identify trends and outliers in the
    perception of stimulus size. The plot incorporates violin plots to show
    distribution shapes and scatter plots to pinpoint individual data points,
    offering a detailed graphical representation of overall perception within
    the experimental context.

    Args:
        data_i (DataFrame):
            Long format DataFrame containing the data of the experiment.
        experiment (str):
            Name of the experiment to handle, affects plot title.
        output_dir (str):
            Directory where the plots will be saved.
    """
    # Sort data by stimulus.
    data_i["stimulus_phy"] = data_i["stimulus_phy"].astype(str)
    data_i["sorted_index"] = data_i["stimulus_phy"].str.extract(r"(\d+)").astype(int)
    data_i.sort_values("sorted_index", inplace=True)
    data_i["stimulus_phy"] = pd.Categorical(
        data_i["stimulus_phy"], categories=data_i["stimulus_phy"].unique(), ordered=True
    )

    # Create the figure.
    plt.figure(figsize=(9, 6))

    # Generate violinplots and scatterplots to show overall perceived size for each stimuli.
    sns.violinplot(
        x="stimulus_phy",
        y="Per_size",
        data=data_i,
        color="grey",
        density_norm="width",
        inner=None,
        cut=0,
        bw_method=0.2,
    )
    sns.stripplot(
        x="stimulus_phy",
        y="Per_size",
        data=data_i,
        color="black",
        size=4,
        jitter=True,
        alpha=0.2,
    )

    # Add title.
    plt.title(f"{experiment}", weight="bold", fontsize=14)

    # Set plot parameters.
    plt.xlabel("Stimulus", weight="bold", fontsize=12)
    plt.ylabel("Perceived Size", weight="bold", fontsize=12)
    plt.ylim(0, 200)
    plt.yticks(range(0, 201, 40))
    sns.despine(trim=True)

    # Save the plot.
    plt.savefig(
        f"{output_dir}/Perception_Overall_{experiment.replace(" ", "_")}.png", dpi=300
    )
    plt.close()


# Plot overall perception for experiment 1: simple conditioning.
dv_pe_gr_plot(data_["s1"], experiments[0], DV_dir)
# Plot overall perception for experiment 2: differential conditioning.
dv_pe_gr_plot(data_["s2"], experiments[1], DV_dir)


# Individual Perception.
def dv_pe_indi_plot(data_i, experiment, output_dir):
    """
    Visualizes individual perception responses across different stimuli for
    a specific experiment, showing variations in perceived size estimation.

    This function helps visualizing participant perceived sizes data for each stimulus
    on an individual basis using box plots and violin plots and presents these alongside
    the actual stimuli physical sizes, allowing direct comparison between perceived
    and physical dimensions. This detailed visualization helps to highlight individual
    differences in perception across the stimuli levels.

    Args:
        data_i (DataFrame):
            Long format DataFrame containing the data of the experiment.
        experiment (str):
            Name of the experiment to handle, affects plot title.
        output_dir (str):
            Directory where the plots will be saved.
    """
    # Sort data by stimulus.
    data_i["stimulus_phy"] = data_i["stimulus_phy"].astype(str)
    data_i["sorted_index"] = data_i["stimulus_phy"].str.extract(r"(\d+)").astype(int)
    data_i.sort_values("sorted_index", inplace=True)
    data_i["stimulus_phy"] = pd.Categorical(
        data_i["stimulus_phy"], categories=data_i["stimulus_phy"].unique(), ordered=True
    )

    # Create the figure and axes.
    participants = data_i["participant"].sort_values().unique()
    num_participants = len(participants)
    cols = 5
    rows = (num_participants // 5) + (num_participants % 5 > 0)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 18), sharex=True, sharey=True)
    axes = axes.flatten()

    # Iterate through the participants.
    for i, participant in enumerate(participants):
        ax = axes[i]
        participant_data = data_i[data_i["participant"] == participant]

        # Generate boxplots and violinplots to show the variation in perceived size
        # of the stimuli for the current participant.
        sns.boxplot(
            x="stimulus_phy",
            y="Per_size",
            data=participant_data,
            width=0.2,
            color="black",
            ax=ax,
            fliersize=0,
            zorder=1,
        )
        sns.violinplot(
            x="stimulus_phy",
            y="Per_size",
            data=participant_data,
            width=0.67,
            inner=None,
            cut=0,
            bw_method=0.2,
            ax=ax,
            color="gray",
            saturation=0.7,
            zorder=0,
        )

        # Generate a lineplot for the stimuli actual physical size.
        sns.lineplot(
            x="stimulus_phy",
            y="Phy_size",
            data=participant_data,
            color="red",
            ax=ax,
            marker="o",
            zorder=2,
        )

        # Set subplot parameters.
        ax.set_title(f"Participant {participant}", weight="bold", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_ylim(0, 200)
        ax.set_yticks(np.arange(0, 201, 40))
        ax.set_facecolor("whitesmoke")

    # Create overall x-label and y-label for the whole figure.
    fig.text(
        0.5, 0.03, "Stimulus", ha="center", va="center", weight="bold", fontsize=12
    )
    fig.text(
        0.006,
        0.5,
        "Size Estimation",
        ha="center",
        va="center",
        rotation="vertical",
        weight="bold",
        fontsize=12,
    )

    legend_handles = []
    legend_handles.append(
        plt.Line2D(
            [],
            [],
            color="red",
            marker="o",
            label="Stimuli physical size",
            linestyle="-",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=1,
        fancybox=True,
        shadow=True,
        fontsize=12,
    )

    # Add title and adjust layout.
    plt.suptitle(f"{experiment}", weight="bold", fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05)
    plt.subplots_adjust(hspace=0.3)

    # Save the plot.
    plt.savefig(
        f"{output_dir}/Perception_Individual_{experiment.replace(" ", "_")}.png",
        dpi=300,
    )
    plt.close(fig)


# Plot individual perception for experiment 1: simple conditioning.
dv_pe_indi_plot(data_["s1"], experiments[0], DV_dir)
# Plot individual perception for experiment 2: differential conditioning.
dv_pe_indi_plot(data_["s2"], experiments[1], DV_dir)


# Correlation.
def dv_cor_plot(data_i, experiment, output_dir):
    """
    Visualizes the correlation between physical and perceived sizes across participants
    for a specific experiment, displaying scatter plots and regression lines for each.

    This function processes the participant data to plot the relationship between the
    physical size of stimuli and their perceived size. It calculates the Pearson
    correlation coefficient for each participant and annotates this on the respective
    subplot. The function aims to provide an individualized analysis of perception
    accuracy across the experimental dataset, highlighting the consistency of
    participant responses in relation to actual stimulus sizes.

    Args:
        data_i (DataFrame):
            Long format DataFrame containing the data of the experiment.
        experiment (str):
            Name of the experiment to handle, affects plot title.
        output_dir (str):
            Directory where the plots will be saved.
    """
    # Ensure that columns needed for plotting are numeric.
    data_i["Phy_size"] = pd.to_numeric(data_i["Phy_size"], errors="coerce")
    data_i["Per_size"] = pd.to_numeric(data_i["Per_size"], errors="coerce")

    # Create figure.
    participants = data_i["participant"].sort_values().unique()
    num_participants = len(participants)
    cols = 5
    rows = (num_participants // cols) + (num_participants % cols > 0)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 18), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    # Find Min and Max for x-axis.
    min_x = data_i["Phy_size"].min()
    max_x = data_i["Phy_size"].max()

    # Iterate through each participant.
    for i, participant in enumerate(participants):
        ax = axes.flatten()[i]
        participant_data = data_i[data_i["participant"] == participant]

        # Generate scatterplot to show size estimations for each stimuli for
        # the current participant.
        sns.scatterplot(x="Phy_size", y="Per_size", data=participant_data, ax=ax)
        # Generate regplot to show size estimations trend for the stimuli for
        # the current participant.
        sns.regplot(
            x="Phy_size",
            y="Per_size",
            data=participant_data,
            ax=ax,
            scatter=False,
            color="blue",
        )

        # Calculate and annotate Pearson correlation coefficient between
        # physical and peceived sizes for the current participant.
        if (
            not participant_data["Phy_size"].isna().all()
            and not participant_data["Per_size"].isna().all()
        ):
            corr_coef, _ = pearsonr(
                participant_data.dropna(subset=["Phy_size", "Per_size"])["Phy_size"],
                participant_data.dropna(subset=["Phy_size", "Per_size"])["Per_size"],
            )
            ax.annotate(
                f"Pearson R = {corr_coef:.2f}",
                xy=(0.05, 0.95),
                xycoords="axes fraction",
                fontsize=10,
                verticalalignment="top",
            )

        # Set subplot parameters.
        ax.set_xlim([min_x - 5, max_x + 5])
        ax.set_ylim([-10, 210])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(f"Participant {participant}", weight="bold", fontsize=10)
        ax.set_facecolor("whitesmoke")

    # Add overall labels and title.
    fig.text(
        0.5, 0.03, "Stimulus", ha="center", va="center", weight="bold", fontsize=12
    )
    fig.text(
        0.006,
        0.5,
        "Size Estimation",
        ha="center",
        va="center",
        rotation="vertical",
        weight="bold",
        fontsize=12,
    )
    plt.suptitle(f"{experiment}", weight="bold", fontsize=12)

    # Adjust layout.
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05)
    plt.subplots_adjust(hspace=0.3)

    # Save plot.
    plt.savefig(f"{output_dir}/Correlation_{experiment.replace(" ", "_")}.png", dpi=300)
    plt.close(fig)


# Plot individual correlation for experiment 1: simple conditioning.
dv_cor_plot(data_["s1"], experiments[0], DV_dir)
# Plot individual correlation for experiment 2: differential conditioning.
dv_cor_plot(data_["s2"], experiments[1], DV_dir)
