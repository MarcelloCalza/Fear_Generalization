"""
Module: group_allocation.py

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
conversion of the original R code to Python, and contains functionalities
to process Bayesian modeling results.
It allocates participants into groups based on the highest likelihood
group, and plot group allocations.

The key features of this module include:
- Loading and handling Bayesian modeling results stored in NetCDF format.
- Grouping participants based on posterior distributions of latent group
  indicators.
- Generating bar plots to visually represent group allocation proportions
  and comparing them against a defined threshold.

### Functions:
- gp_allocation: Analyzes and assigns participants to groups based on the
  dominance of group indicators.
- plot_gp_allocation: Creates and saves bar plots that visualize the
  proportion of participants in each group for given study data.
"""

import arviz as az
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from data_utils import ReshapedInferenceData


def gp_allocation(sample):
    """
    Processes latent group indicators samples from posterior distribution to allocate
    participants into groups based on the highest likelihood group for a single study.

    This function computes the proportion of samples in each group for each
    participant and assigns a dominant group if the proportion exceeds the 75%
    threshold. The output includes each participant's dominant group name and the
    proportions for every group.

    Args:
        sample (ndarray):
            Latent group indicators samples from a posterior distribution.
            Shape: (num_samples, num_participants)

    Returns:
        df_long (DataFrame):
            Columns:
            - Participant_Num (int): the participant number.
            - Group (str): The group name.
            - Proportion (float): The proportion for the corresponding group.
            - Group_Name (str): The name of the participant's dominant group.
    """
    # Create a DataFrame (long format).
    df = pd.DataFrame(
        sample, columns=[f"Participant_{i+1}" for i in range(sample.shape[1])]
    )
    df = df.melt(var_name="Participant", value_name="Group")

    # Extract and convert participant numbers to integers.
    df["Participant_Num"] = df["Participant"].str.extract(r"(\d+)").astype(int)

    # Group by Participant_Num and Group, then count.
    df_count = df.groupby(["Participant_Num", "Group"]).size().reset_index(name="Count")

    # Calculate the total number of samples per participant.
    total_counts = df_count.groupby("Participant_Num")["Count"].transform("sum")

    # Compute proportion for each group within a participant.
    df_count["Proportion"] = df_count["Count"] / total_counts

    # Find the group with maximum proportion for each participant.
    df_max = df_count.loc[
        df_count.groupby("Participant_Num")["Proportion"].idxmax()
    ].copy()

    # Flag the row as threshold-met if Proportion > 0.75.
    df_max["Threshold_Met"] = df_max["Proportion"] > 0.75

    # Map numeric labels to descriptive labels.
    group_map = {
        0: "Non-Learners",
        1: "Overgeneralizers",
        2: "Physical Generalizers",
        3: "Perceptual Generalizers",
    }
    df_max["Max_Group"] = df_max["Group"].map(group_map)

    # Assign dominant group based on the 75% threshold.
    df_max["Group_Name"] = df_max.apply(
        lambda x: x["Max_Group"] if x["Threshold_Met"] else "Unknown", axis=1
    )

    # Merge the dominant group back onto all group-rows.
    df_merged = pd.merge(
        df_count,
        df_max[["Participant_Num", "Group_Name"]],
        on="Participant_Num",
        how="left",
    )

    # Replace the numeric labels with the groups names.
    df_merged["Group"] = df_merged["Group"].map(group_map)

    # Create final dataframe.
    df_long = df_merged.loc[
        :, ["Participant_Num", "Group", "Proportion", "Group_Name"]
    ].copy()

    # Convert dominant group column into categorical and order.
    group_order = [
        "Non-Learners",
        "Overgeneralizers",
        "Physical Generalizers",
        "Perceptual Generalizers",
        "Unknown",
    ]

    df_long["Group_Name"] = pd.Categorical(
        df_long["Group_Name"], categories=group_order, ordered=True
    )

    return df_long


# Plot group allocation.
def plot_gp_allocation(df, title, group_names, color_groups, output_dir):
    """
    Plots the group allocation proportions for a given study.

    This function uses seaborn to create a bar plot visualizing the proportions of each
    participant's groups. A horizontal line indicates the 75% threshold for
    dominant group allocation.

    Args:
        df (DataFrame):
            DataFrame containing participant numbers, group names, and proportions.
        title (str):
            The title (for titling the plot).
        group_names (list):
            List of all possible groups names.
        color_groups (list):
            List of colors for each group name for plotting.
        output_dir (str):
            Directory where the plot will be saved.
    """
    # Create color mapping based on groups.
    color_mapping = {group: color for group, color in zip(group_names, color_groups)}

    # Create the figure and generate the bar plot.
    fig = plt.figure(figsize=(20, 8))
    ax = sns.barplot(
        data=df,
        x="Participant_Num",
        y="Proportion",
        hue="Group",
        palette=color_mapping,
    )

    # Get plot ax.
    ax = plt.gca()

    # Set plot parameters.
    plt.title(f"Group Allocation - {title}", weight="bold", fontsize=14)
    plt.xlabel("Participant Number", weight="bold", fontsize=12)
    plt.ylabel("Proportion", weight="bold", fontsize=12)
    plt.axhline(0.75, color="red", linestyle="--")
    ax.set_facecolor("whitesmoke")

    # Reorder legend to match the group_names order.
    handles, labels = ax.get_legend_handles_labels()
    label_handle_dict = dict(zip(labels, handles))
    ordered_handles = [
        label_handle_dict[label] for label in group_names if label in label_handle_dict
    ]
    ordered_labels = [label for label in group_names if label in label_handle_dict]

    # Add custom handle and label for the threshold line.
    threshold_line = plt.Line2D(
        [], [], color="red", linestyle="--", label="Threshold (75%)"
    )
    ordered_handles.append(threshold_line)
    ordered_labels.append("Threshold (75%)")

    # Set legend.
    ax.legend(
        handles=ordered_handles,
        labels=ordered_labels,
        title="Groups",
        ncol=5,
        bbox_to_anchor=(0.5, 0),
        loc="lower center",
        fancybox=True,
        shadow=True,
        fontsize=12,
    )

    plt.tight_layout()
    fig.savefig(f"{output_dir}/Group_Allocation_{title.replace(" ", "_")}.png", dpi=300)


if __name__ == "__main__":

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

    # Define plotting parameters.
    experiments = [
        "Experiment 1 Simple Conditioning",
        "Experiment 2 Differential Conditioning",
    ]
    group_names = [
        "Non-Learners",
        "Overgeneralizers",
        "Physical Generalizers",
        "Perceptual Generalizers",
        "Unknown",
    ]
    color_groups = ["#CC79A7", "#F0E442", "#56B4E9", "#D55E00", "black"]

    # Extract group indicators samples of both studies from the posterior distributions.
    gp_samples = [
        results["s1"]["CLG2D"].posterior["gp"],
        results["s2"]["CLG2D"].posterior["gp"],
    ]

    # Create a directory to save group allocation plots.
    GA_dir = "../Plots/Group_Allocation/"
    os.makedirs(GA_dir, exist_ok=True)

    # Generate group allocation plot for study 1.
    plot_gp_allocation(
        gp_allocation(gp_samples[0]), experiments[0], group_names, color_groups, GA_dir
    )
    # Generate group allocation plot for study 2.
    plot_gp_allocation(
        gp_allocation(gp_samples[1]), experiments[1], group_names, color_groups, GA_dir
    )
