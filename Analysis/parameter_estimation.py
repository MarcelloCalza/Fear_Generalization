"""
Module: parameter_estimation.py

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
conversion of the original R code to Python.
It facilitates parameter estimation from Bayesian modeling results, utilizing
quantiles for individual comparisons and violin plots for group comparisons
to assess model fits and parameter behaviors across different experimental conditions.

The key functionalities of this module include:
- Extracting and organizing parameter data from Bayesian modeling results.
- Employing plotting techniques from libraries such as Matplotlib and Seaborn
  to visualize individual-level and group-level parameter estimates to highlight learning
  and generalization rates behaviours.

### Functions:
- pe_data_indi: Prepares parameter data for individual-level visualization, calculating
  quantiles for key model parameters for single studies.
- pe_plot_indi: Generates visualizations for individual-level parameter estimates,
  facilitating detailed analysis of model performance on a per-participant basis for
  single studies.
- pe_plot_indi_both: Orchestrate the plotting of both studies together
  for individual-level parameter estimation, enabling comprehensive
  visualization of model performance across different study conditions.
- pe_data_gr: Collects and organizes data for parameters like alpha_mu,
  lambda_mu, and group-level pi across studies.
- pe_plot_gr: Depicts parameter data organized by pe_data_gr using violin plots,
  offering a clear view of the parameters distributions in both studies.
"""

import arviz as az
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import numpy as np
import pandas as pd
import seaborn as sns
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

# Define plotting parameters.
experiments = [
    "Experiment 1: Simple Conditioning",
    "Experiment 2: Differential Conditioning",
]
color_groups = ["#CC79A7", "#F0E442", "#56B4E9", "#D55E00", "black"]

# Extract group indicators samples of both studies from the posterior distributions.
gp_samples = [
    results["s1"]["CLG2D"].posterior["gp"],
    results["s2"]["CLG2D"].posterior["gp"],
]

# Create a directory to save parameter estimation plots.
PE_dir = "../Plots/Parameter_Estimation/"
os.makedirs(PE_dir, exist_ok=True)


# Individual PE.
def pe_data_indi(exp_index, results, gp_processed):
    """
    Prepares and organizes data for parameter estimation plots by participant and group
    for a single study.

    This function extracts and calculates quantiles for key model parameters
    learning and generalization rates from the posterior distribution. It merges these
    data with group allocation results to facilitate grouped visual analysis.

    Args:
        exp_index (int):
            Index indicating which experiment's data to process.
        results (dict):
            InferenceData objects for each study with posterior distributions.
        gp_processed (DataFrame):
            Processed group information data per participant.

    Returns:
        df_final (DataFrame):
            DataFrame with parameters quantiles and group information for plotting.
    """
    # Retrieve learning and generalization rates data from the posterior distribution.
    alpha_data = results[f"s{exp_index}"]["CLG2D"].posterior["alpha"]
    lambda_data = results[f"s{exp_index}"]["CLG2D"].posterior["lambda"]

    # Initialize DataFrame list and quantiles percentages.
    df_list = []
    quantiles = [2.5, 25, 50, 75, 97.5]

    # Compute quantiles for each parameter and add them to the DataFrame list.
    for param_data, param_name in zip([alpha_data, lambda_data], ["alpha", "lambda"]):
        data = {}
        for q in quantiles:
            data[f"q{q}"] = np.nanpercentile(param_data, q, axis=0)
        data["Participant_Num"] = np.arange(1, param_data.shape[1] + 1)
        data["Parameter"] = param_name
        df = pd.DataFrame(data)
        df_list.append(df)

    df_final = pd.concat(df_list)

    # Merge the quantiles DataFrame with the group allocation DataFrame.
    df_final = df_final.merge(gp_processed, on="Participant_Num")

    # Get each participant's median lambda point.
    lambda_meds = df_final[df_final["Parameter"] == "lambda"][
        ["Participant_Num", "q50"]
    ].rename(columns={"q50": "lambda_q50"})

    # Merge so alpha rows also have nedian lambda point for plotting.
    df_final = df_final.merge(lambda_meds, on="Participant_Num", how="left")

    return df_final


def pe_plot_indi(df, axes, color_groups):
    """
    Plots parameter estimation results across participants grouped by their assigned
    latent groups for a single study.

    This function visualizes the median and confidence intervals of the learning and
    generalization rates for each participant, colored by their most likely group.
    It supports visual comparison across different groups and participants.

    Args:
        df (DataFrame):
            DataFrame with parameters quantiles and group information for plotting.
        axes (array of AxesSubplot):
            Array of matplotlib axes to plot on.
        color_groups (list):
            List of colors for each group name for plotting.
    """
    # Create color mapping based on groups.
    cat_order = df["Group_Name"].cat.categories
    color_palette = dict(zip(cat_order, color_groups))

    # Ietrate through parameters.
    for param, ax, title in zip(
        ["alpha", "lambda"], axes, [": Learning rate", ": Generalization rate"]
    ):
        param_df = df[df["Parameter"] == param].copy()

        # Sort by group then ascending lambda median.
        param_df = param_df.sort_values(
            by=["Group_Name", "lambda_q50"], ascending=[True, True]
        )

        # Map participants to y-values.
        part_ids = param_df["Participant_Num"].unique()
        part_to_y = {p: i for i, p in enumerate(part_ids)}
        param_df["y_pos"] = param_df["Participant_Num"].map(part_to_y)

        # Plot intervals.
        for _, row in param_df.iterrows():
            y_val = part_to_y[row["Participant_Num"]]
            group_name = row["Group_Name"]

            # Get the color for that group.
            color = color_palette.get(group_name, "gray")

            # Plot 95% CI interval.
            ax.hlines(
                y=y_val,
                xmin=row["q2.5"],
                xmax=row["q97.5"],
                color=color,
                linewidth=1.0,
                zorder=1,
            )

            # Plot 50% CI interval.
            ax.hlines(
                y=y_val,
                xmin=row["q25"],
                xmax=row["q75"],
                color="black",
                linewidth=3.0,
                zorder=2,
            )

        # Plot median point.
        sns.scatterplot(
            data=param_df,
            x="q50",
            y="y_pos",
            hue="Group_Name",
            style="Group_Name",
            palette=color_palette,
            markers="o",
            edgecolor="black",
            ax=ax,
            s=80,
            zorder=3,
        )

        # Set subplot parameters.
        ax.set_yticks([part_to_y[p] for p in part_ids])
        ax.set_yticklabels(part_ids)
        ax.set_facecolor("whitesmoke")
        ax.set_xlabel(f"$\\mathbf{{\\{param}}}${title}", weight="bold", fontsize=12)
        ax.set_ylabel("Participant", weight="bold", fontsize=12)
        if ax.get_legend():
            ax.get_legend().remove()


def pe_plot_indi_both(studies_data_list, experiments, color_groups, output_dir):
    """
    Organizes and executes the plotting of parameter estimations for both studies.

    This function manages subplots for each parameter across both studies,
    calling the function `pe_plot_indi` to fill each subplot with
    appropriate visualization based on the provided data.

    Args:
        studies_data_list (list of DataFrames):
            List of DataFrames for each studies, containing parameters quantiles
            and group information for plotting.
        experiments (list):
            List of the experiments names connected to the studies.
        color_groups (list):
            List of colors for each group name for plotting.
        output_dir (str):
            Directory where the plots will be saved.
    """
    fig, axes = plt.subplots(1, 4, figsize=(24, 8))

    # Generate plots and title for Experiment 1: simple conditioning.
    pe_plot_indi(studies_data_list[0], [axes[0], axes[1]], color_groups)
    axes[0].set_title(f"{experiments[0]} (N=40)", weight="bold", fontsize=12)

    # Generate plots and title for Experiment 1: differential conditioning.
    pe_plot_indi(studies_data_list[1], [axes[2], axes[3]], color_groups)
    axes[2].set_title(f"{experiments[1]} (N=40)", weight="bold", fontsize=12)

    # Adjust spacing.
    fig.subplots_adjust(wspace=0.2, bottom=0.2)

    # Create overall legend.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=6,
        fancybox=True,
        shadow=True,
        fontsize=12,
    )

    fig.savefig(f"{output_dir}/Parameter_Estimation_Individual.png", dpi=300)


# Prepare individual parameter estimation data for study 1.
study1_data = pe_data_indi(1, results, gp_allocation(gp_samples[0]))
# Prepare individual parameter estimation data for study 2.
study2_data = pe_data_indi(2, results, gp_allocation(gp_samples[1]))
# Plot individual parameter estimation for both studies.
pe_plot_indi_both([study1_data, study2_data], experiments, color_groups, PE_dir)


# Group PE.
def pe_data_gr(results):
    """
    Prepares combined data for alpha_mu, lambda_mu, and group-level pi
    parameters across studies for parameter estimation plots.

    This function collects and structures the mean alpha and lambda values, as well
    as the pi parameters from posterior distributions. It then organizes these into
    DataFrames suitable for visual representation in estimation plots.

    Args:
        results (dict):
            InferenceData objects for each study with posterior distributions.

    Returns:
        alpha_lambda_data, pi_data (tuple):
            Contains two DataFrames, one for alpha_mu and lambda_mu parameters, and one
            for pi parameters, each indexed by study.
    """
    # Initialize dictionaries for data collection.
    alpha_lambda_data = {"alpha_mu": [], "lambda_mu": [], "study": []}
    pi_data = {"p1": [], "p2": [], "p3": [], "p4": [], "study": []}

    # Iterate through studies results.
    for i, exp_key in enumerate(["s1", "s2"], 1):

        # Retrieve current study posterior distribution.
        exp_data = results[exp_key]["CLG2D"]

        # Collect alpha_mu and lambda_mu parameters values.
        alpha_lambda_data["alpha_mu"].extend(exp_data.posterior["alpha_mu"])
        alpha_lambda_data["lambda_mu"].extend(exp_data.posterior["lambda_mu"])
        alpha_lambda_data["study"].extend(
            [f"Exp.{i}"] * len(exp_data.posterior["alpha_mu"])
        )

        # Collect pi parameters values.
        for j in range(4):
            pi_data[f"p{j+1}"].extend(exp_data.posterior["pi"][:, j])
        pi_data["study"].extend([f"Exp.{i}"] * exp_data.posterior["pi"].shape[0])

    return pd.DataFrame(alpha_lambda_data), pd.DataFrame(pi_data)


def pe_plot_gr(alpha_lambda_data, pi_data, color_groups, output_dir):
    """
    Plots parameter estimation results for alpha_mu, lambda_mu, and group-level pi
    parameters from prepared data across studies.

    This function uses violin plots to depict the distribution and individual data points
    of alpha_mu, lambda_mu, and group-level pi parameters for each study, helping to
    visualize the overall variability and central tendency of these parameters.

    Args:
        alpha_lambda_data (DataFrame):
            DataFrame containing alpha_mu and lambda_mu data, segmented by study.
        pi_data (DataFrame):
            DataFrame containing pi parameter data, segmented by study.
        color_groups (list):
            List of colors for each group name for plotting.
        output_dir (str):
            Directory where the plots will be saved.
    """
    # Create the figure.
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 4, height_ratios=[1, 2])

    # Initialize alpha_mu and lambda_mu subplots axes.
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2:])

    # Iterate through axes, parameters names and titles for alpha_mu and lambda_mu.
    for ax, param, title in zip(
        [ax1, ax2],
        ["alpha_mu", "lambda_mu"],
        [
            "$\\mathbf{{\\mu_\\alpha}}$: Mean of learning rates (N = 40)",
            "$\\mathbf{{\\mu_\\lambda}}$: Mean of generalization rates (N = 40)",
        ],
    ):

        # Generate violinplots for the current parameter for both studies.
        sns.violinplot(
            y="study",
            x=param,
            data=alpha_lambda_data,
            ax=ax,
            density_norm="width",
            orient="h",
        )

        # Set subplot parameters.
        ax.set_title(weight="bold", label=title, fontsize=12)
        ax.set_xlabel(
            (
                f"$\\mathbf{{\\mu_\\alpha}}$"
                if "alpha_mu" in param
                else f"$\\mathbf{{\\mu_\\lambda}}$"
            ),
            fontsize=12,
        )
        ax.set_ylabel("")
        ax.set_xlim((0, 0.6))
        ax.set_facecolor("whitesmoke")

    # Iterate through possible pi values.
    for i in range(4):

        # Initialize current pi value subplot ax.
        ax = fig.add_subplot(gs[1, i])

        # Generate violinplots for the current pi value for both studies.
        sns.violinplot(
            y="study",
            hue="study",
            x=f"p{i+1}",
            data=pi_data,
            ax=ax,
            density_norm="width",
            orient="h",
            palette=[color_groups[i]],
            legend=False,
        )

        # Set subplot parameters.
        ax.set_title(weight="bold", label=f"Probability m={i+1} (N = 40)", fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylabel("")
        ax.set_xlabel(f"$\\mathbf{{\\pi_{{ {i+1} }}}}$", fontsize=12)
        ax.set_facecolor("whitesmoke")

    plt.tight_layout()
    fig.savefig(f"{output_dir}/Parameter_Estimation_Group", dpi=300)


# Prepare group parameter estimation data for both studies.
alpha_lambda_data, pi_data = pe_data_gr(results)
# Plot group parameter estimation for both studies.
pe_plot_gr(alpha_lambda_data, pi_data, color_groups, PE_dir)
