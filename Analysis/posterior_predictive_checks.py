"""
Module: plot_grouped_data.py

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
conversion of the original PPCs R code to Python. It includes functionalities
to process Bayesian modeling results and visualize posterior predictive checks
in various detailed ways, enhancing the accessibility and interpretability
of the results through comprehensive plots.

The key functionalities of this module include:
- Loading Bayesian modeling results stored in NetCDF format.
- Utilizing plotting features from Matplotlib and Seaborn to compare model-based
  predictions with observed data across different experimental conditions for
  posterior predictive checks.

### Functions:
- ppc_plot: Plots quantiles and means of predicted
  and observed data by stimulus, facilitating posterior predictive checks for
  single studies.
- ppc_plot_gr: Plots quantiles of predicted and observed data
  by stimulus grouped by participants dominant groups to assess model fit across
  different groups for single studies.
- ppc_plot_both/ppc_plot_gr_both: Orchestrate the plotting of both studies together
  for general posterior predictive checks/group posterior predictive checks,
  enabling a comprehensive view of model performance across different study conditions.
- ppc_plot_indi: Plots detailed predictive checks for each participant,
  showcasing how well the model's predictions align with the actual data on an
  individual level for single studies.
"""

import arviz as az
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
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
# Load PyMC input dictionaries.
PYMCinput = {
    "s1": pickle.load(open("../PYMC_input_data/Data1_PYMCinput_CLG.pkl", "rb")),
    "s2": pickle.load(open("../PYMC_input_data/Data2_PYMCinput_CLG.pkl", "rb")),
}

# Define plotting parameters.
experiments = [
    "Experiment 1 Simple Conditioning",
    "Experiment 2 Differential Conditioning",
]
color_groups = {
    "Non-Learners": "#CC79A7",
    "Overgeneralizers": "#F0E442",
    "Physical Generalizers": "#56B4E9",
    "Perceptual Generalizers": "#D55E00",
}
stimulus_levels = {
    "s1": ["S4", "S5", "S6", "CS+", "S8", "S9", "S10"],
    "s2": ["CS+", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "CS-"],
}

# Extract group indicators samples of both studies from the posterior distributions.
gp_samples = [
    results["s1"]["CLG2D"].posterior["gp"],
    results["s2"]["CLG2D"].posterior["gp"],
]

# Extract posterior predictive distibution data of both studies.
y_pred_study1 = results["s1"]["CLG2D"].posterior_predictive["y_pre"]
y_pred_study2 = results["s2"]["CLG2D"].posterior_predictive["y_pre"]

# Extract observed data of both studies.
y_actual_study1 = PYMCinput["s1"]["y"][:, 14:188]
y_actual_study2 = PYMCinput["s2"]["y"][:, 24:180]

# Extract stimulus column data for both experiments and convert it to wide format.
stimulus_s1 = pd.pivot_table(
    data_["s1"],
    index="participant",
    columns="trials",
    values="stimulus",
    aggfunc="first",
)
stimulus_s2 = pd.pivot_table(
    data_["s2"],
    index="participant",
    columns="trials",
    values="stimulus",
    aggfunc="first",
)
stimulus_s1.columns = stimulus_s1.columns.astype(str)
stimulus_s2.columns = stimulus_s2.columns.astype(str)

# Select stimulus data for the generalization trials.
stimulus_s1 = stimulus_s1.iloc[:, 14:188]
stimulus_s2 = stimulus_s2.iloc[:, 24:180]

# Create a directory to save posterior predictive checks plots.
PPC_dir = "../Plots/Posterior_Predictive_Checks/"
os.makedirs(PPC_dir, exist_ok=True)


# General PPC.
def ppc_plot(y_pred, y_actual, stimulus, stimulus_level, axes, num_samples=10000):
    """
    Plots quantiles and means of predicted and observed data by stimulus for posterior
    predictive checks for a single study.

    This function samples from the posterior predictive distribution to generate plots
    showing the model's performance across different quantiles and the mean for a
    specific study. It aids in comparing the model's predictions against actual
    observed data for all the stimuli.

    Args:
        y_pred (ndarray):
            Predicted response data from the posterior predictive distribution.
        y_actual (ndarray):
            Actual observed response data.
        stimulus (DataFrame):
            Stimulus information data per trial.
        stimulus_level (list):
            Ordered list of stimuli.
        axes (matplotlib.axes._subplots.AxesSubplot):
            Array of plot axes.
        num_samples (int, optional):
            Number of samples to take from the predicted data for plotting.

    """

    # Sample predicted data.
    indices = np.random.choice(y_pred.shape[0], num_samples, replace=False)
    y_pred_sampled = y_pred[indices]

    # Initialize mean and quantiles pairs list and a list for data to plot.
    quantile_pairs = [("Mean", 10), (30, 50), (70, 90)]
    plot_data = []

    # Iterate over the pairs, compute quantiles/means for predicted and observed data.
    for ax_idx, (q1, q2) in enumerate(quantile_pairs):
        for q in [q1, q2]:
            plot_data = []
            for stim_label in stimulus_level:
                stimulus_indices = stimulus == stim_label
                if not stimulus_indices.any().any():
                    continue
                if q == "Mean":
                    pred_quantile = np.nanmean(y_pred_sampled[:, stimulus_indices])
                    act_quantile = np.nanmean(y_actual[stimulus_indices])
                else:
                    pred_quantile = np.nanpercentile(
                        y_pred_sampled[:, stimulus_indices], q
                    )
                    act_quantile = np.nanpercentile(y_actual[stimulus_indices], q)

                # Add the computed data to the data to plot.
                plot_data.append(
                    {
                        "Stimulus": stim_label,
                        "Value": pred_quantile,
                        "Type": "Model-based prediction",
                    }
                )
                plot_data.append(
                    {
                        "Stimulus": stim_label,
                        "Value": act_quantile,
                        "Type": "Observed data",
                    }
                )

            # Create DataFrame for plotting.
            result_df = pd.DataFrame(plot_data)
            result_df["Stimulus"] = pd.Categorical(
                result_df["Stimulus"], categories=stimulus_level, ordered=True
            )

            idx = ax_idx * 2 if q == q1 else ax_idx * 2 + 1

            # Generate lineplots for current quantile or mean for predicted and observed data.
            sns.lineplot(
                data=result_df,
                x="Stimulus",
                y="Value",
                hue="Type",
                style="Type",
                markers="o",
                ax=axes[idx],
                dashes=False,
                palette={"Model-based prediction": "blue", "Observed data": "black"},
            )

            # Set subplot parameters.
            axes[idx].get_legend().remove()
            axes[idx].set_title(
                "Mean" if q == "Mean" else f"Quantile {q}%", weight="bold", fontsize=12
            )
            axes[idx].set_ylabel(
                "US Expectancy (observed data scale: 1-10)", weight="bold", fontsize=10
            )
            axes[idx].set_xlabel("Stimulus", weight="bold", fontsize=10)
            axes[idx].set_ylim(-1.5, 11.5)
            axes[idx].set_yticks(np.arange(1, 11, 1))
            axes[idx].set_facecolor("whitesmoke")


def ppc_plot_both(
    y_pred1,
    y_actual1,
    stimulus1,
    y_pred2,
    y_actual2,
    stimulus2,
    experiments,
    stimulus_levels,
    output_dir,
):
    """
    Plots posterior predictive checks for both studies side by side.

    This function orchestrates the plotting of both experiments by arranging subplots
    and calling another function `ppc_plot` to fill in
    each subplot with appropriate data visualizations based on quantiles and means of
    model predictions versus observed data.

    Args:
        y_pred1 (ndarray):
            Posterior predictions for study 1.
        y_actual1 (ndarray):
            Observed data for study 1.
        stimulus1 (DataFrame):
            Stimulus data for study 1.
        y_pred2 (ndarray):
            Posterior predictions for study 2.
        y_actual2 (ndarray):
            Observed data for study 2.
        stimulus2 (DataFrame):
            Stimulus data for study 2.
        experiments (list):
            List of the experiments names connected to the studies.
        stimulus_levels (dict):
            Specifies the order of stimuli for each study.
        output_dir (str):
            Directory where the plots will be saved.
    """

    # Create figure.
    fig, axes = plt.subplots(3, 4, figsize=(20, 15), sharex="col", sharey="row")

    # Create subplots for study 1.
    ppc_plot(
        y_pred1,
        y_actual1,
        stimulus1,
        stimulus_levels["s1"],
        axes[:, 0:2].flatten(),
        num_samples=1000,
    )

    # Create subplots for study 2.
    ppc_plot(
        y_pred2,
        y_actual2,
        stimulus2,
        stimulus_levels["s2"],
        axes[:, 2:4].flatten(),
        num_samples=1000,
    )

    # Create overall legend.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=2,
        fancybox=True,
        shadow=True,
        fontsize=12,
    )

    fig.subplots_adjust(top=0.88)
    fig.suptitle("")

    # Add titles.
    fig.text(
        0.30, 0.92, experiments[0], ha="center", va="center", fontsize=14, weight="bold"
    )
    fig.text(
        0.71, 0.92, experiments[1], ha="center", va="center", fontsize=14, weight="bold"
    )

    fig.savefig(f"{output_dir}/Posterior_Predictive_Checks_General", dpi=300)


# Plot general posterior predictive checks for both studies.
ppc_plot_both(
    y_pred_study1,
    y_actual_study1,
    stimulus_s1,
    y_pred_study2,
    y_actual_study2,
    stimulus_s2,
    experiments,
    stimulus_levels,
    PPC_dir,
)


# Group PPC.
def ppc_plot_gr(
    y_pred, y_actual, stimulus, gp_processed, stimulus_level, axes, num_samples=10000
):
    """
    Prepares and plots data grouped by dominant groups for posterior predictive checks
    for a single study.

    This function processes the model's predictions and the observed data to evaluate
    the model's performance for different groups across various stimulus levels.
    It computes quantiles and plots them alongside the observed data to visually assess
    model fit per group.

    Args:
        y_pred (ndarray):
            Predicted response data from the posterior predictive distribution.
        y_actual (ndarray):
            Actual observed response data.
        stimulus (DataFrame):
            Stimulus information data per trial.
        gp_processed (DataFrame):
            Processed group information data per participant.
        stimulus_level (list):
            Ordered list of stimuli.
        axes (matplotlib.axes._subplots.AxesSubplot):
            Array of plot axes.
        num_samples (int, optional):
            Number of samples to take from the predicted data for plotting.
    """

    # Sample predicted data.
    indices = np.random.choice(y_pred.shape[0], num_samples, replace=False)
    y_pred_sampled = y_pred[indices]

    # Retrieve unique dominant groups.
    groups = gp_processed.sort_values(by="Group_Name")["Group_Name"].unique()

    # Iterate over each group for plotting
    for idx, group_name in enumerate(groups):
        group_mask = (
            gp_processed.drop_duplicates(subset="Participant_Num")["Group_Name"]
            == group_name
        ).values
    # Iterate over each group for plotting.
    for idx, group_name in enumerate(groups):
        ax = axes[idx]
        group_mask = (
            gp_processed.drop_duplicates(subset="Participant_Num")["Group_Name"]
            == group_name
        ).values
        # Initialize quantiles percentages list and a list for data to plot.
        quantiles = [10, 30, 50]
        plot_data = []

        # Compute quantiles for predicted and observed data for the current group.
        stimulus_grouped = stimulus.iloc[group_mask, :]
        y_pred_grouped = y_pred_sampled[:, group_mask, :]
        y_actual_grouped = y_actual[group_mask, :]
        for q in quantiles:
            for stim_label in stimulus_level:
                stimulus_indices = stimulus_grouped == stim_label
                if not stimulus_indices.any().any():
                    continue
                pred_quantile = np.nanpercentile(y_pred_grouped[:, stimulus_indices], q)
                act_quantile = np.nanpercentile(y_actual_grouped[stimulus_indices], q)

                # Add the computed data to the data to plot.
                plot_data.append(
                    {
                        "Stimulus": stim_label,
                        "Value": pred_quantile,
                        "Quantile": f"{q}%",
                        "Type": "Model-based prediction",
                    }
                )
                plot_data.append(
                    {
                        "Stimulus": stim_label,
                        "Value": act_quantile,
                        "Quantile": f"{q}%",
                        "Type": "Observed data",
                    }
                )

        if plot_data:

            # Create DataFrame for plotting.
            df_plot = pd.DataFrame(plot_data)
            df_plot["Stimulus"] = pd.Categorical(
                df_plot["Stimulus"], categories=stimulus_level, ordered=True
            )

            dark_palette = sns.color_palette("dark", n_colors=len(quantiles))

            # Generate lineplots for each quantile for the current group for predicted
            # and observed data.
            sns.lineplot(
                data=df_plot,
                x="Stimulus",
                y="Value",
                hue="Quantile",
                style="Type",
                markers={"Model-based prediction": "X", "Observed data": "o"},
                dashes={"Model-based prediction": (3, 3), "Observed data": ""},
                palette=dark_palette,
                ax=ax,
                errorbar=None,
            )

            # Set subplot parameters.
            ax.set_facecolor("whitesmoke")
            ax.get_legend().remove()
            ax.set_title(f"{group_name}", weight="bold", fontsize=12)
            ax.set_xlabel("Stimulus", weight="bold", fontsize=10)
            ax.set_ylabel(
                "US Expectancy (observed data scale: 1-10)", weight="bold", fontsize=10
            )
            ax.set_ylim(-1.5, 11.5)
            ax.set_yticks(np.arange(1, 11, 1))


def ppc_plot_gr_both(
    y_pred1,
    y_actual1,
    stimulus1,
    gp_processed1,
    y_pred2,
    y_actual2,
    stimulus2,
    gp_processed2,
    experiments,
    stimulus_levels,
    output_dir,
):
    """
    Plots posterior predictive checks for two both studies side by side.

    This function orchestrates the plotting of both studies by arranging subplots
    and calling another function `ppc_plot_gr` to fill in
    each subplot with appropriate data visualizations based on quantiles and means of
    model predictions versus observed data.

    Args:
        y_pred1 (ndarray):
            Posterior predictions for study 1.
        y_actual1 (ndarray):
            Observed data for study 1.
        stimulus1 (DataFrame):
            Stimulus data for study 1.
        gp_processed1 (DataFrame):
            Group information data per participant for study 1.
        y_pred2 (ndarray):
            Posterior predictions for study 2.
        y_actual2 (ndarray):
            Observed data for study 2.
        stimulus2 (DataFrame):
            Stimulus data for study 2.
        gp_processed2 (DataFrame):
            Group information data per participant for study 2.
        experiments (list):
            List of the experiments names connected to the studies.
        stimulus_levels (dict):
            Specifies the order of stimuli for each study.
        output_dir (str):
            Directory where the plots will be saved.
    """
    # Create figure.
    fig, axes = plt.subplots(2, 5, figsize=(26, 12), sharey=True)

    # Create subplots for study 1.
    ppc_plot_gr(
        y_pred1,
        y_actual1,
        stimulus1,
        gp_processed1,
        stimulus_levels["s1"],
        axes[0, :],
        num_samples=10000,
    )

    # Create subplots for study 2.
    ppc_plot_gr(
        y_pred2,
        y_actual2,
        stimulus2,
        gp_processed2,
        stimulus_levels["s2"],
        axes[1, :],
        num_samples=10000,
    )

    # Create overall legend.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        fontsize=14,
        fancybox=True,
        shadow=True,
        ncol=7,
    )

    plt.subplots_adjust(hspace=0.5, top=0.87)

    # Add titles.
    fig.text(
        0.5, 0.92, experiments[0], ha="center", va="center", fontsize=14, weight="bold"
    )
    fig.text(
        0.5, 0.47, experiments[1], ha="center", va="center", fontsize=14, weight="bold"
    )

    fig.savefig(f"{output_dir}/Posterior_Predictive_Checks_Group", dpi=300)


# Plot groups posterior predictive checks for both studies.
ppc_plot_gr_both(
    y_pred_study1,
    y_actual_study1,
    stimulus_s1,
    gp_allocation(gp_samples[0]),
    y_pred_study2,
    y_actual_study2,
    stimulus_s2,
    gp_allocation(gp_samples[1]),
    experiments,
    stimulus_levels,
    PPC_dir,
)


# Individual PPC.
def ppc_plot_indi(
    y_pred,
    y_actual,
    stimulus,
    gp_processed,
    stimulus_level,
    experiment,
    color_groups,
    output_dir,
    num_samples=10000,
):
    """
    Plots means of predicted and observed data by stimulus for posterior predictive
    checks for each participant for a single study.

    This function is designed to plot detailed comparisons at an individual level,
    showing how well the model's predictions match the observed data for each
    participant within the given stimulus context.

    Args:
        y_pred (ndarray):
            Predicted response data from the posterior predictive distribution.
        y_actual (ndarray):
            Actual observed response data.
        stimulus (DataFrame):
            Stimulus information data per trial.
        gp_processed (DataFrame):
            Processed group information data per participant.
        stimulus_level (list):
            Ordered list of stimuli.
        experiment (str):
            Name of the experiment connected to the study.
        color_groups (list):
            List of colors for each group name for plotting.
        output_dir (str):
            Directory where the plots will be saved.
        num_samples (int, optional):
            Number of samples to take from the predicted data for plotting.
    """
    # Get the number of participants.
    num_participants = len(gp_processed.drop_duplicates(subset="Participant_Num"))

    ''' # Create color mapping based on groups.
    groups = gp_processed.sort_values(by="Group_Name")["Group_Name"].unique()
    print("unique groups indi ppc plot:", groups)
    group_colors = dict(zip(groups, color_groups))'''

    # Create figure for 40 participants.
    rows = (num_participants // 5) + (num_participants % 5 > 0)
    cols = 5
    fig, axes = plt.subplots(
        nrows=rows, ncols=5, figsize=(18, 18), sharex=True, sharey=True
    )
    axes = axes.flatten()

    # Sample predicted data.
    indices = np.random.choice(y_pred.shape[0], num_samples, replace=False)
    y_pred_sampled = y_pred[indices]

    # Make a subplot for each participant.
    for idx, ax in enumerate(axes):
        if idx >= num_participants:
            ax.axis("off")
            continue

        # Retrieve participant's data.
        participant_num = (
            gp_processed.drop_duplicates(subset="Participant_Num").iloc[idx][
                "Participant_Num"
            ]
            - 1
        )
        group_name = gp_processed.drop_duplicates(
            subset="Participant_Num"
        ).iloc[idx]["Group_Name"]

        stimulus_grouped = stimulus.iloc[participant_num, :]
        plot_data = []
        for stim_label in stimulus_level:
            # Find indices where the stimulus occurs.
            trial_indices = stimulus_grouped == stim_label
            if not trial_indices.any():
                continue

            # Calculate means for the actual data.
            act_mean = np.nanmean(y_actual[participant_num, trial_indices])

            # Calculate means for the sampled predicted data.
            pred_mean = np.nanmean(y_pred_sampled[:, participant_num, trial_indices])

            # Add the computed data to the data to plot.
            plot_data.append(
                {
                    "Stimulus": stim_label,
                    "Value": pred_mean,
                    "Type": "Model-based prediction mean",
                }
            )
            plot_data.append(
                {
                    "Stimulus": stim_label,
                    "Value": act_mean,
                    "Type": "Observed data mean",
                }
            )

        # Prepare a DataFrame for plotting.
        df_plot = pd.DataFrame(plot_data)
        df_plot["Stimulus"] = pd.Categorical(
            df_plot["Stimulus"], categories=stimulus_level, ordered=True
        )

        # Generate lineplots for the participant mean of the observed and predicted data.
        sns.lineplot(
            data=df_plot,
            x="Stimulus",
            y="Value",
            hue="Type",
            style="Type",
            markers="o",
            ax=ax,
            dashes=False,
            errorbar=None,
            palette={
                "Model-based prediction mean": "blue",
                "Observed data mean": "black",
            },
        )

        # Set subplot parameters.
        ax.set_title(f"Participant {participant_num + 1}", weight="bold", fontsize=10)
        ax.set_ylim(0, 10.5)
        ax.set_yticks([1, 5, 10])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_facecolor("whitesmoke")

        # Show axis ticks only for external axis.
        if (idx % cols) != 0:
            ax.tick_params(axis="y", which="both", left=False)
        if idx < cols * (rows - 1):
            ax.tick_params(axis="x", which="both", bottom=False)
        ax.get_legend().remove()

        # Add a colored label for the participant's dominant group.
        ax.text(
            0.5,
            0.95,
            group_name,
            transform=ax.transAxes,
            fontsize=9,
            ha="center",
            va="top",
            bbox=dict(facecolor=color_groups.get(group_name, "black"), alpha=0.5),
        )

    # Adjust layout and add overall title.
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05)
    plt.subplots_adjust(hspace=0.3)
    fig.suptitle(experiment, fontsize=12, weight="bold")

    # Create overall x-label and y-label for the whole figure.
    fig.text(
        0.5, 0.03, "Stimulus", ha="center", va="center", weight="bold", fontsize=12
    )
    fig.text(
        0.006,
        0.5,
        "US Expectancy (observed data scale: 1-10)",
        ha="center",
        va="center",
        rotation="vertical",
        weight="bold",
        fontsize=12,
    )

    # Create overall legend.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=2,
        fancybox=True,
        shadow=True,
        fontsize=12,
    )

    fig.savefig(
        f"{output_dir}/Posterior_Predictive_Checks_Individual_{experiment.replace(" ", "_")}.png",
        dpi=300,
    )


# Plot individual posterior predictive checks for study 1.
ppc_plot_indi(
    y_pred_study1,
    y_actual_study1,
    stimulus_s1,
    gp_allocation(gp_samples[0]),
    stimulus_levels["s1"],
    experiments[0],
    color_groups,
    PPC_dir,
)
# Plot individual posterior predictive checks for study 2.
ppc_plot_indi(
    y_pred_study2,
    y_actual_study2,
    stimulus_s2,
    gp_allocation(gp_samples[1]),
    stimulus_levels["s2"],
    experiments[1],
    color_groups,
    PPC_dir,
)
