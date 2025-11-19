"""
Module: simulation_1.py

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
conversion of the original R code to Python, it is designed to demonstrate the impact
of different latent mechanisms, such as learning rates, generalization rates, perceptual
standard deviations and response function weights on observed behaviors in a structured,
illustrative format.
This simulation has an illustrative role and helps in visualizing the theoretical
effects these parameters may have, as described by the computational model in the study.

### Functions:
- lg_fun: Generates detailed trial data with associative learning, similarities,
  response dynamics based on specific model parameters.
- sim_lr_plot: Helper function to plot associative strengths across learning trials
  for specific simulated data, highlighting responses with or without a US.
- sim_lr_comp_plot: Visualizes the effects of high and low learning rates on associative
  strengths across specified learning trials.
- sim_ge_plot: Helper function to plot how physical and perceived similarity to conditioned
  stimuli varies across different stimuli sizes for all simulated trials for specific
  simulatied data.
- sim_ge_comp_plot: Visualizes the effects of high and low generalization rates on the
  similarity to conditioned stimuli across different stimuli sizes for all simulated
  trials.
- sim_pervar_comp_plot: Visualizes the impact of high and low perceptual variability
  on the perceived distances and similarities to conditioned stimuli across various
  stimuli sizes for all simulated trials.
- sim_logit: Provides a visualization of the logistic function's response over a range
  of generalization gradient values, illustrating how changes in logistic weight
  parameters affect response probabilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create a directory to save the plots.
os.makedirs("../Plots", exist_ok=True)

# Define parameters for the simulation.
CSp_index = 4
Nactrials = 24
Ngetrials = 76
Ntrials = Nactrials + Ngetrials
seed = 20221001

# Set the seed for reproducibility.
np.random.seed(seed)

# Generate stimulus sizes.
num_points = int((119.42 - 50.80) / 7.624) + 1
size = np.round(np.linspace(50.80, 119.42, num=num_points), 2)
CSp_size = size[CSp_index]

# Create stimulus sizes for each trial.
s_size = np.concatenate(
    (np.full(Nactrials, CSp_size), np.random.choice(size, Ngetrials, replace=True))
)

# Calculate distance from CS+ size.
d = np.abs(np.round(s_size - CSp_size)).astype(int)

# Create stimulus types.
stimulus = np.where(d == 0, "CS+", "TS")

# Create responses with specified probabilities.
r = np.random.choice([1, 0], size=Ntrials, p=[0.8, 0.2], replace=True)

# Create k values.
k = np.where(d == 0, 1, 0)

# Create a directory to save simulation plots.
SIM_dir = "../Plots/Simulation/Simulation_part_1"
os.makedirs(SIM_dir, exist_ok=True)


# Simulation - generative processes.
def lg_fun(
    alpha,
    lambda_,
    w1=10,
    w0=-5,
    sigma=0,
    trials=Ntrials,
    persd=0,
    K=10,
    A=1,
    seed=20221001,
):
    """
    Generates learning and response data based on provided learning and generalization
    rates parameters.

    This function simulates a learning experiment where stimuli are presented across
    trials, with associative strengths updated based on a specified learning rate (`alpha`).
    It calculates similarity to a conditioned stimulus (CS+) using perceptual, physical
    distances and a specified generalization rate (`lambda_`), influencing the responses
    generated from a logistic function.

    Args:
        alpha (float):
            Learning rate parameter.
        lambda_ (float):
            Generalization rate parameter.
        w1 (float):
            Latent-observed scaling factor.
        w0 (float):
            Baseline weight of the response function (baseline response).
        sigma (float):
            Standard deviation for the response variability (response noise).
        trials (int):
            Total number of trials in the simulation.
        persd (float):
            Standard deviation used for generating perceived distances.
        K (float):
            Upper limit of the logistic function response.
        A (float):
            Lower limit of the logistic function response.
        seed (int):
            Seed for random number generator to ensure reproducibility.

    Returns:
        df (DataFrame):
            Contains all simulation data per trial including:
            - theta (non-linear transformation (latent - observed scale)),
            - s_per (perceived similarity to CS+),
            - s_phy (physical similarity to CS+),
            - v (associative strengths),
            - g (generalization gradients),
            - y (simulated responses),
            - trials (number of trials),
            - phyd (physical distances to CS+),
            - perd (perceived distances to CS+),
            - r (reinforcement received),
            - s_size (stimulus sizes).
    """
    # Initialize arrays for the parameters.
    y = np.zeros(trials)
    theta = np.zeros(trials)
    s_per = np.zeros(trials)
    s_phy = np.zeros(trials)
    perd = np.zeros(trials)
    phyd = d.copy()
    v = np.zeros(trials + 1)
    g = np.zeros(trials)

    # Set the seed for reproducibility.
    np.random.seed(seed)

    # Iterate through trials.
    for t in range(Ntrials):

        # Calculate perceived distance as random value around physical distance with
        # given standard deviation.
        perd[t] = np.abs(np.random.normal(phyd[t], persd))

        # Update associative strength value v based on response and learning parameter.
        v[t + 1] = np.where(k[t] == 1, v[t] + alpha * (r[t] - v[t]), v[t])

        # Calculate similarity based on perceived and physical distances.
        s_per[t] = np.exp(-lambda_ * perd[t])
        s_phy[t] = np.exp(-lambda_ * phyd[t])

        # Compute g generalization gradient.
        g[t] = v[t] * s_per[t]

        # Compute theta non-linear transformation.
        theta[t] = A + (K - A) / (1 + np.exp(-(w0 + w1 * g[t])))

        # Generate y value as random value around theta with given standard deviation.
        y[t] = np.random.normal(theta[t], sigma)

    # Create a DataFrame to hold all the computed data.
    df = pd.DataFrame(
        {
            "theta": theta,
            "s_per": s_per,
            "s_phy": s_phy,
            "v": v[:-1],
            "g": g,
            "stimulus": stimulus,
            "y": y,
            "trials": np.arange(1, trials + 1),
            "phyd": phyd,
            "perd": perd,
            "r": r,
            "s_size": s_size,
        }
    )

    # Create new column with perceived distance group.
    df["percept"] = np.select(
        [df["perd"] < 5, (df["perd"] > 5) & (df["perd"] < 15)], ["1", "2"], default="3"
    )

    return df


# Learning rates comparison using only learning trials.
def sim_lr_comp_plot(output_dir, alpha_high, alpha_low, lambda_, w1):
    """
    Plots associative strength changes for high and low learning rates across
    learning trials.

    This function compares the effects of two learning rates on associative strength
    by generating data for each rate using the `lg_fun` function, and visualizing the
    change in associative strength over the initial learning trials, highlighting
    responses to CS+ and other stimuli.

    Args:
        output_dir (str):
            Directory where the plots will be saved.
        alpha_high (float):
            Higher learning rate to be tested.
        alpha_low (float):
            Lower learning rate to be tested.
        lambda_ (float):
            Generalization rate applied in the simulation.
        w1 (float):
            Latent-observed scaling factor.
    """
    # Generate data for high and low learning rates.
    data_high_learning = lg_fun(alpha=alpha_high, lambda_=lambda_, w1=w1)
    data_low_learning = lg_fun(alpha=alpha_low, lambda_=lambda_, w1=w1)

    # Create the figure.
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Generate a subplot using only learning trials.
    def sim_lr_plot(data, ax, title):
        """
        Plots associative strengths across learning trials for a given set of
        simulation data.

        This helper function generates plots specific to associative strength
        developments during learning trials, highlighting responses in scenarios
        with and without a US present.

        Args:
            data (DataFrame):
                The simulated data containing trials, associative strengths ('v'),
                stimuli types, and US responses.
            ax (matplotlib.axes.Axes):
                The Axes object on which to plot the graph.
            title (str):
                Title for the subplot to provide context about the learning rate settings.
        """
        # Initialize markers and colors for the plot.
        marker_dict = {1: "^", 0: "o"}
        color_dict = {"CS+": "red", "TS": "blue"}

        # Create a lineplot for the associative strength changes across learning trials.
        sns.lineplot(
            data=data[data["trials"] <= 24],
            x="trials",
            y="v",
            ax=ax,
            color="black",
            legend=False,
            zorder=1,
        )

        # Iterate through stimuli types.
        for stim in ["CS+", "TS"]:

            # Filter the data based on the current stimulus type.
            subset = data[data["stimulus"] == stim]

            # Iterate through possible US conditions.
            for resp in [1, 0]:

                # Filter the data based on the current US condition.
                resp_subset = subset[subset["r"] == resp]

                # If the current stimulus type is CS+ use scatter plot to generate
                # points that show if the US is present for a specific trial.
                if stim == "CS+":
                    ax.scatter(
                        resp_subset["trials"],
                        resp_subset["v"],
                        color=color_dict[stim],
                        marker=marker_dict[resp],
                        label=f"{'US' if resp == 1 else 'US absent'}",
                        s=100,
                        zorder=2,
                    )

        # Set subplot parameters.
        ax.set_title(title, weight="bold", fontsize=12)
        ax.set_xlabel("Trials", weight="bold", fontsize=12)
        ax.set_ylabel("Associative strengths", weight="bold", fontsize=12)
        ax.set_facecolor("whitesmoke")

    # Generate a subplot using only learning trials for high learning rate.
    sim_lr_plot(
        data_high_learning[data_high_learning["trials"] <= 24],
        axs[0],
        f"$\\mathbf{{\\alpha}}$ = {alpha_high}; $\\mathbf{{\\lambda}}$ = {lambda_}; perceptual sd = 0",
    )

    # Generate a subplot using only learning trials for low learning rate.
    sim_lr_plot(
        data_low_learning[data_low_learning["trials"] <= 24],
        axs[1],
        f"$\\mathbf{{\\alpha}}$ = {alpha_low}; $\\mathbf{{\\lambda}}$ = {lambda_}; perceptual sd = 0",
    )

    # Add overall titles.
    fig.text(
        0.19,
        0.99,
        "High Learning Rate $\\mathbf{{\\alpha}}$",
        va="top",
        fontsize=14,
        weight="bold",
    )
    fig.text(
        0.67,
        0.99,
        "Low Learning Rate $\\mathbf{{\\alpha}}$",
        va="top",
        fontsize=14,
        weight="bold",
    )

    # Create overall legend.
    handles, labels = axs[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        ncol=2,
        fancybox=True,
        shadow=True,
        fontsize=12,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(
        f"{output_dir}/Simulation_part_1_Learning_Rates_comparison.png", dpi=300
    )


# Plot comparison between high and low learning rates using only learning trials.
sim_lr_comp_plot(SIM_dir, alpha_high=0.3, alpha_low=0.01, lambda_=0.3, w1=10)


# Generalization rates comparison using all trials.
def sim_ge_comp_plot(output_dir, lambda_high, lambda_low, alpha, w1):
    """
    Visualizes the effects of high and low generalization rates on stimulus generalization.

    Compares how high and low generalization rates (`lambda_`) affect the similarity responses to CS+
    across all trials. This function generates data sets for each generalization rate using the `lg_fun` function,
    visualizes similarity measurements as a function of stimulus size, and plots how these measurements scatter for CS+ and other stimuli.

    Args:
        output_dir (str):
            Directory where the plots will be saved.
        lambda_high (float):
            High generalization decay rate to be tested.
        lambda_low (float):
            Low generalization decay rate to be tested.
        alpha (float):
            Learning rate used in the simulation.
        w1 (float):
            Latent-observed scaling factor.
    """
    # Generate data for high and low generalization rates.
    data_high_generalization = lg_fun(alpha=alpha, lambda_=lambda_high, w1=w1)
    data_low_generalization = lg_fun(alpha=alpha, lambda_=lambda_low, w1=w1)

    # Create the figure.
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)  # 1 row, 2 columns

    # Generate a subplot using all trials.
    def sim_ge_plot(data, ax, title):
        """
        Plots generalization gradients across all trials for given simulation data.

        This helper function visualizes the generalization gradients as line and scatter
        plots to show how physical and perceived similarity to a conditioned stimulus
        varies across different stimuli sizes.

        Args:
            data (DataFrame):
                The simulated data containing trials, sizes of stimuli, and similarity
                measures both perceived ('s_per') and physical ('s_phy').
            ax (matplotlib.axes.Axes):
                The Axes object on which to plot the graph.
            title (str):
                Title for the subplot to provide context about the generalization rate
                settings.
        """
        # Sort data by stimuli sizes.
        data_sorted = data.sort_values(by="s_size")

        # Create a lineplot for stimuli physical similarity to CS+ across various sizes.
        sns.lineplot(
            data=data_sorted,
            x="s_size",
            y="s_phy",
            ax=ax,
            color="black",
            markers=None,
            legend=False,
            zorder=1,
        )

        # Use scatter plot to generate points for every stimuli across various sizes.
        sns.scatterplot(
            data=data,
            x="s_size",
            y="s_per",
            hue="stimulus",
            style="stimulus",
            markers="o",
            palette={"CS+": "red", "TS": "blue"},
            ax=ax,
            zorder=2,
        )

        # Set subplot parameters.
        ax.set_title(title, weight="bold", fontsize=12)
        ax.set_xlabel("Stimulus Size", weight="bold", fontsize=12)
        ax.set_ylabel("Similarity to CS+", weight="bold", fontsize=12)
        ax.get_legend().remove()
        ax.set_facecolor("whitesmoke")

    # Generate a subplot using all learning trials for high generalization rate.
    sim_ge_plot(
        data_high_generalization,
        axs[0],
        f"$\\mathbf{{\\lambda}}$ = {lambda_high}; $\\mathbf{{\\alpha}}$ = {alpha}; perceptual sd = 0",
    )

    # Generate a subplot using all learning trials for low generalization rate.
    sim_ge_plot(
        data_low_generalization,
        axs[1],
        f"$\\mathbf{{\\lambda}}$ = {lambda_low}; $\\mathbf{{\\alpha}}$ = {alpha}; perceptual sd = 0",
    )

    # Add overall titles.
    fig.text(
        0.165,
        0.99,
        "High Generalization Rate $\\mathbf{{\\lambda}}$",
        va="top",
        fontsize=14,
        weight="bold",
    )
    fig.text(
        0.64,
        0.99,
        "Low Generalization Rate $\\mathbf{{\\lambda}}$",
        va="top",
        fontsize=14,
        weight="bold",
    )

    # Create overall legend.
    handles, labels = axs[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        ncol=2,
        fancybox=True,
        shadow=True,
        fontsize=12,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(
        f"{output_dir}/Simulation_part_1_Generalization_Rates_comparison.png", dpi=300
    )


# Plot comparison between high and low generalization rates using all trials.
sim_ge_comp_plot(SIM_dir, lambda_high=0.3, lambda_low=0.01, alpha=0.1, w1=10)


# Perceptual variability comparison using all trials.
def sim_pervar_comp_plot(output_dir, persd_low, persd_high, alpha, lambda_):
    """
    Compares the impact of low and high perceptual variability on stimulus
    generalization.

    This function examines how different levels of perceptual noise
    (standard deviation of perceived distances) affect the generalization gradients
    and responses across all trials. It uses the `lg_fun` to generate data for low and
    high perceptual variability and plots the perceived and physical similarity measures.

    Args:
        output_dir (str):
            Directory where the plots will be saved.
        persd_low (float):
            Lower perceptual standard deviation to be tested.
        persd_high (float):
            Higher perceptual standard deviation to be tested.
        alpha (float):
            Learning rate used in the simulations.
        lambda_ (float):
            Generalization rate applied in the simulation.
    """
    # Generate data for low and high perceptual variability.
    data_low_variability = lg_fun(alpha=alpha, lambda_=lambda_, w1=10, persd=persd_low)
    data_high_variability = lg_fun(
        alpha=alpha, lambda_=lambda_, w1=10, persd=persd_high
    )

    # Create the figure.
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    # Generate a subplot using all trials.
    def sim_pervar_plot(data, ax, title):

        # Sort data by stimuli sizes.
        data_sorted = data.sort_values(by="s_size")

        # Categorize perceived distances for the plot.
        data["percept"] = np.select(
            [data["perd"] < 10, (data["perd"] >= 10) & (data["perd"] < 20)],
            ["Small", "Medium"],
            default="Big",
        )

        # Create a column in the sorted data for the mean of the peceived similarity
        # to CS+ grouped by stimuli sizes.
        data_sorted["meansper"] = data.groupby("s_size")["s_per"].transform("mean")

        # Create a lineplot for the physiscal similarity to CS+ across various stimuli sizes.
        sns.lineplot(
            data=data_sorted,
            x="s_size",
            y="s_phy",
            color="gray",
            linestyle="dashed",
            label="Physical similarity",
            ax=ax,
            legend=False,
            alpha=0.5,
            zorder=0,
        )

        # Create a lineplot for the for the mean of the perceived similarity to CS+
        # across various stimuli sizes.
        sns.lineplot(
            data=data_sorted,
            x="s_size",
            y="meansper",
            color="black",
            ax=ax,
            label="Mean perceived similarity",
            zorder=1,
        )

        # Use scatterplot to generate points markers on the subplot to differentiate
        # between small, medium and big perceived distances to CS+ across various
        # stimulus sizes.
        sns.scatterplot(
            x="s_size",
            y="s_per",
            data=data,
            hue="stimulus",
            style=data["percept"],
            markers={"Small": "o", "Medium": "s", "Big": "D"},
            ax=ax,
            palette={"CS+": "red", "TS": "blue"},
            alpha=0.3,
            s=50,
            zorder=2,
        )

        # Set subplot parameters.
        ax.set_title(title, weight="bold", fontsize=12)
        ax.set_xlabel("Stimulus Size", weight="bold", fontsize=12)
        ax.set_ylabel("Similarity to CS+", weight="bold", fontsize=12)
        ax.set_facecolor("whitesmoke")
        ax.get_legend().remove()

    # Generate a subplot using all learning trials for high perceptual variability.
    sim_pervar_plot(
        data_high_variability,
        axs[0],
        f"$\\mathbf{{\\lambda}}$ = {lambda_}; $\\mathbf{{\\alpha}}$ = {alpha}; perceptual sd: {persd_high}",
    )

    # Generate a subplot using all learning trials for low perceptual variability.
    sim_pervar_plot(
        data_low_variability,
        axs[1],
        f"$\\mathbf{{\\lambda}}$ = {lambda_}; $\\mathbf{{\\alpha}}$ = {alpha}; perceptual sd: {persd_low}",
    )

    # Add overall titles.
    fig.text(
        0.145, 0.99, "High Perceptual Variability", va="top", fontsize=14, weight="bold"
    )
    fig.text(
        0.612, 0.99, "Low Perceptual Variability", va="top", fontsize=14, weight="bold"
    )

    # Create overall legend.
    handles, labels = axs[0].get_legend_handles_labels()
    from matplotlib.legend import Legend

    legend_elements = []
    for handle, label in zip(handles, labels):
        if label in ["Small", "Medium", "Big"]:
            legend_elements.append((handle, label))
    other_elements = [
        (handle, label)
        for handle, label in zip(handles, labels)
        if label not in ["Small", "Medium", "Big", "percept", "stimulus"]
    ]
    fig.legend(
        [item[0] for item in other_elements],
        [item[1] for item in other_elements],
        loc="lower right",
        title="Stimulus & Response",
        bbox_to_anchor=(0.65, 0),
        ncol=4,
        fancybox=True,
        shadow=True,
        fontsize=10,
    )
    fig.legend(
        [item[0] for item in legend_elements],
        [item[1] for item in legend_elements],
        loc="lower left",
        title="Perceived distance to CS+",
        bbox_to_anchor=(0.65, 0),
        ncol=3,
        fancybox=True,
        shadow=True,
        fontsize=10,
    )

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    fig.savefig(
        f"{output_dir}/Simulation_part_1_Perceptual_Variability_comparison.png", dpi=300
    )


# Plot comparison between high and low perceptual variability using all trials.
sim_pervar_comp_plot(SIM_dir, persd_low=2, persd_high=20, alpha=0.1, lambda_=0.1)


# Logit function visualization plot.
def sim_logit(output_dir, w01, w02, w03, w04, w05, w06, w11, w12, w13, w14, w15, w16):
    """
    Visualizes the logistic function response across a range of generalization
    gradient values.

    This function explores how different parameter settings for the logistic function
    (`w0` and `w1`) affect the response probabilities. It generates plots for multiple
    parameter combinations, showing the non-linear transformation of the generalization
    gradient into a response probability.

    Args:
        output_dir (str):
            Directory where the plots will be saved.
        w01, w02, ..., w16 (float):
            Weight parameters for the logistic function, representing different
            conditions or sets of parameters to be plotted.
    """
    # Generate 'g' generalization gradient values.
    g = np.linspace(-1, 1, 100)

    # Define parameters settings and their colors for plotting.
    params = [
        (w01, w11, f"w0 = {w01}, w1 = {w11}", "red"),
        (w02, w12, f"w0 = {w02}, w1 = {w12}", "blue"),
        (w03, w13, f"w0 = {w03}, w1 = {w13}", "pink"),
        (w04, w14, f"w0 = {w04}, w1 = {w14}", "green"),
        (w05, w15, f"w0 = {w05}, w1 = {w15}", "purple"),
        (w06, w16, f"w0 = {w06}, w1 = {w16}", "brown"),
    ]

    # Generate color map.
    color_map = {label: color for (_, _, label, color) in params}

    # Initialize a list for the data to plot.
    data_list = []

    # Iterate through the parameters and their settings.
    for w0, w1, label, color in params:

        # Compute response for the current parameters.
        response = 1 + 9 / (1 + np.exp(-(w0 + w1 * g)))

        # Create a DataFrame with g, the response, the label and the plotting color
        # for the current parameters, and add it to the list of the data to plot.
        data_list.append(
            pd.DataFrame({"g": g, "response": response, "param": label, "color": color})
        )

    # Concatenate all data.
    data = pd.concat(data_list)

    # Create the figure.
    plt.figure(figsize=(10, 8))

    # Get figure ax.
    ax = plt.gca()

    # Iterate through the parameters labels in the color map.
    for label in color_map:

        # Filter the data to plot based on the current parameters label.
        subset = data[data["param"] == label]

        # Plot the curve for the current parameters.
        plt.plot(subset["g"], subset["response"], label=label, color=color_map[label])

    # Set plot parameters.
    plt.xlabel("g", weight="bold", fontsize=12)
    plt.ylabel("$\\mathbf{{\\theta}}$", fontsize=12)
    plt.title("Logistic Function Response", weight="bold", fontsize=14)
    plt.legend(title="Parameter Sets", fontsize=10)
    ax.legend(fontsize=12, fancybox=True, shadow=True)
    ax.set_ylim(0.5, 10.5)
    ax.set_yticks(np.arange(1, 11, 1))
    ax.set_facecolor("whitesmoke")

    plt.savefig(
        f"{output_dir}/Simulation_part_1_Logistic_Function_Response.png", dpi=300
    )


# Plot logit function for various parameters pairings.
sim_logit(
    SIM_dir,
    w01=3,
    w02=3,
    w03=0,
    w04=0,
    w05=-3,
    w06=-3,
    w11=5,
    w12=20,
    w13=5,
    w14=20,
    w15=5,
    w16=20,
)
