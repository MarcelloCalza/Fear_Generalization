"""
Module: data_processing.py

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
conversion of the original R code to Python, ensuring reproducibility and
compatibility with PyMC for further analysis. It processes data already
partially preprocessed in R and imported as a starting point.

The preprocessing is tailored to the two experiments and includes:
- Handling US expectancy data.
- Calculating perceptual and physical distances.
- Optionally managing reinforcement and CS indicator data.

### Functions:
- data_input_s1: Processes data for the simple conditioning experiment.
- data_input_s2: Processes data for the differential conditioning experiment.
"""

import numpy as np
import pandas as pd
import pyreadr
import os
import pickle

# Create directories to store the data.
os.makedirs("../Preprocessed_data", exist_ok=True)
os.makedirs("../PYMC_input_data", exist_ok=True)

# Load long format data from the first dataset: simple conditioning.
data_s1 = pyreadr.read_r("../Data/Data_s1.rds")[None]

# Change type of participant and stimulus columns.
data_s1["participant"] = data_s1["participant"].astype(int)
data_s1["stimulus"] = data_s1["stimulus"].astype(str)

# Sort values inside participant and trials columns.
data_s1 = (
    data_s1.sort_values(["participant", "trials"]).reset_index().drop("index", axis=1)
)

# Create a sequence of stimulus sizes ranging from 50.80 to 119.42.
stimulus_size = np.round(np.arange(50.80, 119.42 + 7.624, 7.624), 2)

# Rename the columns to extract from the experiment 1 long format data.
data_s1_long = data_s1.rename(
    columns={
        "US_expect": "y",
        "Phy_size": "Sphy",
        "Per_size": "Sper",
        "CS_phy": "CSphy",
        "CStrials": "CSindicator1",
        "US": "shock",
    }
)

# Create a CSindicator2 column.
data_s1_long["CSindicator2"] = data_s1_long["CSindicator1"]

# Convert the data of experiment 1 from long to wide format.
pymc_input_s1_pre = pd.pivot_table(
    data_s1_long,
    index="participant",
    columns="trials",
    values=["y", "Sphy", "Sper", "CSphy", "CSindicator1", "CSindicator2", "shock"],
    dropna=False,
).sort_index()

# Set the values in the CSindicator2 matrix to 0 for non-learning trials.
pymc_input_s1_pre.loc[:, pd.IndexSlice["CSindicator2", 15:189]] = 0

# Create a CS_index column by calculating the sum of the CSindicator1 values.
data_s1_long["CS_index"] = (
    data_s1_long["CSindicator1"].groupby(data_s1_long["participant"]).cumsum()
)

# Convert the CS_index data from long to wide format.
CS_index = pd.pivot_table(
    data_s1_long,
    index="participant",
    columns="trials",
    values=["CS_index"],
    dropna=False,
)

# Add the CS_index data to the wide format data of experiment 1.
pymc_input_s1_pre = pd.merge(
    pymc_input_s1_pre, CS_index, left_index=True, right_index=True
).sort_index(level=0, axis=1)

# Create CS_per_s1 by extracting CS perception.
CS_per_s1 = data_s1[data_s1["stimulus"] == "CS+"]
CS_per_s1 = CS_per_s1[["participant", "Per_size"]].reset_index(drop=True)
CS_per_s1["trials"] = CS_per_s1.groupby("participant").cumcount() + 1

# Convert the CS_per_s1 data from long to wide format.
CS_per_s1 = CS_per_s1.pivot_table(
    index="participant", columns="trials", values="Per_size", dropna=False
)

# Create a dataframe for CS_per_updatemean_s1.
CS_per_updatemean_s1 = pd.DataFrame(index=CS_per_s1.index, columns=CS_per_s1.columns)

# Fill CS_per_updatemean_s1 by computing the moving average for CS perception.
for i in range(len(CS_per_s1)):
    for j in range(len(CS_per_s1.columns)):
        CS_per_updatemean_s1.iloc[i, j] = CS_per_s1.iloc[i, : j + 1].sum(
            skipna=True
        ) / (j + 1)

# Create empty matrices to store perceptual and physical distance data.
d_list_s1 = {
    "d_per": np.zeros((40, 188)),
    "d_phy": np.zeros((40, 188)),
    "participant": [],
    "trials": [],
}

# Compute perceptual and physical distances.
for i in range(40):
    for j in range(188):
        d_list_s1["d_per"][i, j] = round(
            abs(
                pymc_input_s1_pre.loc[i + 1, ("Sper", j + 1)]
                - CS_per_updatemean_s1.loc[
                    i + 1, pymc_input_s1_pre.loc[i + 1, ("CS_index", j + 1)]
                ]
            ),
            2,
        )
        d_list_s1["d_phy"][i, j] = round(
            abs(pymc_input_s1_pre.loc[i + 1, ("Sphy", j + 1)] - stimulus_size[6]), 2
        )
        d_list_s1["trials"].append(j + 1)
        d_list_s1["participant"].append(i + 1)


# PyMC input data generator function for experiment 1: simple conditioning.
def data_input_s1(L, indicator):
    """
    Prepare input data for PyMC analysis in the context of simple conditioning.

    This function processes and formats the data required for PyMC analysis
    based on the parameters provided. It handles US expectancy (`y_data`),
    perceptual distances (`d_per`), physical distances (`d_phy`), and,
    optionally, reinforcement (`r`) and indicator (`k`) data.

    Args:
        L (int):
            Specifies the type of data to handle:
            - 1: Use data for trials 15–188 only (excluding learning trials).
            - 2: Use the full dataset, including learning trials.
        indicator (int, optional):
            Specifies which CS indicator to use (required if `L == 2`):
            - 1: Use `CSindicator1`.
            - 2: Use `CSindicator2`.
            - None: Skip indicator handling.

    Returns:
        data (dict):
            A dictionary with the following keys:
            - 'Nparticipants' (int): Number of participants.
            - 'Ntrials' (int): Number of trials.
            - 'Nactrials' (int): Number of active trials (set to 14).
            - 'd_per' (ndarray): Perceptual distances for each participant/trial.
            - 'd_phy' (ndarray): Physical distances for each participant/trial.
            - 'y' (ndarray): US expectancy data.
            - 'r' (ndarray, optional): Reinforcement data (if `L == 2`).
            - 'k' (ndarray, optional): CS indicator data (if `L == 2`).
    """
    # Handle US expectancy data.
    if L == 1:
        y_data = (
            pymc_input_s1_pre.loc[:, pd.IndexSlice["y", 15:188]]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy()
        )
    else:
        y_data = pymc_input_s1_pre["y"].apply(pd.to_numeric, errors="coerce").to_numpy()

    # Handle perceptual and physical distances data.
    d_per_data = (
        d_list_s1["d_per"][:, 14:188].astype(np.float64)
        if L == 1
        else d_list_s1["d_per"].astype(np.float64)
    )
    d_phy_data = (
        d_list_s1["d_phy"][:, 14:188].astype(np.float64)
        if L == 1
        else d_list_s1["d_phy"].astype(np.float64)
    )

    data = {
        "Nparticipants": y_data.shape[0],
        "Ntrials": y_data.shape[1],
        "Nactrials": 14,
        "d_per": d_per_data,
        "d_phy": d_phy_data,
        "y": y_data,
    }

    # Handle r and k data.
    if L == 2:
        data["r"] = (
            pymc_input_s1_pre["shock"].apply(pd.to_numeric, errors="coerce").to_numpy()
        )

        if indicator == 1:
            data["k"] = (
                pymc_input_s1_pre["CSindicator1"]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy()
            )
        else:
            data["k"] = (
                pymc_input_s1_pre["CSindicator2"]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy()
            )

    return data


# Generate and save PyMC input data for experiment 1: simple conditioning.
# PyMC input data without learning trials.
pickle.dump(data_input_s1(1, None), open("../PYMC_input_data/Data1_PYMCinput_G.pkl", "wb"))
# PyMC input data with an assumption of non-continuous learning.
pickle.dump(data_input_s1(2, 2), open("../PYMC_input_data/Data1_PYMCinput_LG.pkl", "wb"))
# PyMC input data with an assumption of continuous learning.
pickle.dump(data_input_s1(2, 1), open("../PYMC_input_data/Data1_PYMCinput_CLG.pkl", "wb"))

# Save long format data for experiment 1: simple conditioning.
pickle.dump(data_s1, open("../Preprocessed_data/Data_s1.pkl", "wb"))

# Load long format data from the second dataset: differential conditioning.
data_s2 = pyreadr.read_r("../Data/Data_s2.rds")[None]

# Change type of participant and stimulus columns.
data_s2["participant"] = data_s2["participant"].astype(int)
data_s2["stimulus"] = data_s2["stimulus"].astype(str)

# Sort values inside participant and trials columns.
data_s2 = (
    data_s2.sort_values(["participant", "trials"]).reset_index().drop("index", axis=1)
)

# Rename the columns to extract from the experiment 2 long format data.
data_s2_long = data_s2.rename(
    columns={
        "US_expect": "y",
        "Phy_size": "Sphy",
        "Per_size": "Sper",
        "CS_phy_p": "CSphy_p",
        "CS_phy_m": "CSphy_m",
        "CSptrials": "CSindicator1_p",
        "CSmtrials": "CSindicator1_m",
        "USp": "US_p",
        "USm": "US_m",
    }
)

# Create CSindicator2_p and CSindicator2_m columns.
data_s2_long["CSindicator2_p"] = data_s2["CSptrials"]
data_s2_long["CSindicator2_m"] = data_s2["CSmtrials"]

# Convert the data of experiment 2 from long to wide format.
pymc_input_s2_pre = pd.pivot_table(
    data_s2_long,
    index="participant",
    columns="trials",
    values=[
        "y",
        "Sphy",
        "Sper",
        "CSphy_p",
        "CSphy_m",
        "CSindicator1_p",
        "CSindicator2_p",
        "CSindicator1_m",
        "CSindicator2_m",
        "US_p",
        "US_m",
    ],
    dropna=False,
).sort_index()

# Set the values in CSindicator2_p and CSindicator2_m matrices to 0 for Nactrials.
pymc_input_s2_pre.loc[:, pd.IndexSlice["CSindicator2_p", 25:181]] = 0
pymc_input_s2_pre.loc[:, pd.IndexSlice["CSindicator2_m", 25:181]] = 0

# Initialize CSp_index and CSm_index columns.
data_s2_long["CSp_index"] = np.zeros(
    (
        pymc_input_s2_pre["CSindicator1_p"].shape[0],
        pymc_input_s2_pre["CSindicator1_p"].shape[1],
    )
).flatten()

data_s2_long["CSm_index"] = np.zeros(
    (
        pymc_input_s2_pre["CSindicator1_m"].shape[0],
        pymc_input_s2_pre["CSindicator1_m"].shape[1],
    )
).flatten()

# Convert the CSp_index and CSm_index data from long to wide format.
CSp_index = pd.pivot_table(
    data_s2_long,
    index="participant",
    columns="trials",
    values=["CSp_index"],
    dropna=False,
)

CSm_index = pd.pivot_table(
    data_s2_long,
    index="participant",
    columns="trials",
    values=["CSm_index"],
    dropna=False,
)

# Add CSp_index and CSm_index to the wide format data of experiment 2.
pymc_input_s2_pre = pd.concat(
    [pymc_input_s2_pre, CSp_index, CSm_index], axis=1
).sort_index(level=0, axis=1)

# Calculate the cumulative sum for CSindicator1_p and store it in CSp_index.
pymc_input_s2_pre["CSp_index"] = pymc_input_s2_pre["CSindicator1_p"].cumsum(axis=1)

# Calculate the cumulative sum for CSindicator1_m and store it in CSm_index.
pymc_input_s2_pre["CSm_index"] = pymc_input_s2_pre["CSindicator1_m"].cumsum(axis=1)

# Set 0 values in CSp_index and CSm_index to 1.
pymc_input_s2_pre.loc[:, pd.IndexSlice["CSp_index", :]] = pymc_input_s2_pre.loc[
    :, pd.IndexSlice["CSp_index", :]
].replace(0, 1)
pymc_input_s2_pre.loc[:, pd.IndexSlice["CSm_index", :]] = pymc_input_s2_pre.loc[
    :, pd.IndexSlice["CSm_index", :]
].replace(0, 1)


# Compute moving average for CS perception.

# Create a list for "CS+" and "CS-".
CS_per_s2_list = []

# Iterate through the relevant stimuli.
for CS in ["CS+", "CS-"]:
    # Select the rows with the specified stimulus, participant and Per_size columns.
    CS_per = data_s2[data_s2["stimulus"] == CS][["participant", "Per_size"]]
    # Add a trials column that is a the number of rows in each participant's data.
    CS_per["trials"] = CS_per.groupby("participant").cumcount() + 1
    # Convert the data to wide format, with participant as id and trials as columns.
    CS_per_wide = CS_per.pivot(index="participant", columns="trials", values="Per_size")

    # Ensure all possible trials are represented even if no observations exist for some.
    max_trials = CS_per["trials"].max()
    CS_per_wide = CS_per_wide.reindex(
        columns=range(1, max_trials + 1), fill_value=np.nan
    )

    # Create a CS_per_updatemean for CS+ and CS- by computing the moving average for
    # CS+ and CS- perception and add it to the CS_per_s2_list.
    CS_per_updatemean = np.zeros(CS_per_wide.shape)
    for i in range(CS_per_wide.shape[0]):
        for j in range(CS_per_wide.shape[1]):
            valid_data = CS_per_wide.iloc[i, : j + 1].dropna()
            if len(valid_data) > 0:
                CS_per_updatemean[i, j] = valid_data.sum() / (j + 1)
            else:
                CS_per_updatemean[i, j] = 0.0

    CS_per_s2_list.append(CS_per_updatemean)

# Compute distance to "CS+" and "CS-".

# Create empty matrices to store perceptual and physical distance data.
d_list_s2 = {
    "d_per_p": np.zeros((40, 180)),
    "d_per_m": np.zeros((40, 180)),
    "d_phy_p": np.zeros((40, 180)),
    "d_phy_m": np.zeros((40, 180)),
    "participant": [],
    "trials": [],
}

# Compute perceptual and physical distances for
# both CS+ and CS- for each participant and trial.
for i in range(40):
    for j in range(180):
        # Calculate difference between Sper and CS_per_s2_list.
        d_list_s2["d_per_p"][i, j] = round(
            abs(
                pymc_input_s2_pre.loc[i + 1, ("Sper", j + 1)]
                - CS_per_s2_list[0][
                    i, int(pymc_input_s2_pre.loc[i + 1, ("CSp_index", j + 1)] - 1)
                ]
            ),
            2,
        )
        # Calculate difference between Sper and CS_per_s2_list.
        d_list_s2["d_per_m"][i, j] = round(
            abs(
                pymc_input_s2_pre.loc[i + 1, ("Sper", j + 1)]
                - CS_per_s2_list[1][
                    i, int(pymc_input_s2_pre.loc[i + 1, ("CSm_index", j + 1)] - 1)
                ]
            ),
            2,
        )
        # Calculate difference between Sphy and CSphy_p.
        d_list_s2["d_phy_p"][i, j] = round(
            abs(
                pymc_input_s2_pre.loc[i + 1, ("Sphy", j + 1)]
                - pymc_input_s2_pre.loc[i + 1, ("CSphy_p", j + 1)]
            ),
            2,
        )
        # Calculate difference between Sphy and CSphy_m.
        d_list_s2["d_phy_m"][i, j] = round(
            abs(
                pymc_input_s2_pre.loc[i + 1, ("Sphy", j + 1)]
                - pymc_input_s2_pre.loc[i + 1, ("CSphy_m", j + 1)]
            ),
            2,
        )
        d_list_s2["participant"].append(i + 1)
        d_list_s2["trials"].append(j + 1)


# PyMC input data generator function for experiment 2: differential conditioning.
def data_input_s2(L, indicator):
    """
    Prepare input data for PyMC analysis in the context of differential conditioning.

    This function processes and formats the data required for PyMC analysis
    based on the parameters provided. It handles US expectancy (`y_data`),
    perceptual distances (`d_p_per` and `d_m_per`), physical distances
    (`d_p_phy` and `d_m_phy`), and optionally, reinforcement (`r_plus` and
    `r_minus`) and indicator (`k_plus` and `k_minus`) data.

    Args:
        L (int):
            Specifies the type of data to handle:
            - 1: Use data for trials 24–180 only (excluding learning trials).
            - 2: Use the full dataset, including learning trials.
        indicator (int, optional):
            Specifies which CS indicator to use (required if `L == 2`):
            - 1: Use `CSindicator1_p` and `CSindicator1_m`.
            - 2: Use `CSindicator2_p` and `CSindicator2_m`.
            - None: Skip indicator handling.

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
    # Handle US expectancy, perceptual distances and physical distances data.
    if L == 1:
        y_data = (
            pymc_input_s2_pre.loc[:, pd.IndexSlice["y", 25:180]]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy()
        )

        d_p_per_data = d_list_s2["d_per_p"][:, 24:180].astype(np.float64)
        d_p_phy_data = d_list_s2["d_phy_p"][:, 24:180].astype(np.float64)
        d_m_per_data = d_list_s2["d_per_m"][:, 24:180].astype(np.float64)
        d_m_phy_data = d_list_s2["d_phy_m"][:, 24:180].astype(np.float64)
    else:
        y_data = pymc_input_s2_pre["y"].apply(pd.to_numeric, errors="coerce").to_numpy()

        d_p_per_data = d_list_s2["d_per_p"].astype(np.float64)
        d_p_phy_data = d_list_s2["d_phy_p"].astype(np.float64)
        d_m_per_data = d_list_s2["d_per_m"].astype(np.float64)
        d_m_phy_data = d_list_s2["d_phy_m"].astype(np.float64)

    data = {
        "Nparticipants": y_data.shape[0],
        "Ntrials": y_data.shape[1],
        "Nactrials": 24,
        "d_p_per": d_p_per_data,
        "d_p_phy": d_p_phy_data,
        "d_m_per": d_m_per_data,
        "d_m_phy": d_m_phy_data,
        "y": y_data,
    }

    # Handle r and k data.
    if L == 2:
        data["r_plus"] = (
            pymc_input_s2_pre["US_p"].apply(pd.to_numeric, errors="coerce").to_numpy()
        )

        data["r_minus"] = (
            pymc_input_s2_pre["US_m"].apply(pd.to_numeric, errors="coerce").to_numpy()
        )

        data["k_plus"] = (
            pymc_input_s2_pre[f"CSindicator{indicator}_p"]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy()
        )

        data["k_minus"] = (
            pymc_input_s2_pre[f"CSindicator{indicator}_m"]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy()
        )

    return data


# Generate and save PyMC input data for experiment 2: differential conditioning.
# PyMC input data without learning trials.
pickle.dump(data_input_s2(1, None), open("../PYMC_input_data/Data2_PYMCinput_G.pkl", "wb"))
# PyMC input data with an assumption of non-continuous learning.
pickle.dump(data_input_s2(2, 2), open("../PYMC_input_data/Data2_PYMCinput_LG.pkl", "wb"))
# PyMC input data with an assumption of continuous learning.
pickle.dump(data_input_s2(2, 1), open("../PYMC_input_data/Data2_PYMCinput_CLG.pkl", "wb"))

# Save long format data for experiment 2: differential conditioning.
pickle.dump(data_s2, open("../Preprocessed_data/Data_s2.pkl", "wb"))
