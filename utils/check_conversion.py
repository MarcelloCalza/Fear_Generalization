import pickle
import sys
import numpy as np
import os


os.environ["R_HOME"] = r"C:\Program Files\R\R-4.4.3"


np.set_printoptions(threshold=sys.maxsize)

import rpy2.robjects as ro
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter


# Load the Python preprocessed dictionary from a pickle file
data_py = pickle.load(open("PYMC_input_data/Sim_PYMCinput_CLG.pkl", "rb"))
print("Loaded Python dictionary keys:", data_py.keys())


def debug_array_difference_with_indices(arr_r, arr_py, rtol=1e-6, atol=1e-8, max_print=10):
    """
    Compare two arrays with np.isclose.
    If they differ:
      - Print shapes
      - Print how many elements differ
      - Print the first 'max_print' mismatches, including array indices
    """
    arr_r = np.asarray(arr_r, dtype=float)
    arr_py = np.asarray(arr_py, dtype=float)

    if arr_r.shape != arr_py.shape:
        print(f"Shape mismatch: {arr_r.shape} vs {arr_py.shape}")
        return

    # Build the mask of elements that fail the isclose check
    mask = ~np.isclose(arr_r, arr_py, rtol=rtol, atol=atol, equal_nan=True)

    num_diff = np.count_nonzero(mask)
    if num_diff == 0:
        print("match")
        return

    arr_diff = np.abs(arr_r - arr_py)
    max_diff = arr_diff[mask].max()

    print(f"number of elements differing: {num_diff}")
    print(f"max absolute difference: {max_diff:.6g}")

    # Indices of mismatched elements
    mismatch_indices = np.argwhere(mask)
    n_show = min(num_diff, max_print)

    print(f"showing first {n_show} mismatches (with their indices):")
    for i in range(n_show):
        idx = tuple(mismatch_indices[i])
        val_r = arr_r[idx]
        val_py = arr_py[idx]
        print(f"index={idx}: RDSconverted={val_r}, Preprocessed={val_py}")


python_dict_rds = pickle.load(open("PYMC_input_data/Sim_JAGSinput_CLG.pkl", "rb"))
common_keys = set(data_py.keys()).intersection(python_dict_rds.keys())
print("Common keys:", common_keys)

IGNORE_SHAPE = False

for k in sorted(common_keys):
    val_r = python_dict_rds[k]
    val_py = data_py[k]

    # Convert to float arrays
    arr_r = np.array(val_r, dtype=float, ndmin=1)
    arr_py = np.array(val_py, dtype=float, ndmin=1)

    if IGNORE_SHAPE:
        arr_r = arr_r.flatten()
        arr_py = arr_py.flatten()

    # Check shape
    if not IGNORE_SHAPE and arr_r.shape != arr_py.shape:
        print(f"{k}: differs (shape mismatch: {arr_r.shape} vs {arr_py.shape})")
        debug_array_difference_with_indices(arr_r, arr_py, max_print=5)
        continue

    # Compare
    if np.allclose(arr_r, arr_py, rtol=1e-6, atol=1e-8, equal_nan=True):
        print(f"{k}: matches")
    else:
        print(f"{k}: differs")
        debug_array_difference_with_indices(arr_r, arr_py, max_print=5)

print("check: Nparticipants")
print("From RDS convertion:", python_dict_rds.get("Nparticipants"))
print("From preprocessing:", data_py.get("Nparticipants"))