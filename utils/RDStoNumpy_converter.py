import numpy as np
import os
import pickle 
os.environ["R_HOME"] = r"C:\Program Files\R\R-4.4.3"
import rpy2.robjects as ro
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter

def r_array_to_np(r_obj):
    """
    Convert an R object with possible dim attributes into a NumPy array
    with the same shape. 
    If there's no 'dim', produce a 1D array.
    If length=1 => we return a Python scalar (convert near integers to int).
    """
    if ro.r["is.null"](ro.r["dim"](r_obj))[0]:
        # No dimension attribute
        length = ro.r["length"](r_obj)[0]
        arr = np.array(r_obj)

        # Force numeric arrays to float so we can handle missing as NaN
        if arr.dtype.kind in ('i','u','f'):
            arr = arr.astype(float, copy=False)
            # Replace R's NA_integer_ sentinel with np.nan
            mask = (arr == -2147483648.0)
            if np.any(mask):
                arr[mask] = np.nan

        if length == 1:
            # single scalar
            scalar_val = arr.item()
            # If it's a float close to an integer, convert to int
            if (isinstance(scalar_val, float) 
                and abs(scalar_val - round(scalar_val)) < 1e-12):
                scalar_val = int(round(scalar_val))
            return scalar_val
        else:
            # 1D array of length>1
            return arr
    else:
        # Has dim => multi-dimensional
        dims = ro.r["dim"](r_obj)
        shape = tuple(dims)
        arr = np.array(r_obj)
        # Force numeric arrays to float
        if arr.dtype.kind in ('i','u','f'):
            arr = arr.astype(float, copy=False)
            # Replace sentinel
            mask = (arr == -2147483648.0)
            if np.any(mask):
                arr[mask] = np.nan

        # R is typically column-major, so we reshape with order='F'
        arr = arr.reshape(shape, order='F')
        return arr

def convert_r_list_to_python_dict(r_list):
    """
    Convert an rpy2 ListVector (R named list) into a Python dict,
    using r_array_to_np for each element to preserve shape and fix NA issues.
    """
    py_dict = {}
    r_names = list(r_list.names)
    for i, r_name in enumerate(r_names):
        if r_name is None:
            r_name = f"unnamed_{i}"
        with localconverter(default_converter):
            r_sub = r_list[i]
            py_val = r_array_to_np(r_sub)
        py_dict[r_name] = py_val
    return py_dict

def replace_string_NA_with_nan(obj):
    """
    Recursively replaces NA strings with np.nan in dicts/lists/ndarrays.
    """
    if isinstance(obj, dict):
        return {k: replace_string_NA_with_nan(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(replace_string_NA_with_nan(x) for x in obj)
    elif isinstance(obj, np.ndarray):
        if obj.dtype.kind in ('O','U'):
            arr = obj.copy()
            mask = (arr == 'NA')
            arr[mask] = np.nan
            try:
                arr = arr.astype(float)
            except ValueError:
                pass
            return arr
        return obj
    elif obj == 'NA':
        return np.nan
    return obj

def load_rds_as_pymc_dict(rds_path):
    """
    Read an RDS file into Python, preserving dimension shapes 
    and converting single near integer floats to int. 
    Also handle NA => np.nan, and fix integer NA sentinel -2147483648 to np.nan.
    """
    r_obj = ro.r['readRDS'](rds_path)
    raw_dict = convert_r_list_to_python_dict(r_obj)
    final_dict = replace_string_NA_with_nan(raw_dict)
    return final_dict


if __name__ == "__main__":

    # rds file
    rds_file = "Data/Data_s2.rds"

    # desired output
    pickle_file = "PYMC_input_data/Data_s2_1.pkl"

    # conversion
    data_rds = load_rds_as_pymc_dict(rds_file)

    # Inspect each array
    print("Final dictionary keys:", list(data_rds.keys()))
    for key, val in data_rds.items():
        if isinstance(val, np.ndarray):
            print(f"\nKey: {key}")
            print("Value type:", type(val))
            print("Shape:", val.shape)
            print("Dtype:", val.dtype)
            if val.ndim == 2:
                print("Topleft corner:\n", val[:5,:5])
            else:
                print("Head of array:", val[:5])
        else:
            # Scalar or 1D list
            print(f"\nKey: {key} type: {type(val)}, value: {val}")

    with open(pickle_file, "wb") as f:
        pickle.dump(data_rds, f)

    # Esee if any array is non-positive
    for key in ['d_p_per','d_m_per','d_p_phy','d_m_phy', 'y', 'r_plus', 'r_minus', 'k_plus', 'k_minus']:
        arr = data_rds.get(key, None)
        if isinstance(arr, np.ndarray):
            print(f"{key} min = {np.min(arr)}")
            print(f"{key} max = {np.max(arr)}")
            if np.any(arr <= 0):
                print(f"{key} has non positive values. Min = {arr.min()}")