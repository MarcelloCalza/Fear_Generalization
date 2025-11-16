class _GroupWrapper:
    """
    Wrapper for a single xarray 'group' from an ArviZ InferenceData, which produces
    a reshaped NumPy array (collapsing 'chain' and 'draw' dims) for the 'group'.
    """
    def __init__(self, xr_dataset):
        self.xr_dataset = xr_dataset

    def __getitem__(self, var_name: str):
        xarr = self.xr_dataset[var_name]
        return xarr.values.reshape(-1, *xarr.values.shape[2:])


class ReshapedInferenceData:
    """
    Class to reshape chain/draw dimensions into a single dimension for multiple 'groups' 
    in an InferenceData object.
    """
    def __init__(self, idata):
        self.idata = idata
        if hasattr(idata, "posterior"):
            self.posterior = _GroupWrapper(idata.posterior)
        if hasattr(idata, "posterior_predictive"):
            self.posterior_predictive = _GroupWrapper(idata.posterior_predictive)
        if hasattr(idata, "prior"):
            self.posterior_predictive = _GroupWrapper(idata.prior)
