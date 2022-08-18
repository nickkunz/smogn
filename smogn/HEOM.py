import numpy as np


class HEOM():
    def __init__(self, X, cat_ix, nan_equivalents=[np.nan, 0], normalised="normal"):
        """ Heterogeneous Euclidean-Overlap Metric
        Distance metric class which initializes the parameters
        used in heom function
        
        Parameters
        ----------
        X : array-like of shape = [n_rows, n_features]
            Dataset that will be used with HEOM. Needs to be provided
            here because minimum and maximimum values from numerical
            columns have to be extracted
        
        cat_ix : array-like of shape = [cat_columns_number]
            List containing categorical feature indices
        
        cat_ix : array-like of shape = [x]
            List containing missing values indicators
        normalised: string
            normalises euclidan distance function for numerical variables
            Can be set as "std". Default is a column range
        Returns
        -------
        None
        """      

        self.nan_eqvs = nan_equivalents
        self.cat_ix = cat_ix
        self.col_ix = [i for i in range(X.shape[1])]
        # Get the normalization scheme for numerical variables
        if normalised == "std":
            self.range = 4* np.nanstd(X, axis = 0)
        else:
            self.range = np.nanmax(X, axis = 0) - np.nanmin(X, axis = 0)

    def heom(self, x, y):
        """ Distance metric function which calculates the distance
        between two instances. Handles heterogeneous data and missing values.
        It can be used as a custom defined function for distance metrics
        in Scikit-Learn
        
        Parameters
        ----------
        x : array-like of shape = [n_features]
            First instance 
            
        y : array-like of shape = [n_features]
            Second instance
        Returns
        -------
        result: float
            Returns the result of the distance metrics function
        """
        # Initialise results' array
        results_array = np.zeros(x.shape)

        # Get indices for missing values, if any
        nan_x_ix = np.flatnonzero( np.logical_or(np.isin(x, self.nan_eqvs), np.isnan(x)))
        nan_y_ix = np.flatnonzero( np.logical_or(np.isin(y, self.nan_eqvs), np.isnan(y)))
        nan_ix = np.unique(np.concatenate((nan_x_ix, nan_y_ix)))
        # Calculate the distance for missing values elements
        results_array[nan_ix] = 1

        # Get categorical indices without missing values elements
        cat_ix = np.setdiff1d(self.cat_ix, nan_ix)
        # Calculate the distance for categorical elements
        results_array[cat_ix]= np.not_equal(x[cat_ix], y[cat_ix]) * 1 # use "* 1" to convert it into int

        # Get numerical indices without missing values elements
        num_ix = np.setdiff1d(self.col_ix, self.cat_ix)
        num_ix = np.setdiff1d(num_ix, nan_ix)
        # Calculate the distance for numerical elements
        results_array[num_ix] = np.abs(x[num_ix] - y[num_ix]) / self.range[num_ix]

        # Return the final result
        # Square root is not computed in practice
        # As it doesn't change similarity between instances
        return np.sum(np.square(results_array))
