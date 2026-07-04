import numpy as np


X = [1,10000, 2000, 50000]
def zscore_normalize_features(X):
    # find the mean of each column/ feature
    mu = np.mean(X, axis = 0)
    # find the standard deviation of each column/feature
    sigma = np.std(X, axis = 0)
    # element-ứie, subtract mu for that column from each ẽample, divide by std for that column
    X_norm = (X-mu) /sigma
    #
    return (X_norm, mu, sigma)
# check our work
# from sklear.preprocessing import scale
#scale(X_orig, axis = 0, with_mean = True, with_std = True, copy = True)