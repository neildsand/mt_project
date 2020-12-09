# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---


import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats as st
from scipy import integrate
import tqdm

import bebi103
import iqplot
import bokeh.io
import bokeh.plotting
import itertools
import warnings
import holoviews as hv
bokeh.io.output_notebook()
try:
    import multiprocess
except:
    import multiprocessing as multiprocess

import holoviews as hv
import panel as pn
hv.extension('bokeh')

bebi103.hv.set_defaults()

# Importing data
df_cat = pd.read_csv('../data/gardner_time_to_catastrophe_dic_tidy.csv')

# Filtering for labeled data
cat_lab=np.array(df_cat[df_cat['labeled']==True]['time to catastrophe (s)'])

# Defining functions to compute liklihood and MLE for the Gamma distributed model

def log_like_iid_gamma(params,y):
    """Log likelihood for i.i.d. Gamma distributed measurements"""
    alpha, beta = params

    if alpha <= 0 or beta <= 0 :
        
        return -np.inf

    return np.sum(st.gamma.logpdf(y,alpha,loc=0,scale=1/beta))

def mle_iid_gamma(y):
    """Perform maximum likelihood estimates for parameters for i.i.d.
    Gamma distributed measurements, parametrized by alpha and beta"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        res = scipy.optimize.minimize(
            fun=lambda params, y: -log_like_iid_gamma(params, y),
            x0=np.array((.5,.5)),
            args=(y),
            method='Powell'
        )

    if res.success:
        return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message)
        
        
# Compute MLE
mle = mle_iid_gamma(cat_lab)

# Compute confidence Intervals
# Random seed
rg = np.random.default_rng(3252)

# Defining the Boostrapping functions

def draw_bs_sample(data):
    """Draw a bootstrap sample from a 1D data set."""
    return rg.choice(data, size=len(data))

def draw_bs_reps_mle(mle_fun, data, args=(), size=1, progress_bar=False):
    """Draw nonparametric bootstrap replicates of maximum likelihood estimator.

    Parameters
    ----------
    mle_fun : function
        Function with call signature mle_fun(data, *args) that computes
        a MLE for the parameters
    data : one-dimemsional Numpy array
        Array of measurements
    args : tuple, default ()
        Arguments to be passed to `mle_fun()`.
    size : int, default 1
        Number of bootstrap replicates to draw.
    progress_bar : bool, default False
        Whether or not to display progress bar.

    Returns
    -------
    output : numpy array
        Bootstrap replicates of MLEs.
    """
    if progress_bar:
        iterator = tqdm.tqdm(range(size))
    else:
        iterator = range(size)

    return np.array([mle_fun(draw_bs_sample(data), *args) for _ in iterator])

# 95% confindence interval of boostrapped estimates of alpha and beta
bs_reps_cat_lab = draw_bs_reps_mle(
    mle_iid_gamma, cat_lab, size=10000, progress_bar=True
)
conf_int_cat_lab = np.percentile(bs_reps_cat_lab, [2.5, 97.5], axis=0)



# Poisson Model 
# Defining functions to compute liklihood and MLE 

def log_like_iid_exp(params,y):
    """Log likelihood for i.i.d. Exponentially distributed measurements"""
    beta1, d_beta = params

    if beta1 <= 0 or d_beta <= 0 :
        return -np.inf
    
    #account for limit as dbeta -> 0
    if d_beta < 0.00001:
        log_like_iid = np.log(beta1**2*y*np.exp(-beta1*y))
    
    else:
        log_like_iid = np.log(beta1 * ((beta1 + d_beta)/d_beta)) - beta1*y + np.log(1 - np.exp(-d_beta*y))    

    return np.sum(log_like_iid)
#consider when b2 and b1 are close
#look at limit of eq when dbeta goes to 0

def mle_iid_exp(y):
    """Perform maximum likelihood estimates for parameters for i.i.d.
    Exponentially distributed measurements, parametrized by beta1 and beta2"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        res = scipy.optimize.minimize(
            fun=lambda params, y: -log_like_iid_exp(params, y),
            x0=np.array((0.005463,.005463*0.4)), # initial guesses
            args=(y),
            method='Powell',
            tol = 1e-8
        )

    if res.success:
        return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message)
        
# compute mle
mle_exp = mle_iid_exp(cat_lab)


# compute confidence intervals
# Random seed
rg = np.random.default_rng(3252)

# Defining the Boostrapping functions

def draw_bs_sample(data):
    """Draw a bootstrap sample from a 1D data set."""
    return rg.choice(data, size=len(data))

def draw_bs_reps_mle(mle_fun, data, args=(), size=1, progress_bar=False):
    """Draw nonparametric bootstrap replicates of maximum likelihood estimator.

    Parameters
    ----------
    mle_fun : function
        Function with call signature mle_fun(data, *args) that computes
        a MLE for the parameters
    data : one-dimemsional Numpy array
        Array of measurements
    args : tuple, default ()
        Arguments to be passed to `mle_fun()`.
    size : int, default 1
        Number of bootstrap replicates to draw.
    progress_bar : bool, default False
        Whether or not to display progress bar.

    Returns
    -------
    output : numpy array
        Bootstrap replicates of MLEs.
    """
    if progress_bar:
        iterator = tqdm.tqdm(range(size))
    else:
        iterator = range(size)

    return np.array([mle_fun(draw_bs_sample(data), *args) for _ in iterator])

# Confindence interval of boostrapped estimates of beta1 and delta_beta
bs_reps_exp = draw_bs_reps_mle(
    mle_iid_exp, cat_lab, size=10000, progress_bar=True
)
conf_int_exp = np.percentile(bs_reps_exp, [2.5, 97.5], axis=0)