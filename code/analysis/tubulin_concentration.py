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
import bebi103
hv.extension('bokeh')
from bokeh.io import show
from bokeh.models import ColumnDataSource, Whisker,FactorRange
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
import panel as pn


bebi103.hv.set_defaults()

# load data
df1 = pd.read_csv('../data/gardner_mt_catastrophe_only_tubulin.csv', skiprows=9)

# Defining functions to compute liklihood and MLE GAMMA DIST

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

mle_gam_7 = mle_iid_gamma(np.array(df1['7 uM'].dropna()))
mle_gam_9 = mle_iid_gamma(np.array(df1['9 uM'].dropna()))
mle_gam_10 = mle_iid_gamma(np.array(df1['10 uM'].dropna()))
mle_gam_14 = mle_iid_gamma(np.array(df1['14 uM'].dropna()))

# conf ints
# Random seed
rg = np.random.default_rng(3252)

# Defining the Boostrapping functions

def draw_bs_sample(data):
    """Draw a bootstrap sample from a 1D data set."""
    return rg.choice(data, size=len(data))

def draw_bs_reps_mle(mle_fun, data, args=(), size=1, progress_bar=False):
    """Draw nonparametric bootstrap replicates of maximum likelihood estimator.
    """
    
    if progress_bar:
        iterator = tqdm.tqdm(range(size))
    else:
        iterator = range(size)

    return np.array([mle_fun(draw_bs_sample(data), *args) for _ in iterator])

# 95% confindence interval of boostrapped estimates of alpha and beta
bs_reps_7 = draw_bs_reps_mle(
    mle_iid_gamma, np.array(df1['7 uM'].dropna()), size=10000, progress_bar=True
)

bs_reps_9 = draw_bs_reps_mle(
    mle_iid_gamma, np.array(df1['9 uM'].dropna()), size=10000, progress_bar=True
)

bs_reps_10 = draw_bs_reps_mle(
    mle_iid_gamma, np.array(df1['10 uM'].dropna()), size=10000, progress_bar=True
)


bs_reps_12 = draw_bs_reps_mle(
    mle_iid_gamma, np.array(df1['12 uM'].dropna()), size=10000, progress_bar=True
)

bs_reps_14 = draw_bs_reps_mle(
    mle_iid_gamma, np.array(df1['14 uM'].dropna()), size=10000, progress_bar=True
)




conf_int_7 = np.percentile(bs_reps_7, [2.5, 97.5], axis=0)
conf_int_9 = np.percentile(bs_reps_9, [2.5, 97.5], axis=0)
conf_int_10 = np.percentile(bs_reps_10, [2.5, 97.5], axis=0)
conf_int_12 = np.percentile(bs_reps_12, [2.5, 97.5], axis=0)
conf_int_14 = np.percentile(bs_reps_14, [2.5, 97.5], axis=0)
