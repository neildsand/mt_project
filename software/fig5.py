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

df1 = pd.read_csv('../data/gardner_mt_catastrophe_only_tubulin.csv', skiprows=9)

# Defining functions to compute liklihood and MLE for Gamma distribution

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

### Confidence Intervals

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

# Plot for alpha
vals = [mle_gam_7[0], mle_gam_9[0], mle_gam_10[0],mle_gam[0], mle_gam_14[0]]
upper = [conf_int_7[1][0], conf_int_9[1][0], conf_int_10[1][0], conf_int_12[1][0], conf_int_14[1][0]]
lower = [conf_int_7[0][0], conf_int_9[0][0], conf_int_10[0][0], conf_int_12[0][0], conf_int_14[0][0]]

source = ColumnDataSource(data=dict(cats=cats, vals=vals, upper=upper, lower=lower))

p = figure(x_range=cats, plot_height=400, title="Alpha MLEs with 95% Confidence Interval")


p.circle(x='cats', y = 'vals', size=10, source=source,
       line_color='white', fill_color=factor_cmap('cats', palette=['#c7e9b4','#7fcdbb','#41b6c4','#1d91c0','#225ea8'],
                                                  factors=cats))

# Create the coordinates for the errorbars
err_xs = []
err_ys = []
yerrs = np.subtract(upper,vals)
y2errs  = np.subtract(vals,lower)

for x, y, yerr, y2err in zip(cats, vals, yerrs, y2errs):
    err_xs.append((x, x))
    err_ys.append((y - y2err, y + yerr))

# Plot them
p.multi_line(err_xs, err_ys, color='red',line_width=1,line_cap='round')


p.xgrid.grid_line_color = None

#Plot for Beta
vals1 = [mle_gam_7[1], mle_gam_9[1], mle_gam_10[1],mle_gam[1], mle_gam_14[1]]
upper1 = [conf_int_7[1][1], conf_int_9[1][1], conf_int_10[1][1], conf_int_12[1][1], conf_int_14[1][1]]
lower1 = [conf_int_7[0][1], conf_int_9[0][1], conf_int_10[0][1], conf_int_12[0][1], conf_int_14[0][1]]

source1 = ColumnDataSource(data=dict(cats=cats, vals1=vals1, upper1=upper1, lower1=lower1))

p1 = figure(x_range=cats, plot_height=400, title="Beta MLEs with 95% Confidence Interval")

p1.circle(x='cats', y = 'vals1', size=10, source=source1,
       line_color='white', fill_color=factor_cmap('cats', palette=['#c7e9b4','#7fcdbb','#41b6c4','#1d91c0','#225ea8'],
                                                  factors=cats))

# Create the coordinates for the errorbars
err_xs1 = []
err_ys1 = []
yerrs1 = np.subtract(upper1,vals1)
y2errs1  = np.subtract(vals1,lower1)

for x, y, yerr, y2err in zip(cats, vals1, yerrs1, y2errs1):
    err_xs1.append((x, x))
    err_ys1.append((y - y2err, y + yerr))

# Plot them
p1.multi_line(err_xs1, err_ys1, color='red',line_width=1,line_cap='round')
p.yaxis.axis_label = "Alpha"
p1.yaxis.axis_label = "Beta"
p.xaxis.axis_label = "Tubulin Concentration"
p1.xaxis.axis_label = "Tubulin Concentration"
p1.xgrid.grid_line_color = None

# Create dashboard
row1 = pn.Row(pn.pane.Bokeh(p), pn.Spacer(width=20), pn.pane.Bokeh(p1))
col1 = pn.Column(pn.Spacer(height=25), row1)
dashboard = pn.Row(col1, pn.Spacer(width=30))
dashboard.save('fig5.png')