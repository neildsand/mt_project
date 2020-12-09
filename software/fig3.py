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

# Print MLE
print("MLE for the parameters of the Gamma model [alpha, beta]: {}".format(mle))


# Compute samples using the Gamma model
rg = np.random.default_rng()
alpha = mle[0]
beta = mle[1]
single_samples_gam = np.array(
    [rg.gamma(alpha, 1 /beta, size=len(cat_lab)) for _ in range(100000)]
)



# Defining functions to compute liklihood and MLE for Poisson Model

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

# Compute samples for Poisson Model
rg = np.random.default_rng()
beta1 = mle_exp[0]
d_beta = mle_exp[1]
beta2 = beta1+d_beta
single_samples_1 = np.array(
    [rg.exponential(1 /beta1, size=len(cat_lab)) for _ in range(100000)]
)
single_samples_2 = np.array(
    [rg.exponential(1 /beta2, size=len(cat_lab)) for _ in range(100000)]
)
single_samples = single_samples_1 + single_samples_2


# predictive ecdf using Gamma Model
# Use discrete=True for discrete data
p3 = bebi103.viz.predictive_ecdf(
    samples=single_samples, data=cat_lab, discrete=False, x_axis_label="time to catastrophe (s)", title = 'Poisson Model Predictive ECDF'
)


 # Difference between Predictive and Measured ECDF
p4 = bebi103.viz.predictive_ecdf(
    samples=single_samples, data=cat_lab,diff = 'ECDF', discrete=False, x_axis_label="time to catastrophe (s)", title = 'Poisson Model Difference Between Predictive and Measured ECDF'
)


# predictive ecdf using Poisson
p5 = bebi103.viz.predictive_ecdf(
    samples=single_samples_gam, data=cat_lab, discrete=False, x_axis_label="time to catastrophe (s)", title = 'Gamma Model Predictive ECDF'
)


# Difference between Predictive and Measured ECDF
p6 = bebi103.viz.predictive_ecdf(
    samples=single_samples_gam, data=cat_lab,diff = 'ecdf', discrete=False, x_axis_label="time to catastrophe (s)",title = 'Gamma Model Difference between Predictive and Measured ECDF'
)

row1 = pn.Row(pn.pane.Bokeh(p5), pn.Spacer(width=20), pn.pane.Bokeh(p3))
row2 = pn.Row(pn.pane.Bokeh(p6), pn.Spacer(width=20), pn.pane.Bokeh(p4))
col1 = pn.Column(pn.Spacer(height=25), row1, pn.Spacer(height=35), row2)
dashboard = pn.Row(col1, pn.Spacer(width=30))
dashboard.save('fig3.png')