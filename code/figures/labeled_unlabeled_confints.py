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

# +
import numpy as np
import pandas as pd
import numba
from numba import jit
from scipy import stats
from scipy.stats import norm
import iqplot
import statsmodels.api as sm
from bokeh.plotting import figure, show
from bokeh.models import Legend
import bebi103

import bokeh.io
from bokeh.plotting import figure, output_file,save
bokeh.io.output_notebook()

df = pd.read_csv('../data/gardner_time_to_catastrophe_dic_tidy.csv')

#Create arrays for labelled and unlaballed
labeled = np.array(df[df['labeled']==True]['time to catastrophe (s)'])
unlabeled = np.array(df[df['labeled']==False]['time to catastrophe (s)'])

n=len(labeled)
m=len(unlabeled)

# Using definition from the problem statement
label_int = norm.interval(0.95, loc=np.mean(labeled), scale=np.std(labeled)/np.sqrt(n))
unlabel_int = norm.interval(0.95, loc=np.mean(unlabeled), scale=np.std(unlabeled)/np.sqrt(m))

# plot confidence intervals
conf_ints = [
    dict(estimate=np.mean(labeled), conf_int=label_int, label="labeled"),
    dict(
        estimate=np.mean(unlabeled), conf_int=unlabel_int, label="unlabeled"
    ),
]

bokeh.io.show(bebi103.viz.confints(conf_ints))
# -


