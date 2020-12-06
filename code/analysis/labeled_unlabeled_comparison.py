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


# load data and set up arrays
df = pd.read_csv('../data/gardner_time_to_catastrophe_dic_tidy.csv')

#Create arrays for labelled and unlaballed
labeled = np.array(df[df['labeled']==True]['time to catastrophe (s)'])
unlabeled = np.array(df[df['labeled']==False]['time to catastrophe (s)'])


# compute mles and confidence intervals for time to catastrophe
def draw_bs_sample(data):
    """Draw a bootstrap sample from a 1D data set."""
    return np.random.choice(data, size=len(data))

def draw_bs_reps_mean(data, size=1):
    """Draw boostrap replicates of the mean from 1D data set."""
    out = np.empty(size)
    for i in range(size):
        out[i] = np.mean(draw_bs_sample(data))
    return out

#Bootstrapping the data
bs_reps_mean_lab = draw_bs_reps_mean(df[df['labeled']==True]['time to catastrophe (s)'], size=10000)
bs_reps_mean_unlab = draw_bs_reps_mean(df[df['labeled']==False]['time to catastrophe (s)'], size=10000)


# 95% confidence intervals
mean_labelled_conf_int = np.percentile(bs_reps_mean_lab, [2.5, 97.5])
mean_unlabelled_conf_int = np.percentile(bs_reps_mean_unlab, [2.5, 97.5])


print("""
Labeled mean time to catastrophe 95% confidence interval:   [{0:0.2f}, {1:0.2f}]
""".format(*(tuple(mean_labelled_conf_int))))
print("""
Unlabeled mean time to catastrophe 95% confidence interval:   [{0:0.2f}, {1:0.2f}]
""".format(*(tuple(mean_unlabelled_conf_int))))



# Difference of the Means hypothesis Test
@numba.njit
def draw_perm_sample(x, y):
    """Generate a permutation sample."""
    concat_data = np.concatenate((x, y))
    np.random.shuffle(concat_data)

    return concat_data[:len(x)], concat_data[len(x):]

@numba.njit
def draw_perm_reps_diff_mean(x, y, size=1):
    """Generate array of permuation replicates."""
    out = np.empty(size)
    for i in range(size):
        x_perm, y_perm = draw_perm_sample(x, y)
        out[i] = np.mean(x_perm) - np.mean(y_perm)

    return out

diff_mean = np.mean(unlabeled) - np.mean(labeled)

# Draw replicates
perm_reps = draw_perm_reps_diff_mean(unlabeled, labeled, size=10000000)

# Compute p-value
p_val = np.sum(perm_reps >= diff_mean) / len(perm_reps)

print('p-value =', p_val)
