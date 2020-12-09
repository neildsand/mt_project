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
from bokeh.plotting import figure, save
from bokeh.io import export_png
bokeh.io.output_notebook()

# Read in the data
df = pd.read_csv('../data/gardner_time_to_catastrophe_dic_tidy.csv')

# Create arrays for labelled and unlaballed
labeled = np.array(df[df['labeled']==True]['time to catastrophe (s)'])
unlabeled = np.array(df[df['labeled']==False]['time to catastrophe (s)'])

n=len(labeled)
m=len(unlabeled)

# Using definition from the problem statement
label_int = norm.interval(0.95, loc=np.mean(labeled), scale=np.std(labeled)/np.sqrt(n))
unlabel_int = norm.interval(0.95, loc=np.mean(unlabeled), scale=np.std(unlabeled)/np.sqrt(m))

print(" Confidence interval of the labeled data: {}".format(label_int))
print(" Confidence interval of the unlabeled data: {}".format(unlabel_int))

# plot confidence intervals
conf_ints = [
    dict(estimate=np.mean(labeled), conf_int=label_int, label="labeled"),
    dict(
        estimate=np.mean(unlabeled), conf_int=unlabel_int, label="unlabeled"
    ),
]

plot = bebi103.viz.confints(conf_ints, marker_kwargs={}, line_kwargs={}, palette=['magenta','teal'],x_axis_label='Time to Catastrophe (s)',title = '95% Conf Int for the Plug-in Estimate of the Mean Time to Catastrophe')

show(plot)