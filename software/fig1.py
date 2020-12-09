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

#Create arrays for labelled and unlaballed
labeled = np.array(df[df['labeled']==True]['time to catastrophe (s)'])
unlabeled = np.array(df[df['labeled']==False]['time to catastrophe (s)'])

#plot ecdfs of Time to Catastrophe
p1 = iqplot.ecdf(
        data =df[df['labeled']==True],q='time to catastrophe (s)',palette=['#386cb0'],conf_int=True, title = 'ECDF of Time to Catastrophe for Labeled and Unlabeled Tubulin')
p2 = iqplot.ecdf(
        data =df[df['labeled']==False],q='time to catastrophe (s)',p=p1,palette=['#ffff99'],conf_int=True)
bokeh.io.show(p1)
