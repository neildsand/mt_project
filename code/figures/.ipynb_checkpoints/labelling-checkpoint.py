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

import bokeh.io
bokeh.io.output_notebook()

df = pd.read_csv('../data/gardner_time_to_catastrophe_dic_tidy.csv')
df = df.replace(True, 'labeled')
df = df.replace(False, 'unlabeled')

#plot ecdfs of Time to Catastrophe
p1 = iqplot.ecdf(
        data =df,cats='labeled',q='time to catastrophe (s)',palette=['magenta','teal'],conf_int=True, title = 'ECDF of Time to Catastrophe for Labeled and Unlabeled Tubulin')

bokeh.io.show(p1)