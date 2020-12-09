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

df = pd.read_csv('../data/gardner_mt_catastrophe_only_tubulin.csv', skiprows=9)
df = df.melt().dropna()
df['sort'] = df['variable'].apply(lambda x: int(x[:-3]))
df = df.sort_values('sort')

p = iqplot.ecdf(df, cats = 'variable', q = 'value',palette=['#c7e9b4','#7fcdbb','#41b6c4','#1d91c0','#225ea8'])
bokeh.io.show(p)