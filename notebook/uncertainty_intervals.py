# %load_ext rpy2.ipython
# %matplotlib inline
from fbprophet import Prophet
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import logging
logging.getLogger('fbprophet').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv('../examples/example_wp_peyton_manning.csv')
df['y'] = np.log(df['y'])
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=366)


forecast = Prophet(interval_width=0.95).fit(df).predict(future)


m = Prophet(mcmc_samples=300)
forecast = m.fit(df).predict(future)


m.plot_components(forecast)


prophet_plot_components(m, forecast)

plt.show()
