#%load_ext rpy2.ipython
# %matplotlib inline
import logging
logging.getLogger('fbprophet').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from fbprophet import Prophet


df = pd.read_csv('../examples/example_wp_peyton_manning.csv')
df['y'] = np.log(df['y'])
df.head()


m = Prophet()
m.fit(df)


future = m.make_future_dataframe(periods=365)
future.tail()


forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


m.plot(forecast)


m.plot_components(forecast)
plt.show()

