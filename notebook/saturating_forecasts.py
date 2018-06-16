# %load_ext rpy2.ipython
# %matplotlib inline
from fbprophet import Prophet
import pandas as pd
import logging
logging.getLogger('fbprophet').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

df = pd.read_csv('../examples/example_wp_R.csv')
import numpy as np
df['y'] = np.log(df['y'])


df['cap'] = 8.5


m = Prophet(growth='logistic')
m.fit(df)


future = m.make_future_dataframe(periods=1826)
future['cap'] = 8.5
fcst = m.predict(future)
m.plot(fcst)


df['y'] = 10 - df['y']
df['cap'] = 6
df['floor'] = 1.5
future['cap'] = 6
future['floor'] = 1.5
m = Prophet(growth='logistic')
m.fit(df)
fcst = m.predict(future)
m.plot(fcst)
plt.show()




