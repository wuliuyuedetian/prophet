# %load_ext rpy2.ipython
# %matplotlib inline
import logging
logging.getLogger('fbprophet').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt


df = pd.read_csv("../examples/example_wp_peyton_manning.csv")
print (type(df))
print (df.dtypes)
df["y"] = np.log(df["y"])
df.head()


m = Prophet()
m.fit(df)
#
# print("predict")
# 预测，将结果放入forecast数据框
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
print (forecast.columns)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


m.plot(forecast).show()


m.plot_components(forecast).show()
plt.show()



