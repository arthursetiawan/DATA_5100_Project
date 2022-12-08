# Import necessary libraries
import darts
from darts import TimeSeries
from darts.models import ExponentialSmoothing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import data
df = pd.read_csv('https://github.com/arthursetiawan/DATA_5100_Project/blob/3ef28a95388bcf920158a93f95f13e64e68d1bfc/Fremont_Bridge_Bicycle_Counter.csv?raw=true',index_col=0)

# Remove unnecessary columns
df = df.drop(columns=['Fremont Bridge East Sidewalk','Fremont Bridge West Sidewalk'])

# Change index to DatetimeIndex
df.index=pd.to_datetime(df.index)

# Replace empty data with quadratic interpolated data
df['Fremont Bridge Total'] = df['Fremont Bridge Total'].interpolate(method='quadratic')

# Summarize data on monthly basis
df_monthly = df.resample('M').sum()

# Create a TimeSeries
series = TimeSeries.from_dataframe(df_monthly)

# Set aside the last 36 months (post-covid) as a validation series with prior months (pre-covid) as training data
train, val = series[:-36], series[-36:]

# Fit an exponential smoothing model, and make a (probabilistic) prediction over the validation series’ duration:
model = ExponentialSmoothing()
model.fit(train)
prediction = model.predict(len(val), num_samples=1000)

# Plot
fig, ax = plt.subplots(figsize=(24,10))
series.plot() #Series plot in black
prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95) #Prediction plot in blue
plt.xlabel('Date',fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15) 
plt.ylabel('Monthly Bicycle Traffic Count', fontsize=20)
plt.legend()
plt.legend(loc=2,fontsize=15)
plt.show()

#Reference
#Exponential Smoothing: https://www.statsmodels.org/dev/examples/notebooks/generated/exponential_smoothing.html 
