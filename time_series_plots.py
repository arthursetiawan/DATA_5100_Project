# Colab notebook here: https://colab.research.google.com/drive/1xjnoG_AlWY_lYDQkPIzPx-gnZsdlY88h?usp=sharing

## Run code from data_upload.py

# Import pandas, numpy
import pandas as pd
import numpy as np

# Download dataset
df = pd.read_csv('https://github.com/arthursetiawan/DATA_5100_Project/blob/3ef28a95388bcf920158a93f95f13e64e68d1bfc/Fremont_Bridge_Bicycle_Counter.csv?raw=true',index_col=0)

# Remove unnecessary columns
df = df.drop(columns=['Fremont Bridge East Sidewalk','Fremont Bridge West Sidewalk'])

# Change index to DatetimeIndex
df.index=pd.to_datetime(df.index)

# Replace empty data with quadratic interpolated data
df['Fremont Bridge Total'] = df['Fremont Bridge Total'].interpolate(method='quadratic')

# Change index to DatetimeIndex
df.index=pd.to_datetime(df.index)

## End code from data_upload.py

## Run code from data_date_indexing.py

# Summarize data on daily, monthly, yearly bases
df_daily = df.resample('D').sum()
df_weekly = df.resample('W-MON').sum()
df_monthly = df.resample('M').sum()
df_quarterly = df.resample('Q').sum()
df_yearly = df.resample('Y').sum()

## End code from data_date_indexing.py

# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Define plotting function to simplify plotting
def plot_df(df,x,y,title="",xlabel='Date',ylabel='Count',dpi=100):
  plt.figure(figsize=(16,5),dpi=dpi)
  plt.plot(x,y,color='tab:red')
  plt.gca().set(title=title,xlabel=xlabel,ylabel=ylabel)
  plt.show()
  
# Plot same plot for Hourly, Daily, Weekly, Monthly, Quarterly, and Yearly counts

# Hourly
plot_df(df,x=df.index,y=df['Fremont Bridge Total'],title='Hourly Bike Count on Fremont Bridge')

# Daily
plot_df(df_daily,x=df_daily.index,y=df_daily['Fremont Bridge Total'],title='Daily Bike Count on Fremont Bridge')

# Weekly
plot_df(df_weekly,x=df_weekly.index,y=df_weekly['Fremont Bridge Total'],title='Weekly Bike Count on Fremont Bridge')

# Monthly
plot_df(df_monthly,x=df_monthly.index,y=df_monthly['Fremont Bridge Total'],title='Monthly Bike Count on Fremont Bridge')

# Quarterly
plot_df(df_quarterly,x=df_quarterly.index,y=df_quarterly['Fremont Bridge Total'],title='Quarterly Bike Count on Fremont Bridge')

# Yearly
plot_df(df_yearly,x=df_yearly.index,y=df_yearly['Fremont Bridge Total'],title='Yearly Bike Count on Fremont Bridge')
