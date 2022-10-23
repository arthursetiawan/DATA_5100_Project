# Run code from data_date_indexing.py

# Change index to DatetimeIndex
df.index=pd.to_datetime(df.index)

# Summarize data on daily, monthly, yearly bases
df_daily = df.resample('D').sum()
df_weekly = df.resample('W-MON').sum()
df_monthly = df.resample('M').sum()
df_quarterly = df.resample('Q').sum()
df_yearly = df.resample('Y').sum()

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
