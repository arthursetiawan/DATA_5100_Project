# Import pandas, numpy
import pandas as pd
import numpy as np

# Download dataset
df = pd.read_csv('https://github.com/arthursetiawan/DATA_5100_Project/blob/3ef28a95388bcf920158a93f95f13e64e68d1bfc/Fremont_Bridge_Bicycle_Counter.csv?raw=true',index_col=0)

# Remove unnecessary columns
df = df.drop(columns=['Fremont Bridge East Sidewalk','Fremont Bridge West Sidewalk'])

# Count how many empty data points in df
print(df.isnull().sum())

# Replace empty data with '0'
df.fillna(0,inplace=True)
print(df.isnull().sum())
