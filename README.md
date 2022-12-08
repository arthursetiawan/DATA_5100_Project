#  The Future of Cycling in Seattle: A Data Science Approach to Ridership Forecasting

This project is a part of the DATA 5100: Foundations of Data Science class in Fall 2022 at Seattle University. 
We will be focusing on extracting and comprehending bicycle traffic over the Fremont bridge in Seattle. This dataset is provided 
by the Seattle Department of Transportation where cumulative observations are made every hour.
Through exploratory data analysis, we will observe trends in the data and attempt to use Python to extrapolate on the dataset to forecast future ridership on the Fremont bridge.

**This repository is currently split into 3 main branches:**

**Data Upload:**
Our dataset is based on the data provided by the Seattle Department of Transportation (SDOT) live Fremont Bicycle Counter which is updated monthly:
https://www.seattle.gov/transportation/projects-and-programs/programs/bike-program/bike-counters/fremont-bike-counters

**Data Exploration:**
In this section, we have pre-processed our data by removing unnecessary columns, filling in missing values, and renaming headers. This is covered in the 'data_date_indexing.py' file. Further to that, we performed exploraratory data analysis to understand trends and patterns in the existing data. We utilized the universal Python language and a mixture of libraries ranging from pandas, numpy, matplotlib, and seaborn. Time series plots are summarized in the 'time_series_plots.py' file. Boxplots showing statistical distributions are summarized by the 'boxplots.py' file. 

**Data Forecast:**
This branch includes our modeling phase of the project, where we utilized various time series modeling methods to fit and forecast the time series bicycle traffic data. Namely, we used Statsmodels and Facebook Prophet in our analysis. With Statsmodels, we were able to seasonally decompose the time series data with season-trend decomposition using LOESS (STL) where LOESS stands for LOcally Estimated Scatterplot Smoothing. From here, we were able to forecast using STLForecast using both ARIMA and Exponential Smoothing methods. With Prophet, we were able to apply a simple seasonal decomposition and forecasting using the method with minimal conditions. Additionally, we were able to apply the 'holiday' seasonality condition in Prophet to treat COVID-19 affected dates as outliers. This provided some fine-tuning of the model. Lastly, we attempted to use the TensorFlow library that utilizes neural networks for time series forecasting. Unfortunately, we were not able to make this system work for our modeling purposes. The files used are housed in this branch and include the Facebook Prophet code in the 'seasonal_decompose_prophet.py' file, the Statsmodels code in 'statsmodels_seasonal_forecast.py', and TensorFlow code in 'tensorflow.py'.

**References Source Code:**
FBProphet: 
STL Decomposition:
Tensor Flow: https://www.tensorflow.org/tutorials/structured_data/time_series
