import numpy as np
import pandas as pd
from datetime import datetime
import datetime
import xarray as xr

ds = xr.open_dataset('cru_ts4.04.1901.2019.tmp.dat.nc.gz')
tmpData = ds.to_dataframe()


#clean temp data
tmpData.dropna(inplace=True)
tmpData = tmpData.drop(['stn'], axis = 1)

#convert temperature time series to datetime
tmpData = tmpData.reset_index(level=['lat', 'lon', 'time'])

tmpData['time'] = tmpData['time'].astype(str)
tmpData['time'] = pd.to_datetime(tmpData.time, format='%Y-%m-%d')

tmpData = tmpData[(tmpData.time.dt.month <= 3) | (tmpData.time.dt.month >= 12)]
tmpData = tmpData.rename(columns={'tmp' : 'winter_tmp'})
print(tmpData.head(20))
tmpData = tmpData.groupby(['lat', 'lon']).mean()
tmpData = tmpData.reset_index(level=['lat', 'lon'])

print(tmpData.head())
print(tmpData.describe())

#write to csv file
tmpData.to_csv('winterTemp.csv', index=False)
