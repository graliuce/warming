import numpy as np
import pandas as pd
from datetime import datetime
import datetime
import xarray as xr

ds = xr.open_dataset('cru_ts4.04.1941.1950.tmn.dat.nc.gz')
tmnData = ds.to_dataframe()

ds = xr.open_dataset('cru_ts4.04.1941.1950.tmx.dat.nc.gz')
tmxData = ds.to_dataframe()

#clean temp data
tmnData.dropna(inplace=True)
tmnData = tmnData.drop(['stn'], axis = 1)

#convert temperature time series to datetime
tmnData = tmnData.reset_index(level=['lat', 'lon', 'time'])

tmnData['time'] = tmnData['time'].astype(str)
tmnData['time'] = pd.to_datetime(tmnData.time, format='%Y-%m-%d')

tmnData = tmnData.groupby(['lat', 'lon', tmnData.time.dt.year]).min()
tmnData = tmnData.reset_index(level=['lat', 'lon'])

#write to csv file
tmnData.to_csv('minTemp.csv', index=False)


#clean temp data
tmxData.dropna(inplace=True)
tmxData = tmxData.drop(['stn'], axis = 1)

#convert temperature time series to datetime
tmxData = tmxData.reset_index(level=['lat', 'lon', 'time'])

tmxData['time'] = tmxData['time'].astype(str)
tmxData['time'] = pd.to_datetime(tmxData.time, format='%Y-%m-%d')

tmxData = tmxData.groupby(['lat', 'lon', tmxData.time.dt.year]).max()
tmxData = tmxData.reset_index(level=['lat', 'lon'])
print(tmxData.head())
print(tmxData.dtypes)
#write to csv file
tmxData.to_csv('maxTemp.csv', index=False)

ampData = pd.merge(tmxData, tmnData, on=['lat', 'lon'], how = 'inner')

ampData['temp_range'] = ampData['tmx'] - ampData['tmn']
#ampData = ampData.drop(['tmx', 'tmn'])

ampData = ampData.groupby(['lat', 'lon']).mean()
ampData = ampData.reset_index(level=['lat', 'lon'])
print(ampData.head())
ampData.to_csv('ampDataSmall.csv', index = False)
