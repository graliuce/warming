import fiona
import numpy as np
import pandas as pd 
import geopandas as gpd
import shapely
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import descartes
from sklearn.linear_model import LogisticRegression
import geoplot as gplt
import geoplot.crs as gcrs
import joblib

pd.set_option('display.max_columns', None)

#read in world map data and processed lake data (HydroLAKES merged with CRU temperature data)
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
glakeData = gpd.read_file("LakeData.gpkg")
glakeData.dropna(inplace = True)

#filter data to lakes with mean winter temperature of < -0.4 celsius
glakeData = glakeData[glakeData['winter_tmp'] < -0.4]

#drop columns not used in log reg and rename columns
glakeData = glakeData.drop(['Lake_name', 'Country', 'Continent', 'Vol_src', 'winter_tmp',
                            'Pour_long', 'Pour_lat', 'Lake_area', 'Vol_total', 'Res_time',
                            'Dis_avg', 'Shore_len', 'Wshd_area', 'Slope_100'], axis = 1)

glakeData = glakeData.rename(columns={'Elevation' : 'Elevation_m','Depth_avg' : 'MeanDepth_m', 'Shore_dev':'ShorelineDevelopment', 'mean_temp' : 'MeanAnnualAirTemp_c'})

lakeData = glakeData.drop(['geometry'], axis = 1)

lakeData = lakeData[['MeanAnnualAirTemp_c',  'MeanDepth_m', 'Elevation_m', 'ShorelineDevelopment', 'tmn']]

print(list(lakeData.columns.values))
print(lakeData.shape)


#Load model and classify lakes for different warming scenarios, print number of lakes in each category
dt = joblib.load('treeSharmaMinGood.sav')

y_pred = dt.predict(lakeData)
classification = pd.DataFrame(data = y_pred, columns = ['intermittent'])
classifiedData = glakeData.copy()
classifiedData['intermittent'] = classification['intermittent'].values
intermittentLakes = classifiedData[(classifiedData['intermittent'] == 1)]
annualLakes = classifiedData[(classifiedData['intermittent'] == 0)]
print(intermittentLakes.shape)
print(annualLakes.shape)


lakeData2 = lakeData.copy()
lakeData2['MeanAnnualAirTemp_c'] = lakeData['MeanAnnualAirTemp_c'] + 2
lakeData2['tmn'] = lakeData['tmn'] + 2
y_pred = dt.predict(lakeData2)
classification = pd.DataFrame(data = y_pred, columns = ['intermittent'])
classifiedData = glakeData.copy()
classifiedData['intermittent'] = classification['intermittent'].values
intermittentLakes2 = classifiedData[(classifiedData['intermittent'] == 1)]
print(intermittentLakes2.shape)


lakeData3 = lakeData.copy()
lakeData3['MeanAnnualAirTemp_c'] = lakeData['MeanAnnualAirTemp_c'] + 3.2
lakeData3['tmn'] = lakeData['tmn'] + 3.2
y_pred = dt.predict(lakeData3)
classification = pd.DataFrame(data = y_pred, columns = ['intermittent'])
classifiedData = glakeData.copy()
classifiedData['intermittent'] = classification['intermittent'].values
intermittentLakes3 = classifiedData[(classifiedData['intermittent'] == 1)]
print(intermittentLakes3.shape)


lakeData4 = lakeData.copy()
lakeData4['MeanAnnualAirTemp_c'] = lakeData['MeanAnnualAirTemp_c'] + 4.5
lakeData4['tmn'] = lakeData['tmn'] + 4.5
y_pred = dt.predict(lakeData4)
classification = pd.DataFrame(data = y_pred, columns = ['intermittent'])
classifiedData = glakeData.copy()
classifiedData['intermittent'] = classification['intermittent'].values
intermittentLakes4 = classifiedData[(classifiedData['intermittent'] == 1)]
print(intermittentLakes4.shape)


lakeData5 = lakeData.copy()
lakeData5['MeanAnnualAirTemp_c'] = lakeData['MeanAnnualAirTemp_c'] + 8
lakeData5['tmn'] = lakeData['tmn'] + 8
y_pred = dt.predict(lakeData5)
classification = pd.DataFrame(data = y_pred, columns = ['intermittent'])
classifiedData = glakeData.copy()
classifiedData['intermittent'] = classification['intermittent'].values
intermittentLakes5 = classifiedData[(classifiedData['intermittent'] == 1)]
print(intermittentLakes5.shape)

#Plot current intermittent and annual lakes
ax = gplt.polyplot(world, projection=gplt.crs.NorthPolarStereo(), facecolor = 'whitesmoke', figsize = (15, 15))

gplt.pointplot(annualLakes, color = 'black', ax = ax, s = 0.5, label = 'Annual lakes')
gplt.pointplot(intermittentLakes, color = 'tab:orange', ax = ax, s = 0.5, label = 'Intermittent lakes')
lgnd = plt.legend(loc="lower left", scatterpoints=1, fontsize=18)
lgnd.legendHandles[0]._sizes = [100]
lgnd.legendHandles[1]._sizes = [100]
plt.savefig('currentLakeMapSharmaMinGood.png', bbox_inches='tight')
plt.clf()

#Plot warming scenarioes
ax = gplt.polyplot(world, projection=gplt.crs.NorthPolarStereo(), facecolor = 'whitesmoke', figsize = (15, 15))

gplt.pointplot(annualLakes, color = 'black', ax = ax, s = 0.5, label = 'Annual lakes')
gplt.pointplot(intermittentLakes5, color = 'tab:red', ax = ax, s = 0.5, label = '8° warming')
gplt.pointplot(intermittentLakes4, color = 'tab:blue', ax = ax, s = 0.5, label = '4.5° warming')
gplt.pointplot(intermittentLakes3, color = 'yellow', ax = ax, s = 0.5, label = '3.2° warming' )
gplt.pointplot(intermittentLakes2, color = 'tab:purple', ax = ax, s = 0.5, label = '2° warming')
gplt.pointplot(intermittentLakes, color = 'tab:orange', ax = ax, s = 0.5, label = 'Intermittent lakes - current')
lgnd = plt.legend(loc="lower left", scatterpoints=1, fontsize=18)
lgnd.legendHandles[0]._sizes = [100]
lgnd.legendHandles[1]._sizes = [100]
lgnd.legendHandles[2]._sizes = [100]
lgnd.legendHandles[3]._sizes = [100]
lgnd.legendHandles[4]._sizes = [100]
lgnd.legendHandles[5]._sizes = [100]
plt.savefig('warmingLakeMapSharmaMinGood.png', bbox_inches='tight')
plt.clf()
