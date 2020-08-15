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
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

RES_CONST = 0.25
pd.set_option('display.max_columns', None)


world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
glakeData = gpd.read_file("LakeData.gpkg")
glakeData.dropna(inplace = True)
#print(lakeData.head())

#filter data to lakes under 60 degrees latitude
#glakeData = glakeData[glakeData['Pour_lat'] > 34]
glakeData = glakeData[glakeData['winter_tmp'] < -0.4]

#drop columns not used in log reg and rename columns
glakeData = glakeData.drop(['Lake_name', 'Country', 'Continent', 'Vol_src', 'winter_tmp', 'Res_time', 'Dis_avg', 'Shore_len', 'Shore_dev', 'Slope_100', 'Wshd_area'], axis = 1)

#print(glakeData.columns.values)

#glakeData = glakeData.rename(columns={'Res_time':'ResidenceTime_days', 'Dis_avg':'MeanDischarge_m3_sec', 'Shore_len':'ShorelineLength_km', 'Shore_dev':'ShorelineDevelopment', 'Slope_100':'Slope_degrees', 'Wshd_area':'WatershedArea_km2', 'Lake_area':'SurfaceArea_km2', 'Vol_total':'Volume_mcm', 'Pour_long':'Longitude_dd', 'Pour_lat':'Latitude_dd', 'Elevation':'Elevation_m','Depth_avg':'MeanDepth_m', 'mean_temp' : 'MeanAnnualAirTemp_c'})


glakeData = glakeData.rename(columns={'Depth_avg':'MeanDepth_m',  'Pour_lat': 'Latitude_dd', 'Lake_area':'SurfaceArea_km2','Pour_long':'Longitude_dd',  'Elevation':'Elevation_m', 'Vol_total':'Volume_mcm', 'mean_temp': 'MeanAnnualAirTemp_c'})


#feed data into logreg model
lakeData = glakeData.drop(['geometry'], axis = 1)

#lakeData = lakeData[['ResidenceTime_days', 'MeanDischarge_m3_sec','ShorelineLength_km', 'ShorelineDevelopment','Slope_degrees','WatershedArea_km2','SurfaceArea_km2','Volume_mcm', 'Longitude_dd', 'Latitude_dd', 'Elevation_m', 'MeanDepth_m', 'MeanAnnualAirTemp_c', 'tmn', 'tmx', 'temp_range']]


lakeData = lakeData[['Latitude_dd', 'Longitude_dd', 'Elevation_m', 'MeanAnnualAirTemp_c', 'SurfaceArea_km2', 'MeanDepth_m', 'Volume_mcm', 'temp_range', 'tmx', 'tmn']]

print(list(lakeData.columns.values))


rfc = joblib.load('grad.sav')

y_pred = rfc.predict(lakeData)
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
lakeData2['tmx'] = lakeData['tmx'] + 2
y_pred = rfc.predict(lakeData2)
classification = pd.DataFrame(data = y_pred, columns = ['intermittent'])
classifiedData = glakeData.copy()
classifiedData['intermittent'] = classification['intermittent'].values
intermittentLakes2 = classifiedData[(classifiedData['intermittent'] == 1)]
print(intermittentLakes2.shape)


lakeData3 = lakeData.copy()
lakeData3['MeanAnnualAirTemp_c'] = lakeData['MeanAnnualAirTemp_c'] + 3.2
lakeData3['tmn'] = lakeData['tmn'] + 3.2
lakeData3['tmx'] = lakeData['tmx'] + 3.2
y_pred = rfc.predict(lakeData3)
classification = pd.DataFrame(data = y_pred, columns = ['intermittent'])
classifiedData = glakeData.copy()
classifiedData['intermittent'] = classification['intermittent'].values
intermittentLakes3 = classifiedData[(classifiedData['intermittent'] == 1)]
print(intermittentLakes3.shape)


lakeData4 = lakeData.copy()
lakeData4['MeanAnnualAirTemp_c'] = lakeData['MeanAnnualAirTemp_c'] + 4.5
lakeData4['tmn'] = lakeData['tmn'] + 4.5
lakeData4['tmx'] = lakeData['tmx'] + 4.5
y_pred = rfc.predict(lakeData4)
classification = pd.DataFrame(data = y_pred, columns = ['intermittent'])
classifiedData = glakeData.copy()
classifiedData['intermittent'] = classification['intermittent'].values
intermittentLakes4 = classifiedData[(classifiedData['intermittent'] == 1)]
print(intermittentLakes4.shape)


lakeData5 = lakeData.copy()
lakeData5['MeanAnnualAirTemp_c'] = lakeData['MeanAnnualAirTemp_c'] + 8
lakeData5['tmn'] = lakeData['tmn'] + 8
lakeData5['tmx'] = lakeData['tmx'] + 8
y_pred = rfc.predict(lakeData5)
classification = pd.DataFrame(data = y_pred, columns = ['intermittent'])
classifiedData = glakeData.copy()
classifiedData['intermittent'] = classification['intermittent'].values
intermittentLakes5 = classifiedData[(classifiedData['intermittent'] == 1)]
print(intermittentLakes5.shape)


ax = gplt.polyplot(world, projection=gplt.crs.NorthPolarStereo(), facecolor='whitesmoke', figsize = (15, 15))

gplt.pointplot(annualLakes, color = 'black', ax = ax, s = 0.5, label = 'Annual lakes')
gplt.pointplot(intermittentLakes, color = 'tab:orange', ax = ax, s = 0.5, label = 'Intermittent lakes')
lgnd = plt.legend(loc="lower left", scatterpoints=1, fontsize=18)
lgnd.legendHandles[0]._sizes = [100]
lgnd.legendHandles[1]._sizes = [100]
plt.savefig('currentLakeMapGrad.png', bbox_inches='tight')
plt.clf()

ax = gplt.polyplot(world, projection=gplt.crs.NorthPolarStereo(), facecolor='whitesmoke', figsize = (15, 15))

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
plt.savefig('warmingLakeMapGrad.png', bbox_inches='tight')
plt.clf()
