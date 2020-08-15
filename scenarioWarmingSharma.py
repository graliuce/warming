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
from sklearn.tree import DecisionTreeClassifier
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
glakeData = glakeData.drop(['Lake_name', 'Country', 'Continent', 'Vol_src', 'winter_tmp',
                            'Pour_long', 'Pour_lat', 'Lake_area', 'Vol_total', 'Res_time',
                            'Dis_avg', 'Shore_len', 'Wshd_area', 'Slope_100'], axis = 1)

#print(glakeData.columns.values)

glakeData = glakeData.rename(columns={'Elevation' : 'Elevation_m','Depth_avg' : 'MeanDepth_m', 'Shore_dev':'ShorelineDevelopment', 'mean_temp' : 'MeanAnnualAirTemp_c'})


#feed data into logreg model
lakeData = glakeData.drop(['geometry'], axis = 1)

lakeData = lakeData[['MeanAnnualAirTemp_c',  'MeanDepth_m', 'Elevation_m', 'ShorelineDevelopment']]

print(list(lakeData.columns.values))
print(lakeData.shape)

dt = joblib.load('treeSharmaTrain.sav')

y_pred = dt.predict(lakeData)
classification = pd.DataFrame(data = y_pred, columns = ['intermittent'])
classifiedData = glakeData.copy()
classifiedData['intermittent'] = classification['intermittent'].values
intermittentLakes = classifiedData[(classifiedData['intermittent'] == 1)]
annualLakes = classifiedData[(classifiedData['intermittent'] == 0)]
print(intermittentLakes.shape)
print(annualLakes.shape)

# tracing odd intermittent points at high latitudes
#print(intermittentLakes[(intermittentLakes['MeanAnnualAirTemp_c'] < 4)])

lakeData2 = lakeData.copy()
lakeData2['MeanAnnualAirTemp_c'] = lakeData['MeanAnnualAirTemp_c'] + 2
y_pred = dt.predict(lakeData2)
classification = pd.DataFrame(data = y_pred, columns = ['intermittent'])
classifiedData = glakeData.copy()
classifiedData['intermittent'] = classification['intermittent'].values
intermittentLakes2 = classifiedData[(classifiedData['intermittent'] == 1)]
print(intermittentLakes2.shape)


lakeData3 = lakeData.copy()
lakeData3['MeanAnnualAirTemp_c'] = lakeData['MeanAnnualAirTemp_c'] + 3.2
y_pred = dt.predict(lakeData3)
classification = pd.DataFrame(data = y_pred, columns = ['intermittent'])
classifiedData = glakeData.copy()
classifiedData['intermittent'] = classification['intermittent'].values
intermittentLakes3 = classifiedData[(classifiedData['intermittent'] == 1)]
print(intermittentLakes3.shape)


lakeData4 = lakeData.copy()
lakeData4['MeanAnnualAirTemp_c'] = lakeData['MeanAnnualAirTemp_c'] + 4.5
y_pred = dt.predict(lakeData4)
classification = pd.DataFrame(data = y_pred, columns = ['intermittent'])
classifiedData = glakeData.copy()
classifiedData['intermittent'] = classification['intermittent'].values
intermittentLakes4 = classifiedData[(classifiedData['intermittent'] == 1)]
print(intermittentLakes4.shape)


lakeData5 = lakeData.copy()
lakeData5['MeanAnnualAirTemp_c'] = lakeData['MeanAnnualAirTemp_c'] + 8
y_pred = dt.predict(lakeData5)
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
plt.savefig('currentLakeMapSharmaTrain.png', bbox_inches='tight')
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
plt.savefig('warmingLakeMapSharmaTrain.png', bbox_inches='tight')
plt.clf()