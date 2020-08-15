import fiona
import numpy as np
import pandas as pd 
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import descartes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

RES_CONST = 0.25
pd.set_option('display.max_columns', None)

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
lakes = gpd.read_file('HydroLAKES_points_v10.gdb.zip')

tempData = pd.read_csv('tempData.csv')
winterData = pd.read_csv('winterTemp.csv')

tempData = tempData.merge(winterData, on = ['lat', 'lon'])
print(tempData.head(-10))
print(tempData.describe())

#Create polygons for each grid in temp data for spatial join
geoms = []
for index, row in tempData.iterrows():
    ln = row.lon
    lt = row.lat
    geom = Polygon([(ln - RES_CONST, lt + RES_CONST), (ln + RES_CONST, lt + RES_CONST), ((ln + RES_CONST), (lt - RES_CONST)), (ln - RES_CONST, (lt - RES_CONST))])
    geoms.append(geom)

#create geodataframes to spatially merge datasets
gTempData = gpd.GeoDataFrame(tempData, geometry = gpd.points_from_xy(tempData.lon, tempData.lat))
gTempData['geometry'] = geoms 
lakes.crs = gTempData.crs
gdt = gpd.sjoin(lakes, gTempData, op = 'within')

gdt.dropna(inplace = True)

#fill in later
gdt = gdt.drop(['lat', 'lon', 'Lake_type', 'Hylak_id', 'Grand_id', 'Poly_src', 'Vol_res', 'index_right'], axis = 1)
#gdt = gdt.drop(['lat', 'lon', 'Lake_type', 'Grand_id', 'Vol_res', 'Vol_src', 'Poly_src', 'Hylak_id', 'Lake_name', 'index_right', 'geometry', 'Longitude_dd'], axis = 1)
print(gdt.head())
gdt.to_file("LakeData.gpkg", layer='lakes', driver="GPKG")

'''
base = world.plot(color='white', edgecolor='black')
lakes.plot(ax=base, marker = 'o', color = 'blue', markersize = .0001)

plt.savefig('lakemap.png')

'''
