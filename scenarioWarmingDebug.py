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

from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus


RES_CONST = 0.25
pd.set_option('display.max_columns', None)

tempData = pd.read_csv('tempData.csv')

infile1  ="https://pasta.lternet.edu/package/data/eml/edi/267/2/1c716f66bf3572a37a9f67035f9e02ac".strip() 
infile1  = infile1.replace("https://","http://")
                 
dt1 =pd.read_csv(infile1 
          ,skiprows=1
            ,sep=","  
                ,quotechar='"' 
           , names=[
                    "lakecode",     
                    "lakename",     
                    "continent",     
                    "country",     
                    "state",     
                    "IntermittentIceCover",     
                    "Latitude_dd",     
                    "Longitude_dd",     
                    "Elevation_m",     
                    "MeanAnnualAirTemp_c",     
                    "SurfaceArea_km2",     
                    "MeanDepth_m",     
                    "MaximumDepth_m",     
                    "Volume_mcm",     
                    "WatershedArea_km2",     
                    "ShorelineLength_km",     
                    "ResidenceTime_days",     
                    "MeanDischarge_m3_sec",     
                    "Slope_degrees",     
                    "ShorelineDevelopment",     
                    "JFMCloudCover_perc",     
                    "JFMPrecipitation_mm",     
                    "DistanceToCoast_km",     
                    "MaximumDistanceToLand_km"    ]
    )
# Coerce the data into the types specified in the metadata  
dt1.lakecode=dt1.lakecode.astype('category')  
dt1.lakename=dt1.lakename.astype('category')  
dt1.continent=dt1.continent.astype('category')  
dt1.country=dt1.country.astype('category')  
dt1.state=dt1.state.astype('category')  
dt1.IntermittentIceCover=dt1.IntermittentIceCover.astype('category') 
dt1.Latitude_dd=pd.to_numeric(dt1.Latitude_dd,errors='coerce') 
dt1.Longitude_dd=pd.to_numeric(dt1.Longitude_dd,errors='coerce') 
dt1.Elevation_m=pd.to_numeric(dt1.Elevation_m,errors='coerce') 
dt1.MeanAnnualAirTemp_c=pd.to_numeric(dt1.MeanAnnualAirTemp_c,errors='coerce') 
dt1.SurfaceArea_km2=pd.to_numeric(dt1.SurfaceArea_km2,errors='coerce') 
dt1.MeanDepth_m=pd.to_numeric(dt1.MeanDepth_m,errors='coerce') 
dt1.MaximumDepth_m=pd.to_numeric(dt1.MaximumDepth_m,errors='coerce') 
dt1.Volume_mcm=pd.to_numeric(dt1.Volume_mcm,errors='coerce') 
dt1.WatershedArea_km2=pd.to_numeric(dt1.WatershedArea_km2,errors='coerce') 
dt1.ShorelineLength_km=pd.to_numeric(dt1.ShorelineLength_km,errors='coerce') 
dt1.ResidenceTime_days=pd.to_numeric(dt1.ResidenceTime_days,errors='coerce') 
dt1.MeanDischarge_m3_sec=pd.to_numeric(dt1.MeanDischarge_m3_sec,errors='coerce') 
dt1.Slope_degrees=pd.to_numeric(dt1.Slope_degrees,errors='coerce') 
dt1.ShorelineDevelopment=pd.to_numeric(dt1.ShorelineDevelopment,errors='coerce') 
dt1.JFMCloudCover_perc=pd.to_numeric(dt1.JFMCloudCover_perc,errors='coerce') 
dt1.JFMPrecipitation_mm=pd.to_numeric(dt1.JFMPrecipitation_mm,errors='coerce') 
dt1.DistanceToCoast_km=pd.to_numeric(dt1.DistanceToCoast_km,errors='coerce') 
dt1.MaximumDistanceToLand_km=pd.to_numeric(dt1.MaximumDistanceToLand_km,errors='coerce') 
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))


dt = dt1.drop(['lakecode', 'lakename', 'continent', 'country', 'state', 'MaximumDepth_m', 'DistanceToCoast_km', 'MaximumDistanceToLand_km', 'JFMCloudCover_perc', 'JFMPrecipitation_mm', 'Elevation_m', 'SurfaceArea_km2', 'MeanDepth_m', 'MaximumDepth_m', 'Volume_mcm', 'WatershedArea_km2', 'ShorelineLength_km', 'ResidenceTime_days', 'MeanDischarge_m3_sec', 'Slope_degrees', 'ShorelineDevelopment', 'MeanAnnualAirTemp_c'], axis=1)

#Create polygons for each grid in temp data for spatial join
geoms = []
for index, row in tempData.iterrows():
    ln = row.lon
    lt = row.lat
    geom = Polygon([(ln - RES_CONST, lt + RES_CONST), (ln + RES_CONST, lt + RES_CONST), ((ln + RES_CONST), (lt - RES_CONST)), (ln - RES_CONST, (lt - RES_CONST))])
    geoms.append(geom)

#create geodataframes to spatially merge datasets
gCharData = gpd.GeoDataFrame(dt, geometry = gpd.points_from_xy(dt.Longitude_dd, dt.Latitude_dd))

gTempData = gpd.GeoDataFrame(tempData, geometry = gpd.points_from_xy(tempData.lon, tempData.lat))

gTempData['geometry'] = geoms 

gdt = gpd.sjoin(gCharData, gTempData, op = 'within')


#fill in later
gdt = gdt.drop(['lat', 'lon', 'index_right', 'geometry', 'Longitude_dd', 'mean_temp', 'tmn', 'tmx', 'temp_range'], axis = 1)
dt = pd.DataFrame(gdt)
#print(dt.head())

print(dt.columns.values)
dt['IntermittentIceCover'] = dt['IntermittentIceCover'].map({'Y': 1, 'N': 0})

dt = dt[ [ col for col in dt.columns if col != 'IntermittentIceCover' ] + ['IntermittentIceCover'] ]

imputer = SimpleImputer(missing_values=np.nan, strategy='median')
dt = pd.DataFrame(imputer.fit_transform(dt), columns = dt.columns)


X = dt.loc[:, dt.columns != 'IntermittentIceCover']
y = dt.loc[:, dt.columns == 'IntermittentIceCover']

os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=3)
columns = X_train.columns
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['IntermittentIceCover'])

# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of annual lakes",len(os_data_y[os_data_y['IntermittentIceCover']==0]))
print("Number of intermittent lakes",len(os_data_y[os_data_y['IntermittentIceCover']==1]))
print("Proportion of annual lakes in oversampled data is ",len(os_data_y[os_data_y['IntermittentIceCover']==0])/len(os_data_X))
print("Proportion of intermittent lakes in oversampled data is ",len(os_data_y[os_data_y['IntermittentIceCover']==1])/len(os_data_X))

X=os_data_X
y=os_data_y['IntermittentIceCover']
print(X.columns.values)
# Instantiate model with 1000 decision trees
#rfc = RandomForestClassifier()
rfc = DecisionTreeClassifier(max_depth=1)

# Train the model on training data
rfc.fit(X, y.values.ravel())

#joblib.dump(rfc, 'rfc.sav')

# Use the forest's predict method on the test data
predictions = rfc.predict(X_test)

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test.values.ravel(), predictions))
print('\n')

print("=== Classification Report ===")
print(classification_report(y_test.values.ravel(), predictions))
print('\n')

rfc_cv_score = cross_val_score(rfc, X_test, y_test.values.ravel(), cv=4, scoring='roc_auc')
print("=== All AUC Scores ===")
print(rfc_cv_score)
print('\n')

print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())

# Visualize tree
dot_data = StringIO()
export_graphviz(rfc, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = columns,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('treeLatitude.png')

glakeData = gpd.read_file("LakeData.gpkg")
glakeData.dropna(inplace = True)

#filter data to lakes under 60 degrees latitude
#glakeData = glakeData[glakeData['Pour_lat'] > 40]
glakeData = glakeData[glakeData['winter_tmp'] < -1]

#drop columns not used in log reg and rename columns
glakeData = glakeData.drop(['Lake_name', 'Country', 'Continent', 'Vol_src', 'Pour_long',
                             'Lake_area', 'Vol_total', 'Res_time', 'mean_temp',
                            'Elevation', 'Depth_avg', 'Dis_avg', 'Shore_len', 'Shore_dev', 
                            'Wshd_area', 'Slope_100', 'winter_tmp', 'tmx', 'tmn', 'temp_range'], axis = 1)

#print(glakeData.columns.values)

glakeData = glakeData.rename(columns={'Pour_lat' : 'Latitude_dd'})

glakeData = glakeData.sort_values('Latitude_dd')
#feed data into logreg model
lakeData = glakeData.drop(['geometry'], axis = 1)

lakeData = lakeData[['Latitude_dd']]

print(list(lakeData.columns.values))
print("glakeData")
print(glakeData.shape)
print(glakeData.iloc[ 0, : ]) #inter
print(glakeData.iloc[ 64, : ]) #inter
print(glakeData.iloc[ 65, : ]) #missing
print(glakeData.iloc[ 66, : ]) #missing

#rfc = joblib.load('rfc.sav')

y_pred = rfc.predict(lakeData)

print("y_pred")
print(y_pred[0]) #inter
print(y_pred[64]) #inter
print(y_pred[65]) #missing
print(y_pred[66]) #missing

#concat lake classification to glake df, add color and markersize column for plotting
classification = pd.DataFrame(data = y_pred, columns = ['intermittent'])
print("classification")
print(classification.shape)
print(classification.iloc[ 0, : ]) #inter
print(classification.iloc[ 64, : ]) #inter
print(classification.iloc[ 65, : ]) #missing
print(classification.iloc[ 66, : ]) #missing

classifiedData = glakeData.copy()
classifiedData['intermittent'] = classification['intermittent'].values
#classifiedData = glakeData.merge(classification, left_index=True, right_index=True)

print("classifiedData")
print(classifiedData.shape)
print(classifiedData.iloc[ 0, : ]) #inter
print(classifiedData.iloc[ 64, : ]) #inter
print(classifiedData.iloc[ 65, : ]) #missing
print(classifiedData.iloc[ 66, : ]) #missing

intermittentLakes = classifiedData[(classifiedData['intermittent'] == 1)]
annualLakes = classifiedData[(classifiedData['intermittent'] == 0)]
print(intermittentLakes.shape)
print(intermittentLakes.head())
print(intermittentLakes.describe())
print(annualLakes.shape)
print(annualLakes.head())
print(annualLakes.describe())


data = {'Latitude_dd':[28.518093, 28.527036, 28.527707, 28.005006, 28.008236, 28.008893]}
testData = pd.DataFrame(data)
testPred = rfc.predict(testData)
print(testPred)

lakeData2 = lakeData.copy()
#lakeData2['MeanAnnualAirTemp_c'] = lakeData['MeanAnnualAirTemp_c'] + 2
#lakeData2['tmn'] = lakeData['tmn'] + 2
#lakeData2['tmx'] = lakeData['tmx'] + 2
lakeData2['Latitude_dd'] = lakeData['Latitude_dd'] - 5
y_pred = rfc.predict(lakeData2)

#concat lake classification to glake df, add color and markersize column for plotting
classification = pd.DataFrame(data = y_pred, columns = ['intermittent'])
classifiedData = glakeData.merge(classification, left_index=True, right_index=True)

intermittentLakes2 = classifiedData[(classifiedData['intermittent'] == 1)]
print(intermittentLakes2.shape)


lakeData3 = lakeData.copy()
#lakeData3['MeanAnnualAirTemp_c'] = lakeData['MeanAnnualAirTemp_c'] + 3.2
#lakeData3['tmn'] = lakeData['tmn'] + 3.2
#lakeData3['tmx'] = lakeData['tmx'] + 3.2
lakeData3['Latitude_dd'] = lakeData['Latitude_dd'] - 10
y_pred = rfc.predict(lakeData3)

#concat lake classification to glake df, add color and markersize column for plotting
classification = pd.DataFrame(data = y_pred, columns = ['intermittent'])
classifiedData = glakeData.merge(classification, left_index=True, right_index=True)

intermittentLakes3 = classifiedData[(classifiedData['intermittent'] == 1)]
print(intermittentLakes3.shape)


lakeData4 = lakeData.copy()
#lakeData4['MeanAnnualAirTemp_c'] = lakeData['MeanAnnualAirTemp_c'] + 4.5
#lakeData4['tmn'] = lakeData['tmn'] + 4.5
#lakeData4['tmx'] = lakeData['tmx'] + 4.5
lakeData4['Latitude_dd'] = lakeData['Latitude_dd'] - 15
y_pred = rfc.predict(lakeData4)

#concat lake classification to glake df, add color and markersize column for plotting
classification = pd.DataFrame(data = y_pred, columns = ['intermittent'])
classifiedData = glakeData.merge(classification, left_index=True, right_index=True)

intermittentLakes4 = classifiedData[(classifiedData['intermittent'] == 1)]
print(intermittentLakes4.shape)


lakeData5 = lakeData.copy()
#lakeData5['MeanAnnualAirTemp_c'] = lakeData['MeanAnnualAirTemp_c'] + 8
#lakeData5['tmn'] = lakeData['tmn'] + 8
#lakeData5['tmx'] = lakeData['tmx'] + 8
lakeData5['Latitude_dd'] = lakeData['Latitude_dd'] - 20
y_pred = rfc.predict(lakeData5)

#concat lake classification to glake df, add color and markersize column for plotting
classification = pd.DataFrame(data = y_pred, columns = ['intermittent'])
classifiedData = glakeData.merge(classification, left_index=True, right_index=True)

intermittentLakes5 = classifiedData[(classifiedData['intermittent'] == 1)]
print(intermittentLakes5.shape)
print(intermittentLakes5.columns.values)

ax = gplt.polyplot(world, projection=gplt.crs.NorthPolarStereo(), figsize = (15, 15))

gplt.pointplot(annualLakes, color = 'black', ax = ax, s = 1)
gplt.pointplot(intermittentLakes, color = 'orange', ax = ax, s = 1)

plt.savefig('currentLakeMapLat2.png', bbox_inches='tight')
plt.clf()

ax = gplt.polyplot(world, projection=gplt.crs.NorthPolarStereo(), figsize = (15, 15))

gplt.pointplot(annualLakes, color = 'black', ax = ax, s = 1)
gplt.pointplot(intermittentLakes5, color = 'red', ax = ax, s = 1)
gplt.pointplot(intermittentLakes4, color = 'blue', ax = ax, s = 1)
gplt.pointplot(intermittentLakes3, color = 'yellow', ax = ax, s = 1)
gplt.pointplot(intermittentLakes2, color = 'purple', ax = ax, s = 1)
gplt.pointplot(intermittentLakes, color = 'orange', ax = ax, s = 1)

plt.savefig('warmingLakeMapLat2.png', bbox_inches='tight')
plt.clf()

