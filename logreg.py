import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
import geopandas as gpd
from shapely.geometry import Polygon
import statsmodels.api as sm

RES_CONST = 0.25
pd.set_option('display.max_columns', None)

tempData = pd.read_csv('ampData.csv')

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

dt = dt1.filter([
                    "MeanAnnualAirTemp_c", 'Latitude_dd', 'Longitude_dd',
                    "MaximumDepth_m",     
                   'MeanDepth_m',
                   'IntermittentIceCover'])

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

gdt.dropna(inplace = True)
gdt['IntermittentIceCover'] = gdt['IntermittentIceCover'].map({'Y': 1, 'N': 0})


#fill in later
gdt = gdt.drop(['tmx', 'tmn', 'lat', 'lon', 'index_right', 'geometry', 'Longitude_dd'], axis = 1)
dt = pd.DataFrame(gdt)
#print(dt.head())

X = dt.loc[:, dt.columns != 'IntermittentIceCover']
y = dt.loc[:, dt.columns == 'IntermittentIceCover']

os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
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

dt_vars=dt.columns.values.tolist()
y=['IntermittentIceCover']
X=[i for i in dt_vars if i not in y]

logreg = LogisticRegression()
rfe = RFE(logreg, 20)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)

cols=[ "MeanAnnualAirTemp_c", "MaximumDepth_m", 'Latitude_dd', 'temp_range']
#cols=[ "Elevation_m", "MeanAnnualAirTemp_c", "MaximumDepth_m", 'Latitude_dd']

X=os_data_X[cols]
y=os_data_y['IntermittentIceCover']

logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)
logit = LogisticRegression(max_iter=10000)
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

print(classification_report(y_test, y_pred))
'''
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC.png')
'''
