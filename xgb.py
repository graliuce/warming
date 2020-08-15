import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, make_scorer, precision_recall_fscore_support, average_precision_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
import geopandas as gpd
from shapely.geometry import Polygon
import statsmodels.api as sm
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import joblib
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus
import xgboost as xgb
from xgboost.sklearn import XGBClassifier  

RES_CONST = 0.25
pd.set_option('display.max_columns', None)

# Read in CRU temperature data for min temp, max temp, and temp range
tempData = pd.read_csv('tempData.csv')

infile1  ="https://pasta.lternet.edu/package/data/eml/edi/267/2/1c716f66bf3572a37a9f67035f9e02ac".strip() 
infile1  = infile1.replace("https://","http://")
                 
# Read in lake characteristic data published by Sharma et al.
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
dt = dt1.drop(['lakecode', 'lakename', 'continent', 'country', 'state', 'MaximumDepth_m', 'DistanceToCoast_km', 'MaximumDistanceToLand_km', 'JFMCloudCover_perc', 'JFMPrecipitation_mm', 'MeanDischarge_m3_sec', 'ResidenceTime_days', 'ShorelineLength_km', 'Slope_degrees', 'WatershedArea_km2', 'ShorelineDevelopment'], axis=1)

#Create polygons for each grid in temp data for spatial join
geoms = []
for index, row in tempData.iterrows():
    ln = row.lon
    lt = row.lat
    geom = Polygon([(ln - RES_CONST, lt + RES_CONST), (ln + RES_CONST, lt + RES_CONST), ((ln + RES_CONST), (lt - RES_CONST)), (ln - RES_CONST, (lt - RES_CONST))])
    geoms.append(geom)

#create geodataframes to spatially merge lake characteristic and temperature data
gCharData = gpd.GeoDataFrame(dt, geometry = gpd.points_from_xy(dt.Longitude_dd, dt.Latitude_dd))

gTempData = gpd.GeoDataFrame(tempData, geometry = gpd.points_from_xy(tempData.lon, tempData.lat))

gTempData['geometry'] = geoms 

gdt = gpd.sjoin(gCharData, gTempData, op = 'within')

#drop columns that will not be used to train the model
gdt = gdt.drop(['lat', 'lon', 'mean_temp', 'index_right', 'geometry'], axis = 1)
dt = pd.DataFrame(gdt)
#print(dt.head())
#print(dt.corr())
print(dt.columns.values)

#Clean data by changing classes to 1 and 0
dt['IntermittentIceCover'] = dt['IntermittentIceCover'].map({'Y': 1, 'N': 0})

#fill missing data with 0 (xgboost defaults to 0 for missing data)
dt = dt.fillna(0)

#Split into test and training dataframes
X = dt.loc[:, dt.columns != 'IntermittentIceCover']
y = dt.loc[:, dt.columns == 'IntermittentIceCover']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=3)

columns = X_train.columns

#Create and train xgb model with tuned hyperparameters
model_xgb = XGBClassifier(objective = 'binary:logistic', missing = None, scale_pos_weight = 5, reg_lambda = 10, max_depth = 5, learning_rate = 0.1, gamma = 0, colsample_bytree = 0.5,  seed = 3)

model_xgb.fit(X_train, y_train)

'''
# tune hyperparameters
params = {
    "gamma": [0],
    "learning_rate": [0.1], # default 0.1 
    "max_depth": [5], # default 3
    "reg_lambda": [10],
    "scale_pos_weight": [5],
    "colsample_bytree" : [0.5] 
}


model_xgb = XGBClassifier(objective = 'binary:logistic', missing = None, seed = 3)

search = RandomizedSearchCV(model_xgb, param_distributions=params, random_state=42, n_iter=1, cv=3, verbose=1, n_jobs=1, return_train_score=True)

search.fit(X_train, y_train.values.ravel())


print("\n=== Best Parameters ===")
print(search.best_params_)

model_xgb = search.best_estimator_
'''

#model_xgb.fit(X_train, y_train, early_stopping_rounds=10, eval_metric = 'aucpr', eval_set = [(X_test, y_test)])

#Test the model on the test set and print performance metrics
y_pred = model_xgb.predict(X_test)

target_names = ['Annual', 'Intermittent']
print(classification_report(y_test, y_pred,target_names=target_names))

#Save the model 
joblib.dump(model_xgb, 'xgboost.sav')

#Evaulate the model with cross validate metrics (ROC AUC, PR AUC, Fscore, etc.)
def evaluate(model, X, y):

    print("=== Cross Validated Stats ===")
    rfc_cv_score = cross_val_score(model, X, y.values.ravel(), cv=4, scoring='roc_auc')
    print("=== All ROC AUC Scores ===")
    print(rfc_cv_score)
    print('\n')


    print("=== Mean ROC AUC Score ===")
    print("Mean AUC: %0.2f (+/- %0.2f)\n" % (rfc_cv_score.mean(), rfc_cv_score.std() * 2))
    kf = StratifiedKFold(n_splits=3, random_state=3, shuffle=True)

    score_array =[]
    pr_auc_array = []
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        clf = model.fit(X_train,y_train.values.ravel())
        y_pred = clf.predict(X_test)
        score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))
        pr_auc_array.append(average_precision_score(y_test, y_pred))

    avg_score = np.mean(score_array,axis=0)
    print("=== Average Precision, Recall, Fscore, Support (0, 1) ===")
    print(avg_score)
    print("\n")

    pr_auc_score = np.mean(pr_auc_array, axis=0)
    print("=== All PR AUC Scores ===")
    print(pr_auc_array)
    print("=== Mean PR AUC Score ===")
    print(pr_auc_score)

evaluate(model_xgb, X, y)
