import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, make_scorer, precision_recall_fscore_support, average_precision_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTE
#from sklearn.feature_selection import RFE
import geopandas as gpd
from shapely.geometry import Polygon
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import joblib
from sklearn.impute import SimpleImputer
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus
from pprint import pprint
from sklearn.pipeline import Pipeline, make_pipeline

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
dt = dt1.drop(['lakecode', 'lakename', 'continent', 'country', 'state', 'MaximumDepth_m', 'DistanceToCoast_km', 'MaximumDistanceToLand_km', 'JFMCloudCover_perc', 'JFMPrecipitation_mm', 'MeanDischarge_m3_sec', 'ResidenceTime_days', 'ShorelineLength_km', 'Slope_degrees', 'WatershedArea_km2', 'ShorelineDevelopment'], axis=1)

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
gdt = gdt.drop(['lat', 'lon', 'mean_temp', 'Longitude_dd','index_right', 'geometry'], axis = 1)
dt = pd.DataFrame(gdt)
#print(dt.head())
#print(dt.corr())
print(dt.columns.values)
dt['IntermittentIceCover'] = dt['IntermittentIceCover'].map({'Y': 1, 'N': 0})

dt = dt[ [ col for col in dt.columns if col != 'IntermittentIceCover' ] + ['IntermittentIceCover'] ]

imputer = SimpleImputer(missing_values=np.nan, strategy='median')
dt = pd.DataFrame(imputer.fit_transform(dt), columns = dt.columns)


X = dt.loc[:, dt.columns != 'IntermittentIceCover']
y = dt.loc[:, dt.columns == 'IntermittentIceCover']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=3)
columns = X_train.columns


# Maximum number of levels in tree
max_depth = [5, 10, 15, None]
max_leaf_nodes = [5, 10, 30, None]
class_weight = ['balanced', 'balanced_subsample', None]
min_samples_split = [2, 5, 10, 20]
# Create the random grid
random_grid = {
               'max_depth' : max_depth,
               'max_leaf_nodes' : max_leaf_nodes,
               'class_weight' : class_weight,
               'min_samples_split' : min_samples_split}
pprint(random_grid)

rfc = RandomForestClassifier(random_state=3)

rfc_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 100, cv = 4, verbose=3, random_state=3, n_jobs = -1, scoring = 'recall')
#rfc_random = GridSearchCV(estimator = rfc, param_grid = random_grid, cv = 4, verbose=3, n_jobs = -1)

# Train the model on training data
rfc_random.fit(X_train, y_train.values.ravel())

print("\n=== Best Parameters ===")
print(rfc_random.best_params_)

rfc = rfc_random.best_estimator_
'''
# Instantiate model
rfc = RandomForestClassifier(random_state = 4)

# Train the model on training data
rfc.fit(X_train, y_train.values.ravel())
'''
def evaluate(model, X, y):
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
    print(avg_score)
    print("\n")

    pr_auc_score = np.mean(pr_auc_array, axis=0)
    print("=== All PR AUC Scores ===")
    print(pr_auc_array)
    print("=== Mean PR AUC Score ===")
    print(pr_auc_score)

evaluate(rfc, X, y)


joblib.dump(rfc, 'rfc.sav')


# Get numerical feature importances
importances = list(rfc.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(columns, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

'''
#Plot feature importance
# Set the style
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, columns, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');
plt.savefig('featureImportance2.png', bbox_inches='tight')
'''

'''
base_model = RandomForestClassifier(n_estimators = 10, random_state = 3)
base_model.fit(X_train, y_train.values.ravel())
evaluate(base_model, X, y)
'''

