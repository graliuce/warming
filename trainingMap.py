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

dt = dt1.filter(['Latitude_dd', 'Longitude_dd', 'IntermittentIceCover'])
dt['IntermittentIceCover'] = dt['IntermittentIceCover'].map({'Y': 1, 'N': 0})

dt = gpd.GeoDataFrame(dt, geometry = gpd.points_from_xy(dt.Longitude_dd, dt.Latitude_dd))

annualLakes= dt[dt['IntermittentIceCover']==0]
print(annualLakes.shape)
intermittentLakes= dt[dt['IntermittentIceCover']==1]
print(intermittentLakes.shape)

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

ax = gplt.polyplot(world, projection=gplt.crs.NorthPolarStereo(), facecolor='whitesmoke', figsize = (15, 15))

gplt.pointplot(annualLakes, color = 'black', ax = ax, s = 10, label = 'Annual winter ice')
gplt.pointplot(intermittentLakes, color = 'tab:orange', ax = ax, s = 10, label = 'Intermittent winter ice')
lgnd = plt.legend(loc="lower left", scatterpoints=1, fontsize=18)
lgnd.legendHandles[0]._sizes = [100]
lgnd.legendHandles[1]._sizes = [100]
plt.savefig('trainingLakeMap.png', bbox_inches='tight')
plt.clf()
