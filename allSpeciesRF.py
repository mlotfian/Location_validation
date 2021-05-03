import pandas as pd
import numpy as np
import geopandas as gpd

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import csv

from imblearn.ensemble import BalancedRandomForestClassifier
import functools
import joblib
import fiona
from shapely.geometry import Polygon, mapping
import csv

def clip_points(targets,poly):
  test_target = targets.intersection(poly)
  test_target =pd.DataFrame(test_target)
  test_target = test_target.rename(columns={0: 'geometry'})
  to_remove_index =[]
  to_get_index =[]
  for index, row in test_target.iterrows():
    if row['geometry'].is_empty:
      to_remove_index.append(index)
  test_target = test_target.drop(to_remove_index)
  for index, row in test_target.iterrows():
    to_get_index.append(index)
  test_target = targets.loc[to_get_index, :]

  return test_target

def convert_to_gdp(df):
    from shapely import wkt
    df = gpd.GeoDataFrame(df)

    df['geometry'] = df['geometry'].apply(wkt.loads)
    df = df.set_geometry('geometry')
    return df

def combine_rfs(rf_a, rf_b):
    rf_a.estimators_ += rf_b.estimators_
    rf_a.n_estimators = len(rf_a.estimators_)
    return rf_a



evaluations = []

import csv

with open('C:/notes/Location_validation/API_Django/env/all_names.csv', newline='') as f:
    reader = csv.reader(f)
    all_names = [row for row in reader][0]

for sp_name in all_names:
    folds_summary_RF = []
    i = 1
    all_models =[]

    PRES_ABS = pd.read_csv("C:/notes/Location_validation/API_Django/env/pres_abs_env/" + sp_name +"_pres_abs_env.csv", index_col=0)
    PRES_ABS = gpd.GeoDataFrame(PRES_ABS, geometry=gpd.points_from_xy(PRES_ABS.lon, PRES_ABS.lat))

    spBlocks = pd.read_csv("C:/notes/Location_validation/API_Django/env/best_folds/" + sp_name +"_spFold.csv", index_col=0)
    spBlocks = convert_to_gdp(spBlocks)
    spBlocks = spBlocks.dissolve(by='grid_id')

    for index, row in spBlocks.iterrows():
        test_df = clip_points(PRES_ABS, row['geometry'])
        train_df = PRES_ABS.drop(test_df.index)
        to_test = (test_df.drop(columns=['lat','lon','geometry']))
        to_train = (train_df.drop(columns=['lat','lon','geometry']))
        x_train = to_train.drop(columns=['pres_abs'])
        #print(x_train.columns)
        y_train = to_train[['pres_abs']]
      #y_train['PRES_ABS'] = y_train['PRES_ABS'].map({'t': 1, 'f': 0})
        x_test = to_test.drop(columns=['pres_abs'])
        y_test = to_test[['pres_abs']]
      #y_test['PRES_ABS'] = y_test['PRES_ABS'].map({'t': 1, 'f': 0})

        train_0, train_1 = len(y_train[y_train['pres_abs']==0]), len(y_train[y_train['pres_abs']==1])
        test_0, test_1 = len(y_test[y_test['pres_abs']==0]), len(y_test[y_test['pres_abs']==1])

        #start training
        globals()["RF" + str(i)]=BalancedRandomForestClassifier(n_estimators=500)
        (globals()["RF" + str(i)]).fit(x_train,y_train)

        all_models.append(globals()["RF" + str(i)])

        y_pred=(globals()["RF" + str(i)]).predict(x_test)
        RF_accu = metrics.accuracy_score(y_test, y_pred)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
        RF_auc = metrics.auc(fpr, tpr)
        folds_summary_RF.append(
          {
            "species_name": sp_name,
            "model_num" : i,
            "No_presTrain":train_1,
            "No_absTrain":train_0,
            "No_presTest": test_1,
            "No_absTest":test_0,
          "ACC": RF_accu,
          "AUC":RF_auc

          }
      )

        i+=1
    eval = pd.DataFrame(folds_summary_RF)
    evaluations.append(eval)
    rf_combined = functools.reduce(combine_rfs, all_models)
    joblib.dump(rf_combined, "C:/notes/Location_validation/API_Django/RFmodels/" + sp_name + "_RFmodel.pkl", compress=3)

all_eval = pd.concat(evaluations)
all_eval.to_csv("C:/notes/Location_validation/API_Django/RF_evaluation/RFperformance.csv")




#model_columns = list(to_train.columns)
#joblib.dump(x_train, 'model_columns_RF.pkl')
