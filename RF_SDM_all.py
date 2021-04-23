import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import csv

all_data = pd.read_csv("C:/notes/Location_validation/API_Django/env/allBirdsEnv_pres_abs.csv")
with open("C:/notes/Location_validation/API_Django/env/all_sci_names.csv", newline='') as f:
    reader = csv.reader(f)
    all_names = list(reader)
all_names = [item for sublist in all_names for item in sublist]

for sp in all_names[1:]:
  print("Start data preparation for {sp}".format(sp=sp))
  df_filter = all_data.loc[all_data['scientificname']==sp]
  to_train = (df_filter.drop(columns=['scientificname','sampling_id','obs_date','id'])).drop_duplicates()

  num_pres = len(to_train[to_train['pres_abs'] == 't'])
  num_abs = len(to_train[to_train['pres_abs'] == 'f'])

  if num_pres<num_abs:
    pres_abs = to_train.groupby('pres_abs').apply(lambda x: x.sample(num_pres)).reset_index(drop = True)
    pres_abs = pres_abs.sample(frac=1)
  else:
    pres_abs = to_train.sample(frac=1)

  x = pres_abs.drop(columns=['pres_abs','geom'])
  y = pres_abs[['pres_abs']]
  print("Start training for {sp}".format(sp=sp))
  x_train, x_test,y_train, y_test = train_test_split(x,y, test_size=0.20,random_state=42)

  clf2=RandomForestClassifier(n_estimators=2000)
  clf2.fit(x_train,np.ravel(y_train))

  import joblib
  joblib.dump(clf2, 'model_RF_'+sp+'.pkl')
  print("Model dumped!")

  model_columns = list(x_train.columns)
  print(model_columns)
  joblib.dump(model_columns, 'model_columns_RF.pkl')
  print("Models columns dumped!")
