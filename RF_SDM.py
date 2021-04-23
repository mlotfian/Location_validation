import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

corvus_shuffle = pd.read_csv("C:/notes\Location_validation/new/eBird_data/21Oct/corvus_21Oct.csv")


#x is the landscape proportion values and y is the labels (present and absent)
x = corvus_shuffle.drop(columns=['TOD','Sampling_I','Locality_I','Scientific','number_obs','duration','distance','presabs','geometry'])
y = corvus_shuffle['presabs']

#Split data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)




clf2=RandomForestClassifier(n_estimators=2000)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf2.fit(x_train,y_train)
importance = clf2.feature_importances_

y_pred=clf2.predict(x_test)
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
RF_accu2 = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:",RF_accu2)
predicted_probs = clf2.predict_proba(x_test)
print(predicted_probs[1:3])

cm = confusion_matrix(y_test, y_pred)


# Save your model
import joblib
joblib.dump(clf2, 'model_RF'+'Corvus'+'.pkl')
print("Model dumped!")

# Load the model that you just saved
clf2 = joblib.load('model_RF.pkl')

# Saving the data columns from training
model_columns = list(x_train.columns)
print(model_columns)
joblib.dump(model_columns, 'model_columns_RF.pkl')
print("Models columns dumped!")
