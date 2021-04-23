# Dependencies
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np


#from utils import createZone
from utils2 import createZone

# print("Enter species name:")
# species = str(input())
# Your API definition
app = Flask(__name__)
clf2 = joblib.load("model_RF_Anas crecca.pkl") # Load "model.pkl"
print ('Model loaded')

@app.route('/v1/predict', methods=['POST'])
def predict():
    json_ = request.json
    toPredict = createZone(json_['lat'],json_['lon'])

    # clf2 = joblib.load("model_RF_"+json_['species']+".pkl") # Load "model.pkl"
    # print ('Model loaded')
    # model_columns = joblib.load("model_columns_RF.pkl") # Load "model_columns.pkl"
    # print ('Model columns loaded')

    # query = pd.get_dummies(toPredict)
    # query = query.reindex(columns=model_columns, fill_value=0)
    # print(query)

    predictionProb = clf2.predict_proba(toPredict)[0][1]

    prediction = clf2.predict(toPredict)

    #return jsonify({'prediction': str(prediction), 'probability of occurrence':str(predictionProb)})
    return jsonify({'prediction': str(prediction), 'probability of occurrence':str(predictionProb)})


if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345


    #model_columns = joblib.load("model_columns_RF.pkl") # Load "model_columns.pkl"
    # print ('Model columns loaded')

    app.run(port=5000, debug=True)
