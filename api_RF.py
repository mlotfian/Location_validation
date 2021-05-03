# Dependencies
from flask import Flask, request, jsonify, render_template, flash
import joblib
import traceback
import pandas as pd
import numpy as np
import csv


#from utils import createZone
from utils2 import createZone

from memory_profiler import profile






# print("Enter species name:")
# species = str(input())
# Your API definition
app = Flask(__name__,template_folder='templates')
app.secret_key = "\x05\xce\\2\xe9\xb4!lA:\xa2\xfa\xa3\x04\x8a\x90\xb2\x88\xe3\r\x9a\x15:O"



#clf2 = joblib.load("model_RF_Anas crecca.pkl") # Load "model.pkl"
@app.route('/')
def home():
    return render_template('testLocation.html')

@app.route('/v1/predict', methods=['POST'])
def predict():
    # json_ = request.json
    # toPredict = createZone(json_['lat'],json_['lon'])
    # sp_name = json_['species_name']
    toPredict = createZone(float(request.form['lat']),float(request.form['lon']))
    sp_name = request.form['sp_name']
    clf = globals()["RF" + sp_name]

    # query = pd.get_dummies(toPredict)
    # query = query.reindex(columns=model_columns, fill_value=0)
    # print(query)

    predictionProb = clf.predict_proba(toPredict)[0][1]
    print(predictionProb)

    prediction = clf.predict(toPredict)
    prediction_text='The probability of occurrence of {sp} in this location equals {pb:.2f}%'.format(sp=sp_name, pb = predictionProb*100)

    flash(prediction_text)
    #return jsonify({'prediction': str(prediction), 'probability of occurrence':str(predictionProb)})
    #return jsonify({'prediction': str(prediction), 'probability of occurrence':str(predictionProb)})
    return render_template('testLocation.html', prediction_text = prediction_text)




if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345

    with open('C:/notes/Location_validation/API_Django/env/all_names.csv', newline='') as f:
        reader = csv.reader(f)
        all_names = [row for row in reader][0]


    for sp_name in all_names[:15]:
        globals()["RF" + sp_name] = joblib.load("C:/notes/Location_validation/API_Django/RFmodels/" + sp_name + "_RFmodel.pkl")
        print("model loaded for {}".format(sp_name))

    #model_columns = joblib.load("model_columns_RF.pkl") # Load "model_columns.pkl"
    # print ('Model columns loaded')

    app.run(port=5000, debug=True)
