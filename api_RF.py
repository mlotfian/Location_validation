# Dependencies
from flask import Flask, request, jsonify, render_template, flash
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
#from sqlalchemy_imageattach.entity import Image, image_attachment
import joblib
#import traceback
import pandas as pd
import numpy as np
import csv
import os
import geopandas as gpd
import psycopg2
import numpy as np

#from utils import createZone
from utils2 import createZone, getGridId

#import waitress to serve the api
from waitress import serve

import logging

import ssl


from psycopg2.extensions import register_adapter, AsIs
psycopg2.extensions.register_adapter(np.int64, psycopg2._psycopg.AsIs)

context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
context.load_cert_chain('C:/Certif/cert_new2.pem','C:/Certif/key_new2.pem')


# my API definition
app = Flask(__name__,template_folder='templates')
#app.config.from_object(os.environ['APP_SETTINGS'])
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:mary3000@localhost:5432/location_API'
db = SQLAlchemy(app)
migrate = Migrate(app, db)
app.secret_key = "\x05\xce\\2\xe9\xb4!lA:\xa2\xfa\xa3\x04\x8a\x90\xb2\x88\xe3\r\x9a\x15:O"

class Location_val(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    grid_id = db.Column(db.Integer)
    sp_name = db.Column(db.String(200))
    prob = db.Column(db.Float)
    comm_name = db.Column(db.String(200), nullable=True)
    #sp_image = image_attachment('SP_Picture')
    url = db.Column(db.String(500), nullable=True)
    habitat = db.Column(db.String(500), nullable=True)


# run once only to create the tab
#db.create_all()
swiss_grid = gpd.read_file('C:/Bio_Loc_API/Bio_API_V2/all_swiss/swiss_4326.shp')


#with open('C:/Bio_Loc_API/Bio_API_V2/all_names.csv', newline='') as f:
#    reader = csv.reader(f)
#    all_names = [row for row in reader][0]
#print(all_names)

with open('C:/Bio_Loc_API/Bio_API_V2/all_commNames.csv', newline='') as f:
    reader = csv.reader(f)
    all_names = [row for row in reader]
#print(all_names2[:2])


for sp_name in all_names:
    globals()["RF" + sp_name[0]] = joblib.load("C:/Bio_Loc_API/Bio_API_V2/RFmodels_commname/" + sp_name[0] + "_RFmodel.pkl")
    print("model loaded for {}".format(sp_name[0]))


@app.route('/')
def home():
    return render_template('testLocation.html')

# get the list of possible species to be observed in a 2km2 zone from the location of user
@app.route('/v1/suggest', methods=['POST', 'GET'])
def suggest():
    # location = request.json
    # lat = location['lat']
    # lon = location['lon']
    app.logger.info('Info level log')
    lat = request.args.get('lat', None)
    lon = request.args.get('lon', None)
    index = getGridId(float(lat), float(lon), swiss_grid)
    grid_id = swiss_grid.loc[index[0]].grid_id
    #suggested_sp = (Location_val.query.filter_by(grid_id=grid_id)).order_by(Location_val.prob.desc()).limit(5).all()
    connection = psycopg2.connect(database="location_API",user="postgres", password="mary3000", host='localhost')
    cursor = connection.cursor()
    print(grid_id)
    query = """select comm_name, prob, url from location_val where grid_id=""" + str(grid_id) + """order by prob desc limit 5"""
    cursor.execute(query)
    rows=cursor.fetchall()
    #print(suggested_sp[0].fetchone())
    print(rows)
    # suggestionL = []
    # for el in rows:
    #     suggestionL.append(el)
    # print(suggestionL)
    # flash(suggestionL, 'suggest')
    # return render_template('testLocation.html', suggestionL=suggestionL)
    suggestion= []
    for el in rows:
        suggestion.append({
            "sp_name": el[0],
            "url": el[2]
        })
    response = jsonify(suggestion)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

# predict the probability of occurance of species in a 2km2 zone from the give location
@app.route('/v1/predict', methods=['POST', 'GET'])
def predict():

    app.logger.info('Info level log')
    # json_ = request.json
    # toPredict = createZone(json_['lat'],json_['lon'])
    # sp_name = json_['sp_name']
    lat = request.args.get('lat', None)
    lon = request.args.get('lon', None)
    sp_name = request.args.get('sp_name', None)
    toPredict = createZone(float(lat),float(lon))
    #toPredict = createZone(float(request.form['lat']),float(request.form['lon']))
    #sp_name = request.form['sp_name']
    clf = globals()["RF" + sp_name]

    # query = pd.get_dummies(toPredict)
    # query = query.reindex(columns=model_columns, fill_value=0)
    # print(query)

    predictionProb = clf.predict_proba(toPredict)[0][1]
    print(predictionProb)

    prediction = clf.predict(toPredict)

    connection = psycopg2.connect(database="location_API",user="postgres", password="mary3000", host='localhost')
    cursor = connection.cursor()
    query = """select distinct(habitat), comm_name from location_val where comm_name=""" + """'""" + sp_name + """'"""
    cursor.execute(query)
    rows=cursor.fetchall()
    print(rows)
    habitat = rows[0][0]
    print(habitat)
    comm_name = rows[0][1]
    print(comm_name)
    #pred_feedback = [comm_name, pred_prob, habitat]
    prob = round(predictionProb*100,2)
    pred_feedback = [(comm_name, prob, habitat)]
    print(pred_feedback)
    #prediction_text='The probability of occurrence of {sp} in this location equals {pb:.2f}%'.format(sp=sp_name, pb = predictionProb*100)

    flash(pred_feedback, 'predict')
    response = jsonify({'prediction': str(prediction), 'probability of occurrence':str(prob), "habitat":habitat})
    response.headers.add('Access-Control-Allow-Origin', '*')
    #return jsonify({'prediction': str(prediction), 'probability of occurrence':str(predictionProb)})
    return response
    #return render_template('testLocation.html', pred_feedback = pred_feedback)

@app.route('/v1/names', methods=['GET'])
def names():
    connection = psycopg2.connect(database="location_API",user="postgres", password="mary3000", host='localhost')
    cursor = connection.cursor()
    query = """select distinct(comm_name) from location_val"""
    cursor.execute(query)
    rows=cursor.fetchall()
    response=[]
    for el in rows:
        response.append(el[0])
    response = jsonify({'names': response})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response



if __name__ == '__main__':
    # try:
    #     port = int(sys.argv[1]) # This is for a command-line input
    # except:
    #     port = 12345

    app.run(host='0.0.0.0', threaded=True, port=5000, debug=False, ssl_context=context)
    #serve(app, host='0.0.0.0', port=5000, threads=50, url_scheme='https')
    #app = create_app()
