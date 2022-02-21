# -*- coding: utf-8 -*-
import joblib
import json

from flask import Flask, request, jsonify
import numpy as np

log_clf = joblib.load('log_model.joblib')
tree_clf = joblib.load('tree_model.joblib')
rf_clf = joblib.load('rf_model.joblib')
ada_clf = joblib.load('ada_model.joblib')
vote_clf = joblib.load('vote_model.joblib')

with open('dummy_column_mapper.json') as fin:
    dummy_column_mapper = json.load(fin)
    
with open('scaler_info.json') as fin:
    scaler_info = json.load(fin)
    
with open('col_order.json') as fin:
    col_order = json.load(fin)
    

# make app
app = Flask(__name__)
app.config["DEBUG"] = True

# define endpoints
@app.route('/', methods=['GET'])
def home():       
            
    return 'App is Healthy'


@app.route('/predict', methods=['POST'])
def predict():
    
    payload = request.json
    for column, dummy_columns in dummy_column_mapper.items():
        for dummy_column in dummy_columns:
            payload[dummy_column] = 0
        if column in payload:
            column_val = payload.pop(column)
            target_column = f'{column}_{column_val}'
            payload[target_column] = 1

    for key, scaler_params in scaler_info.items():
        if key in payload:
            payload[key] = (payload[key] - scaler_params['mean'])/scaler_params['std']
        else:
            payload[key] = scaler_params['mean']

    ordered_payload = {}
    for col in col_order:
        ordered_payload[col] = payload[col]
    
    log_prediction = int(log_clf.predict(np.array(list(ordered_payload.values())).reshape(1, -1)))
    tree_prediction = int(tree_clf.predict(np.array(list(ordered_payload.values())).reshape(1, -1)))
    rf_prediction = int(rf_clf.predict(np.array(list(ordered_payload.values())).reshape(1, -1)))
    ada_prediction = int(ada_clf.predict(np.array(list(ordered_payload.values())).reshape(1, -1)))
    vote_prediction = int(vote_clf.predict(np.array(list(ordered_payload.values())).reshape(1, -1)))
    
    prediction_list = [log_prediction, tree_prediction, rf_prediction, ada_prediction, vote_prediction]
    
    prediction = int(prediction_list.count(1) > prediction_list.count(0))
    
    # return str(prediction)
    return 1


if __name__ == '__main__':
    app.run(debug=True)