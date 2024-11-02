from flask import Flask, render_template, request, jsonify, make_response
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
import random

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn import set_config
set_config(transform_output="pandas")
from sklearn.metrics import r2_score, mean_squared_log_error

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

import joblib

app = Flask(__name__)

df=pd.read_csv("train.csv")

@app.route('/')
def index():
    return render_template('deploy.html')

@app.route('/predict', methods=(['POST']))
def predict():
    data=request.json
    print(data)
    res={}
    key_dict={
        "cc":"Engine CC",
        "width":"Width (mm)",
        "length":"Length (mm)",
        "frim":"Front Rim (inches)",
        "rrim":"Rear Rim (inches)",
        "weight":"Kerb Weight (kg)",
        "num_gears":"Number of Gears",
        "pp":"Peak Power (hp)",
        "pt":"Peak Torque (Nm)",
        "fthread":"Front Thread",
        "rthread":"Rear Thread",
        "parking":"Parking sensor",
        "side":"Side mirror turning indicators"
    }
    df2=df.loc[(df.model==data['model']) & (df.series==data['series'])]
    for col in df2.columns:
        res[col]=df2[col].mode()
    for key,val in data.items():
        if key in key_dict: key=key_dict[key]
        if key in ['model','series']:continue
        elif key in ["CD","Immobilizer","USB","Parking sensor","Side mirror turning indicators"]:
            res[key]=int(val)
        else:
            if val: 
                print(val)
                res[key]=int(val)
    df2=pd.DataFrame(res)
    print(df2)
    pipeline = joblib.load('pipeline.pkl')
    price=round(pipeline.predict(df2)[0],2)
    response=jsonify({'price':price})
    resp=make_response(response,200)
    return resp