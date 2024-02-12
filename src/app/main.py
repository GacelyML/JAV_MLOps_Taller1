from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import joblib
import json
import pandas as pd

# Carga los elementos entrenados
label_encoder = joblib.load('../models/label_encoder.pkl')
encoder = joblib.load('../models/variable_encoder.pkl')
knn_model = joblib.load('../models/knn_model.pkl')
lr_model = joblib.load('../models/lr_model.pkl')
lda_model = joblib.load('../models/lda_model.pkl')
scaler = joblib.load('../models/standard_scaler.pkl')

app = FastAPI()

class model_input(BaseModel):
    model : Optional[str] = None
    culmenLen : List[float]
    culmenDepth : List[float]
    flipperLen : List[float]
    bodyMass : List[int]
    sex : List[str]
    delta15N : List[float]
    delta13C : List[float]

@app.post("/predict/")
def predict(item:model_input):

    global label_encoder, encoder, knn_model, lr_model, lda_model, scaler

    data_dict = item.dict()
    model_info = data_dict.pop('model', None)
    
    # Predice con respecto al modelo que se le especifique. 
    X = pd.DataFrame({key: value for key, value in data_dict.items()})

    numerical_features = [
    "culmenLen",
    "culmenDepth",
    "flipperLen",
    "bodyMass",
    "delta15N",
    "delta13C"
    ]
    categorical_features = ["sex"]

    X[numerical_features] = scaler.transform(X[numerical_features])
    X[categorical_features] = encoder.transform(X[categorical_features])

    if model_info == "knn":
        output = knn_model.predict(X)
        output_cod = label_encoder.inverse_transform(output).tolist()
        return {"prediction" : output_cod}
    elif model_info == "lda":
        output = lda_model.predict(X)
        output_cod = label_encoder.inverse_transform(output).tolist()
        return {"prediction" : output_cod}
    elif model_info == "lr":
        output = lr_model.predict(X)
        output_cod = label_encoder.inverse_transform(output).tolist()
        return {"prediction" : output_cod}
    else:
        raise HTTPException(status_code=400, detail="Unsupported model")
    
