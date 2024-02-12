from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
    model : str
    culmenLen : float
    culmenDepth : float
    flipperLen : float
    bodyMass : int
    sex : str
    delta15N : float
    delta13C : float

@app.post("/predict/")
def predict(item:model_input):

    global label_encoder, encoder, knn_model, lr_model, lda_model, scaler

    # Predice con respecto al modelo que se le especifique. 
    X = pd.DataFrame([item.dict().values()], columns=item.dict().keys())

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

    X_train = X.drop(columns=['model'])

    if item.model == "knn":
        output = knn_model.predict(X_train)
        output_cod = label_encoder.inverse_transform(output).tolist()
        return {"prediction" : output_cod}
    elif item.model == "lda":
        output = lda_model.predict(X_train)
        output_cod = label_encoder.inverse_transform(output).tolist()
        return {"prediction" : output_cod}
    elif item.model == "lr":
        output = lr_model.predict(X_train)
        output_cod = label_encoder.inverse_transform(output).tolist()
        return {"prediction" : output_cod}
    else:
        raise HTTPException(status_code=400, detail="Unsupported model")
    
