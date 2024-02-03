from app.model import PenguinModel
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

@app.on_event('startup')
async def startup_event():
    global penguin_model_1, penguin_model_2, penguin_model_3
    penguin_model_1 = PenguinModel("../models/knn_model.pkl")
    penguin_model_2 = PenguinModel("../models/lda_model.pkl")
    penguin_model_3 = PenguinModel("../models/lr_model.pkl")

class Pattern(BaseModel):
    island: str
    culmen_length_mm: float
    culmen_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    sex: str

class PatternModel(BaseModel):
    model: str
    island: str
    culmen_length_mm: float
    culmen_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    sex: str

@app.get("/")
def read_root():
    return {"Hello": "This is a API to get prediction from Model"}


@app.post("/predict/")
def read_item(pattern: Pattern):
    global penguin_model_1
    prediction = penguin_model_1.predict(pattern)
    return {"Prediction": prediction}

@app.post("/predict_model/")
def read_item(pattern: PatternModel):
    global penguin_model_1, penguin_model_2, penguin_model_3
    if pattern.model == "knn":
        return {"Prediction": penguin_model_1.predict(pattern)}
    elif pattern.model == "lda":
        return {"Prediction": penguin_model_2.predict(pattern)}
    elif pattern.model == "lr":
        return {"Prediction": penguin_model_3.predict(pattern)}
    else:
        raise HTTPException(status_code=400, detail="Unsupported model")