import pickle
import pandas as pd

class PenguinModel():
    def __init__(self, model_path):
        self.model = pickle.load(open(model_path, 'rb'))
        self.scaler = pickle.load(open("../models/standard_scaler.pkl", 'rb'))
        self.l_encoder = pickle.load(open("../models/label_encoder.pkl", 'rb'))
        self.v_encoder = pickle.load(open("../models/variable_encoder.pkl", 'rb'))
        self.numerical_features = [
            "culmen_length_mm",
            "culmen_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
        ]
        self.categorical_features = ["island", "sex"]

    def predict(self, pattern):
        features = pd.DataFrame.from_dict({
            "island": [pattern.island],
            "culmen_length_mm": [pattern.culmen_length_mm],
            "culmen_depth_mm": [pattern.culmen_depth_mm],
            "flipper_length_mm": [pattern.flipper_length_mm],
            "body_mass_g": [pattern.body_mass_g],
            "sex": [pattern.sex]
        })

        features[self.numerical_features] = self.scaler.transform(features[self.numerical_features])
        features[self.categorical_features] = self.v_encoder.transform(features[self.categorical_features])

        prediction = self.model.predict(features)      
        prediction = self.l_encoder.inverse_transform(prediction)
        return prediction[0]