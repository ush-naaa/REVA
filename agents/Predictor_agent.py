# agents/Predictor_agent.py

import joblib
import pandas as pd
import numpy as np
from utils.preprocessing import preprocess_for_model


class PredictorAgent:
    def __init__(self, model_path="reva_ann_model_assets.joblib"):
        assets = joblib.load(model_path)
        self.model = assets["model"]
        self.preprocessor = assets["preprocessor"]
        self.label_encoder = assets["label_encoder"]
        self.location_map = assets["location_map"]

    def predict(self, user_input_dict):
        """
        Returns:
            predicted_label (str)
            confidence (float in %)
        """

        # Preprocess input
        processed = preprocess_for_model(user_input_dict, self.location_map)
        processed_final = self.preprocessor.transform(processed)

        # Predict probabilities
        probs = self.model.predict_proba(processed_final)[0]

        # Get predicted class
        pred_idx = np.argmax(probs)
        pred_label = self.label_encoder.inverse_transform([pred_idx])[0]
        confidence = round(probs[pred_idx] * 100, 2)

        return pred_label, confidence

    def predict_proba(self, user_input_dict):
        """
        Returns full probability distribution (for internal use if needed)
        """

        processed = preprocess_for_model(user_input_dict, self.location_map)
        processed_final = self.preprocessor.transform(processed)
        probs = self.model.predict_proba(processed_final)[0]

        return probs
