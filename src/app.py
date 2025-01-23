from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("./model/housing_best_gradient_boosting_random_model.pkl")  # Ensure model.pkl exists in the same directory

# Define all expected feature columns, including one-hot encoded columns
EXPECTED_COLUMNS = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "ocean_proximity_INLAND",
    "ocean_proximity_ISLAND",
    "ocean_proximity_NEAR BAY",
    "ocean_proximity_NEAR OCEAN"
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse JSON data from the request
        data = request.get_json()

        # Convert JSON input to a DataFrame
        input_data = pd.DataFrame(data)

        # Ensure all expected columns are present
        for col in EXPECTED_COLUMNS:
            if col not in input_data.columns:
                input_data[col] = 0  # Add missing column with default value of 0

        # Reorder columns to match the training data
        input_data = input_data[EXPECTED_COLUMNS]

        # Make predictions
        predictions = model.predict(input_data)

        # Reverse the log transformation to get actual prices
        actual_predictions = np.expm1(predictions)  # expm1 reverses log1p

        # Return predictions as JSON
        return jsonify({"predictions": actual_predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
