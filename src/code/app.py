import logging
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model
model_path = '../model/best_gradient_boosting_model.pkl'
model = joblib.load(model_path)

# Define the expected features, ensuring compatibility with the model
EXPECTED_FEATURES = [
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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON request
        data = request.get_json()

        # Extract the input features from the request
        input_data = data.get('features')
        
        if not input_data:
            return jsonify({'error': 'Missing input features'}), 400
        
        # Ensure all expected features are in the input
        for feature in EXPECTED_FEATURES:
            if feature not in input_data:
                input_data[feature] = 0  # Set missing features to zero (or default)

        # Convert the input data into a DataFrame
        input_df = pd.DataFrame([input_data])

        # Log the incoming data (for debugging and monitoring)
        logging.info(f'Incoming prediction request with features: {input_data}')

        # Make the prediction using the trained model
        prediction = model.predict(input_df)

        # Reverse the log transformation (to get the prediction back to original scale)
        actual_prediction = np.expm1(prediction[0])  # Reverse the log1p transformation

        # Log the prediction
        logging.info(f'Prediction made: {actual_prediction}')

        # Return the prediction in a structured response
        response = {
            'Housing prediction for given data': {
                'predicted_value': actual_prediction
            }
        }
        
        return jsonify(response), 200

    except Exception as e:
        # Log the error
        logging.error(f'Error during prediction: {str(e)}')
        return jsonify({'error': 'An error occurred during prediction'}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
