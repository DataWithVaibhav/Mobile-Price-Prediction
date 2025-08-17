import os
import pickle
import logging
import traceback
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from flask import Flask, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
try:
    model = pickle.load(open('pipe_data.pkl', 'rb'))
    logging.info("\u2705 Model loaded successfully.")
except Exception as e:
    logging.error(f"\u274C Error loading model: {e}")
    model = None

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model failed to load."}), 500
        
        # Collect form data
        phone_name = request.form.get('phone_name', "Unknown")  # Add phone name input
        rating = float(request.form.get('rating', 0))
        num_ratings = int(request.form.get('num_ratings', 0))
        ram = int(request.form.get('ram', 0))
        storage = int(request.form.get('storage', 0))
        battery = int(request.form.get('battery', 0))
        brand = request.form.get('brand', "Unknown")
        processor = request.form.get('processor', "Unknown")
        front_camera = int(request.form.get('front_camera', 0))
        rear_camera = int(request.form.get('rear_camera', 0))

        # Create DataFrame
        input_data = pd.DataFrame([[phone_name, rating, num_ratings, ram, storage, rear_camera, front_camera, battery, processor]], 
                                columns=['Phone Name', 'Rating ?/5', 'Number of Ratings', 'RAM', 'ROM/Storage', 'Back/Rare Camera', 
                                        'Front Camera', 'Battery', 'Processor'])

        # Predict price
        predicted_price = model.predict(input_data)
        logging.info(f"\U0001F52E Predicted Price: {predicted_price}")

        return render_template('index.html', prediction=round(float(predicted_price[0]), 2))

    except Exception as e:
        logging.error(f"⚠️ Prediction error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
