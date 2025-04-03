from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from flask_cors import CORS  # Import CORS
from mainLogic import predict_from_user_input

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to predict RO system performance based on input parameters.
    
    Expected JSON input format:
    {
        "Feed Flow (m3/hr)": 100.0,
        "Feed Temperature": 25.0,
        ... (all required input parameters)
    }
    
    Returns:
    JSON with predicted values for all output parameters
    """
    try:
        # Get input data from request
        input_data = request.json
        
        # Validate input data
        if not input_data:
            return jsonify({"error": "No input data provided"}), 400
        
        # Make prediction using the function from mainLogic.py
        predictions = predict_from_user_input(input_data)
        
        # Return predictions as JSON
        return jsonify({
            "status": "success",
            "predictions": predictions
        })
        
    except ValueError as e:
        # Handle missing input features
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # Handle other errors
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/input_schema', methods=['GET'])
def get_input_schema():
    """
    Returns the required input parameters for the prediction model.
    """
    # These should match the input features in mainLogic.py
    required_inputs = [
        'Feed Flow (m3/hr)', 'Feed Temperature', 'Feed water pH', 'Pass Stage',
        'Pressure Vessel', 'Elements', 'Element age(years)', 'Recovery(%)',
        'Ca_FW', 'Mg_FW', 'Na_FW', 'K_FW', 'NH4_FW', 'Ba_FW', 'Sr_FW', 'H_FW',
        'CO3_FW', 'HCO3_FW', 'SO4_FW', 'Cl_FW', 'F_FW', 'NO3_FW', 'PO4_FW',
        'OH_FW', 'SiO2_FW', 'B_FW', 'CO2_FW', 'NH3_FW', 'Feed Water TDS',
        'CaSO4 / ksp * 100, %_FW', 'SrSO4 / ksp * 100, %_FW',
        'BaSO4 / ksp * 100, %_FW', 'SiO2 saturation, %_FW',
        'CaF2 / ksp * 100, %_FW'
    ]
    
    return jsonify({
        "required_inputs": required_inputs,
        "example_input": {
            'Feed Flow (m3/hr)': 100.0,
            'Feed Temperature': 25.0,
            'Feed water pH': 7.5,
            'Pass Stage': 1,
            'Pressure Vessel': 6,
            'Elements': 6,
            'Element age(years)': 2.5,
            'Recovery(%)': 75.0,
            'Ca_FW': 80.0,
            'Mg_FW': 30.0,
            'Na_FW': 120.0,
            'K_FW': 10.0,
            'NH4_FW': 0.5,
            'Ba_FW': 0.05,
            'Sr_FW': 0.8,
            'H_FW': 0.01,
            'CO3_FW': 5.0,
            'HCO3_FW': 150.0,
            'SO4_FW': 90.0,
            'Cl_FW': 130.0,
            'F_FW': 1.0,
            'NO3_FW': 5.0,
            'PO4_FW': 2.0,
            'OH_FW': 0.01,
            'SiO2_FW': 20.0,
            'B_FW': 0.5,
            'CO2_FW': 5.0,
            'NH3_FW': 0.2,
            'Feed Water TDS': 500.0,
            'CaSO4 / ksp * 100, %_FW': 10.0,
            'SrSO4 / ksp * 100, %_FW': 5.0,
            'BaSO4 / ksp * 100, %_FW': 2.0,
            'SiO2 saturation, %_FW': 15.0,
            'CaF2 / ksp * 100, %_FW': 3.0
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)