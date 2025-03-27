import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the excel file
data = pd.read_excel("TrainingData.xlsx")

# Define Input (X) and Output (Y) variables
X = data[['Feed Flow (m3/hr)', 'Feed Temperature', 'Feed water pH','Pass Stage','Pressure Vessel','Elements','Element age(years)','Recovery(%)',
          'Ca_FW', 'Mg_FW', 'Na_FW', 'K_FW', 'NH4_FW', 'Ba_FW', 'Sr_FW', 'H_FW', 'CO3_FW', 'HCO3_FW', 'SO4_FW', 'Cl_FW', 'F_FW',
          'NO3_FW', 'PO4_FW', 'OH_FW', 'SiO2_FW', 'B_FW', 'CO2_FW', 'NH3_FW', 'Feed Water TDS','CaSO4 / ksp * 100, %_FW','SrSO4 / ksp * 100, %_FW','BaSO4 / ksp * 100, %_FW','SiO2 saturation, %_FW','CaF2 / ksp * 100, %_FW'
         ]]

Y = data[['Feed Pressure(bar)','Specific Energy(kwh/m3)','Flux(lmh)','Ca_P', 'Mg_P', 'Na_P', 'K_P', 'NH4_P', 'Ba_P', 'Sr_P', 'H_P', 'CO3_P', 'HCO3_P',
                    'SO4_P', 'Cl_P', 'F_P', 'NO3_P', 'PO4_P', 'OH_P', 'SiO2_P', 'B_P', 'CO2_P', 'NH3_P',
                    'Permeate TDS', 'Ca_C', 'Mg_C', 'Na_C', 'K_C', 'NH4_C', 'Ba_C', 'Sr_C', 'H_C',
                    'CO3_C', 'HCO3_C', 'SO4_C', 'Cl_C', 'F_C', 'NO3_C', 'PO4_C', 'OH_C', 'SiO2_C',
                    'B_C', 'CO2_C', 'NH3_C', 'Concentrate TDS', 'CaSO4 / ksp * 100, %_C',
                    'SrSO4 / ksp * 100, %_C', 'BaSO4 / ksp * 100, %_C', 'SiO2 saturation, %_C',
                    'CaF2 / ksp * 100, %_C']]

# Remove columns with zero variance (columns with only one unique value)
low_variance_columns = [col for col in Y.columns if Y[col].nunique() <= 1]
Y = Y.drop(columns=low_variance_columns)

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Scale Input Data
x_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

# Scale Output Data (Fixes Negative R²)
y_scaler = StandardScaler()
Y_train_scaled = y_scaler.fit_transform(Y_train)
Y_test_scaled = y_scaler.transform(Y_test)

# Train KNN Model
knn_model = KNeighborsRegressor(n_neighbors=5, metric='euclidean')  
knn_model.fit(X_train_scaled, Y_train_scaled)

# Predict
Y_pred_scaled = knn_model.predict(X_test_scaled)

# Convert predictions back to original scale
Y_pred = y_scaler.inverse_transform(Y_pred_scaled)

# Compute Metrics
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, Y_pred)  

# Compute R² & MAE for Each Output Parameter
r2_scores = {col: r2_score(Y_test[col], Y_pred[:, i]) for i, col in enumerate(Y_test.columns)}
mae_scores = {col: mean_absolute_error(Y_test[col], Y_pred[:, i]) for i, col in enumerate(Y_test.columns)}

# # Predict for Row 390
# row_390 = data.iloc[[389]]  
# X_row_390 = row_390[X.columns]
# X_row_390_scaled = x_scaler.transform(X_row_390)

# predicted_row_390_scaled = knn_model.predict(X_row_390_scaled)
# predicted_row_390 = y_scaler.inverse_transform(predicted_row_390_scaled)

# # Compare Actual vs Predicted
# print("\nPredicted for Row 390:")
# for i, col in enumerate(Y_test.columns):
#     print(f"{col}: Predicted = {predicted_row_390[0][i]:.4f}")

# To predict for new input data:
def predict_for_new_input(input_data):
    """
    Predict output values for new input data.
    
    Parameters:
    input_data (dict): Dictionary with feature names as keys and values as values
                      Must contain all features used in training
    
    Returns:
    dict: Dictionary with predicted values for each output
    """
    # Convert input dictionary to DataFrame
    new_input_df = pd.DataFrame([input_data])
    
    # Ensure all required columns are present
    missing_cols = set(X.columns) - set(new_input_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required input features: {missing_cols}")
    
    # Reorder columns to match training data
    new_input_df = new_input_df[X.columns]
    
    # Scale the input using the same scaler used for training
    new_input_scaled = x_scaler.transform(new_input_df)
    
    # Make prediction
    new_pred_scaled = knn_model.predict(new_input_scaled)
    
    # Transform prediction back to original scale
    new_pred = y_scaler.inverse_transform(new_pred_scaled)
    
    # Create dictionary of predictions
    predictions = {col: new_pred[0][i] for i, col in enumerate(Y.columns)}
    
    return predictions

# Example usage:
# Define new input values
# new_input = {
#     'Feed Flow (m3/hr)': 100.0,
#     'Feed Temperature': 25.0,
#     'Feed water pH': 7.5,
#     'Pass Stage': 1,
#     'Pressure Vessel': 6,
#     'Elements': 6,
#     'Element age(years)': 2.5,
#     'Recovery(%)': 75.0,
#     'Ca_FW': 80.0,
#     'Mg_FW': 30.0,
#     'Na_FW': 120.0,
#     'K_FW': 10.0,
#     'NH4_FW': 0.5,
#     'Ba_FW': 0.05,
#     'Sr_FW': 0.8,
#     'H_FW': 0.01,
#     'CO3_FW': 5.0,
#     'HCO3_FW': 150.0,
#     'SO4_FW': 90.0,
#     'Cl_FW': 130.0,
#     'F_FW': 1.0,
#     'NO3_FW': 5.0,
#     'PO4_FW': 2.0,
#     'OH_FW': 0.01,
#     'SiO2_FW': 20.0,
#     'B_FW': 0.5,
#     'CO2_FW': 5.0,
#     'NH3_FW': 0.2,
#     'Feed Water TDS': 500.0,
#     'CaSO4 / ksp * 100, %_FW': 10.0,
#     'SrSO4 / ksp * 100, %_FW': 5.0,
#     'BaSO4 / ksp * 100, %_FW': 2.0,
#     'SiO2 saturation, %_FW': 15.0,
#     'CaF2 / ksp * 100, %_FW': 3.0
# }

# # Make prediction
# try:
#     predictions = predict_for_new_input(new_input)
    
#     print("\nPredictions for new input data:")
#     for output_name, predicted_value in predictions.items():
#         print(f"{output_name}: {predicted_value:.4f}")
        
# except ValueError as e:
#     print(f"Error: {e}")
#     print("Please provide values for all required input features.")