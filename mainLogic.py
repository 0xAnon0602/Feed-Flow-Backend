import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Function to train the model and return necessary components
def train_model(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    
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
    # Commented out to ensure all output parameters are included
    # low_variance_columns = [col for col in Y.columns if Y[col].nunique() <= 1]
    # Y = Y.drop(columns=low_variance_columns)
    
    # Train-Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Scale Input Data
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    
    # Scale Output Data
    y_scaler = StandardScaler()
    Y_train_scaled = y_scaler.fit_transform(Y_train)
    Y_test_scaled = y_scaler.transform(Y_test)
    
    # Train KNN Model
    knn_model = KNeighborsRegressor(n_neighbors=20, metric='euclidean')  
    knn_model.fit(X_train_scaled, Y_train_scaled)
    
    return knn_model, x_scaler, y_scaler, X, Y, X_test, Y_test

# Function to evaluate model performance
def evaluate_model(model, x_scaler, y_scaler, X_test, Y_test):
    # Predict
    Y_pred_scaled = model.predict(x_scaler.transform(X_test))
    Y_pred = y_scaler.inverse_transform(Y_pred_scaled)
    
    # Compute Metrics
    mae = mean_absolute_error(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test, Y_pred)  
    
    # Compute R² & MAE for Each Output Parameter
    r2_scores = {col: r2_score(Y_test[col], Y_pred[:, i]) for i, col in enumerate(Y_test.columns)}
    mae_scores = {col: mean_absolute_error(Y_test[col], Y_pred[:, i]) for i, col in enumerate(Y_test.columns)}
    
    # Print Results
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Overall R² Score: {r2:.4f}")
    
    print("\nR² Scores for Each Output Parameter:")
    for col, r2_value in r2_scores.items():
        print(f"{col}: {r2_value:.4f}")
    
    print("\nMean Absolute Error for Each Output Parameter:")
    for col, mae_value in mae_scores.items():
        print(f"{col}: {mae_value:.4f}")

# Function to get prediction based on input data
def predict_from_user_input(input_data=None, file_path="Training_Data.csv"):
    # Train model and get necessary components
    model, x_scaler, y_scaler, X, Y, _, _ = train_model(file_path)
    
    # Get column names from the training data
    X_columns = X.columns
    Y_columns = Y.columns
    
    # If no input data is provided, use the interactive mode
    if input_data is None:
        # Ask for user input
        print("\nEnter values for prediction:")
        user_input = {}
        for column in X_columns:
            while True:
                try:
                    value = float(input(f"Enter value for {column}: "))
                    user_input[column] = value
                    break
                except ValueError:
                    print("Please enter a valid number.")
    else:
        # Use the provided input data
        user_input = input_data
        
        # Validate that all required columns are present
        missing_columns = [col for col in X_columns if col not in user_input]
        if missing_columns:
            raise ValueError(f"Missing required input parameters: {', '.join(missing_columns)}")
    
    # Convert user input to DataFrame
    user_input_df = pd.DataFrame([user_input])
    
    # Scale user input
    user_input_scaled = x_scaler.transform(user_input_df)
    
    # Make prediction
    predicted_scaled = model.predict(user_input_scaled)
    predicted = y_scaler.inverse_transform(predicted_scaled)
    
    # Create a dictionary of predictions
    prediction_dict = {col: float(predicted[0][i]) for i, col in enumerate(Y_columns)}
    
    # If running in interactive mode, display results
    if input_data is None:
        print("\nPredicted values:")
        for col, value in prediction_dict.items():
            print(f"{col}: {value:.4f}")
    
    return prediction_dict

# Main execution
if __name__ == "__main__":
    file_path = "Training_Data.csv"
    
    print("KNN Regression Model for Water Treatment Parameters")
    print("=" * 50)
    
    # Ask user what they want to do
    print("\nOptions:")
    print("1. Evaluate model performance")
    print("2. Make predictions with new input")
    print("3. Both (evaluate model and then make predictions)")
    
    choice = input("\nEnter your choice (1/2/3): ")
    
    # Train model (required for all options)
    model, x_scaler, y_scaler, X, Y, X_test, Y_test = train_model(file_path)
    if choice == '1' or choice == '3':
        print("\nModel Evaluation:")
        print("-" * 30)
        evaluate_model(model, x_scaler, y_scaler, X_test, Y_test)
    
    if choice == '2' or choice == '3':
        if choice == '3':
            print("\nNow proceeding to prediction...")
        predict_from_user_input()