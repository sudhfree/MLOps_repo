import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import warnings
import joblib


# Suppress a specific warning
warnings.filterwarnings('ignore', category=UserWarning)  # Example for user warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


mlflow.set_tracking_uri("http://localhost:5000")
# Reload the dataset
file_path = '../../data/housing.csv'
data = pd.read_csv(file_path)

# One-hot encode the 'ocean_proximity' column
ocean_proximity_encoded = pd.get_dummies(data['ocean_proximity'], prefix='ocean_proximity', drop_first=True)

# Feature selection and preprocessing
data_encoded = pd.concat([data.drop(columns=['ocean_proximity']), ocean_proximity_encoded], axis=1)

# Apply log scaling to numeric features to handle skewness
numeric_features = data_encoded.select_dtypes(include=['float64', 'int64']).columns
epsilon = 1e-5  # Small constant to avoid log(0)
data_log_scaled = data_encoded.copy()

for feature in numeric_features:
    data_log_scaled[feature] = np.log1p(data_log_scaled[feature] + epsilon)

# Handle missing values
data_log_scaled['longitude'] = data['longitude'].fillna(data['longitude'].median())
data_log_scaled['total_bedrooms'] = data['total_bedrooms'].fillna(data['total_bedrooms'].median())

# Ensure no NaN values remain
assert data_log_scaled.isna().sum().sum() == 0, "There are still NaN values in the dataset!"

# Define features (X) and target (y)
X = data_log_scaled.drop(columns=['median_house_value'])
y = data_log_scaled['median_house_value']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter grids for RandomizedSearchCV
param_grid_list = [
    {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 4],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2],
        "subsample": [0.8, 1.0]
    },
    {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.05],
        "max_depth": [3, 5],
        "min_samples_split": [5, 10],
        "min_samples_leaf": [2, 4],
        "subsample": [0.7, 0.9]
    },
    {
        "n_estimators": [75, 150],
        "learning_rate": [0.05, 0.1],
        "max_depth": [4, 6],
        "min_samples_split": [3, 6],
        "min_samples_leaf": [2],
        "subsample": [0.85, 1.0]
    }
]

# Track the best model across all runs
best_overall_model = None
best_overall_params = None
best_overall_mse = float("inf")
best_overall_run_name = None

# Loop over each parameter grid and perform randomized search
for i, param_grid in enumerate(param_grid_list):
    with mlflow.start_run(run_name=f"RandomizedSearchCV_Run_{i+1}"):
        # Initialize RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=GradientBoostingRegressor(random_state=42),
            param_distributions=param_grid,
            scoring="neg_mean_squared_error",
            n_iter=10,  # Evaluate 10 random combinations
            cv=3,
            verbose=1,
            random_state=42,
            n_jobs=-1
        )

        # Fit RandomizedSearchCV
        random_search.fit(X_train, y_train)

        # Get the best model and parameters
        best_params = random_search.best_params_
        best_model = random_search.best_estimator_

        # Make predictions using the best model
        y_pred = best_model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log hyperparameters and metrics
        for param, value in best_params.items():
            mlflow.log_param(param, value)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)

        # Log the model with input example
        input_example = X_test.iloc[0].to_numpy().reshape(1, -1)
        mlflow.sklearn.log_model(best_model, f"gradient_boosting_model_run_{i+1}", input_example=input_example)

        # Print results for this run
        print(f"Run {i+1}:")
        print(f"Best Params: {best_params}")
        print(f"MSE: {mse}")
        print(f"R2 Score: {r2}")
        print("-" * 50)

        # Update the best overall model if current run is better
        if mse < best_overall_mse:
            best_overall_model = best_model
            best_overall_params = best_params
            best_overall_mse = mse
            best_overall_run_name = f"RandomizedSearchCV_Run_{i+1}"

# Export the best overall model
if best_overall_model is not None:
    print(f"Best Model Found: {best_overall_run_name}")
    print(f"Best Params: {best_overall_params}")
    print(f"Best MSE: {best_overall_mse}")

    # Save the best model locally
    best_model_path = "../model/best_gradient_boosting_model.pkl"
    joblib.dump(best_overall_model, best_model_path)
    print(f"Best model saved at: {best_model_path}")
