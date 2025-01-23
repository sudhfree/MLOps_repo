# Re-importing necessary libraries after reset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Reload the dataset
file_path = './data/housing.csv'
data = pd.read_csv(file_path)

# One-hot encode the 'ocean_proximity' column
ocean_proximity_encoded = pd.get_dummies(data['ocean_proximity'], prefix='ocean_proximity', drop_first=True)


# Concatenate the encoded features with the rest of the dataset
data_encoded = pd.concat([data.drop(columns=['ocean_proximity']), ocean_proximity_encoded], axis=1)

# Apply log scaling to numeric features to handle skewness
numeric_features = data_encoded.select_dtypes(include=['float64', 'int64']).columns
epsilon = 1e-5  # Small constant to avoid log(0)
data_log_scaled = data_encoded.copy()
for feature in numeric_features:
    data_log_scaled[feature] = np.log1p(data_log_scaled[feature] + epsilon)

# Handle missing values
data_log_scaled['longitude'] = data['longitude'].fillna(data['longitude'].median())
data_log_scaled['total_bedrooms'] = data_log_scaled['total_bedrooms'].fillna(data_log_scaled['total_bedrooms'].median())

# Ensure no NaN values remain
assert data_log_scaled.isna().sum().sum() == 0, "There are still NaN values in the dataset!"


# Define features (X) and target (y)
X = data_log_scaled.drop(columns=['median_house_value'])
y = data_log_scaled['median_house_value']

print(X.head())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reduce the dataset size for faster tuning
X_train_small, _, y_train_small, _ = train_test_split(X_train, y_train, test_size=0.8, random_state=42)

# Define parameter grid for RandomizedSearchCV
param_dist = {
    "n_estimators": [50, 100, 150],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [3, 5],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2],
    "subsample": [0.8, 1.0]
}

# Initialize Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(random_state=42)

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=gb_model,
    param_distributions=param_dist,
    scoring="neg_mean_squared_error",
    cv=3,
    n_iter=10,  # Evaluate 10 random combinations
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Fit RandomizedSearchCV
random_search.fit(X_train_small, y_train_small)

# Best parameters
best_params_random = random_search.best_params_
best_model_random = random_search.best_estimator_

# Make predictions using the best model
y_pred_best_random = best_model_random.predict(X_test)

# Evaluate the tuned model
mse_best_random = mean_squared_error(y_test, y_pred_best_random)
r2_best_random = r2_score(y_test, y_pred_best_random)

# Save the best model
best_model_random_file_path = "./model/housing_best_gradient_boosting_random_model.pkl"
joblib.dump(best_model_random, best_model_random_file_path)

# Return the results
best_params_random, mse_best_random, r2_best_random, best_model_random_file_path
print(best_params_random, mse_best_random, r2_best_random, best_model_random_file_path)