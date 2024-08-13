import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = pd.Series(california.target, name='MEDV')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train, y_train)
y_pred_ridge = ridge_reg.predict(X_test)
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}\n")
evaluate_model(y_test, y_pred_lin, "Linear Regression")
evaluate_model(y_test, y_pred_ridge, "Ridge Regression")
def display_predictions(y_true, y_pred, model_name, num_examples=5):
    comparison = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred
    })
    print(f"\n{model_name} Predictions (First {num_examples} Examples):")
    print(comparison.head(num_examples))
    print("\n")
display_predictions(y_test.reset_index(drop=True), y_pred_lin, "Linear Regression")
display_predictions(y_test.reset_index(drop=True), y_pred_ridge, "Ridge Regression")
plt.scatter(y_test, y_pred_lin, color='blue', label='Linear Regression', alpha=0.5)
plt.scatter(y_test, y_pred_ridge, color='red', label='Ridge Regression', alpha=0.5)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.legend()
plt.show()
