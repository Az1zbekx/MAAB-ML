# ================================
# Simple Linear Regression Project
# Predicting House Prices
# ================================

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# 2. LOAD DATA
data = pd.read_csv("data/housing.csv")

print("First 5 rows:")
print(data.head())

print("\nInfo:")
print(data.info())

print("\nMissing values:")
print(data.isnull().sum())


# 3. VISUALIZATION (Scatter Plot)
plt.figure(figsize=(8, 5))
plt.scatter(data['area'], data['price'])
plt.xlabel("Area (sq ft)")
plt.ylabel("Price ($)")
plt.title("House Price vs Area")
plt.show()


# 4. PREPARE DATA
X = data['area'].values.reshape(-1, 1)
y = data['price'].values


# 5. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nShapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# 6. TRAIN MODEL
model = LinearRegression()
model.fit(X_train, y_train)

m = model.coef_[0]
b = model.intercept_

print("\nModel Parameters:")
print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")


# 7. PREDICTION ON TEST DATA
y_pred = model.predict(X_test)


# 8. EVALUATION
mse = np.mean((y_test - y_pred) ** 2)
mae = np.mean(np.abs(y_test - y_pred))
r2 = model.score(X_test, y_test)

print("\nEvaluation Metrics:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R²: {r2}")


# 9. VISUALIZATION (Regression Line)
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, label="Actual")
plt.plot(X_test, y_pred, color='red', label="Prediction")
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Regression Line")
plt.legend()
plt.show()


# 10. PREDICTION FOR 1000 sq ft
area_1000 = np.array([[1000]])
price_1000 = model.predict(area_1000)

print("\nPrediction:")
print(f"Predicted price for 1000 sq ft: {price_1000[0]}")


# 11. FINAL INTERPRETATION
print("\nConclusion:")
print("This model shows a moderate relationship between area and price.")
print("However, the R² score indicates that area alone is not enough to accurately predict house prices.")
print("More features (location, rooms, etc.) are needed for better performance.")