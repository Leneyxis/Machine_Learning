import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset from an Excel file
dataset = pd.read_excel("/content/customer_churn_large_dataset.xlsx")

# Print the column names of the dataset
print("Column Names:", dataset.columns)

# Display the last 10 rows of the dataset
print("Last 10 Rows of the Dataset:")
print(dataset.tail(10))

# Split the dataset into input features (X) and the target variable (y)
X = dataset.iloc[:, 2:-1]
y = dataset.iloc[:, -1]

# Apply one-hot encoding to categorical columns (column 1 and 2)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1, 2])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a Linear Regression model and fit it to the training data
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = regressor.predict(X_test)

# Set print options to display only 5 decimal places
np.set_printoptions(precision=5)

# Print the predicted values
print("Predicted Values:")
print(y_pred)

# Calculate and print the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Calculate and print the Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse:.2f}")

# Calculate and print the R-squared (R2) value
r2 = abs(r2_score(y_test, y_pred))
print(f"R-squared (R2): {r2:.2f}")

# Create a scatter plot to visualize actual vs. predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values")
plt.show()
