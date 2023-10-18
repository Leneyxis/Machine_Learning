# Customer Churn Prediction with Linear Regression

## Overview

This Python code is designed to perform customer churn prediction using a Linear Regression model. Customer churn refers to the rate at which customers stop doing business with a company. The code reads a dataset from an Excel file, preprocesses the data, applies one-hot encoding to categorical features, and then trains a Linear Regression model to make predictions. It also calculates and displays various evaluation metrics, such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2). Finally, it visualizes the actual vs. predicted values using a scatter plot.

## Prerequisites

To run this code, you will need the following:

- Python (3.7 or later)
- NumPy
- pandas
- scikit-learn
- Matplotlib

You can install these dependencies using pip:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Usage

1. **Dataset**: Ensure you have a dataset in Excel format (e.g., "customer_churn_large_dataset.xlsx") that contains the necessary data for customer churn prediction.

2. **Code Execution**:

   - Clone or download this repository.
   - Place the dataset file in the same directory as the code file.

3. **Run the Code**:

   Execute the code by running it with Python. You can do this from the command line or using your preferred Python IDE or Jupyter Notebook.

   ```bash
   python customer_churn_prediction.py
   ```

   Replace `customer_churn_prediction.py` with the actual name of your code file.

4. **Results**:

   The code will display the following results:

   - Column names of the dataset.
   - Last 10 rows of the dataset.
   - Predicted values for the test data.
   - Mean Squared Error (MSE).
   - Root Mean Squared Error (RMSE).
   - R-squared (R2) value.
   - A scatter plot visualizing actual vs. predicted values.

5. **Interpretation**:

   Analyze the results to understand the model's predictive performance and how well it fits the data. The scatter plot provides a visual representation of the model's accuracy.

## Customization

You can customize the code by:

- Using a different dataset by changing the Excel file path.
- Modifying the machine learning model or preprocessing steps to fit your specific use case.
- Adjusting the test/train split ratio.
- Changing visualization options or adding more plots.

## License

This code is provided under the MIT License. You are free to use and modify it as needed. Refer to the "LICENSE" file for more details.

## Author

Syed Umer Ahmed

Feel free to include your name as the author if you want to claim ownership of the code.

---
