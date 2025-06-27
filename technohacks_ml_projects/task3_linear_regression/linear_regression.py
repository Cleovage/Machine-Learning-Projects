import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

def linear_regression_model(input_filepath):
    """
    Loads a dataset, splits it into training and testing sets,
    builds a linear regression model, and evaluates its performance.
    """
    df = pd.read_csv(input_filepath)

    # Assuming the dataset has 'YearsExperience' as feature and 'Salary' as target
    X = df[["YearsExperience"]]
    y = df["Salary"]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model's performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- Model Performance ---")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R2): {r2:.2f}")

    # Visualize the results
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color="blue", label="Actual Salary")
    plt.plot(X_test, y_pred, color="red", label="Predicted Salary")
    plt.title("Salary vs. Years of Experience (Linear Regression)")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.legend()
    plt.grid(True)
    plt.savefig("linear_regression_plot.png")
    plt.show()

if __name__ == "__main__":
    input_file = "../Salary_Data.csv"
    # Fix the path to use the correct location of Salary_Data.csv
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "..", "Salary_Data.csv")
    linear_regression_model(input_file)


