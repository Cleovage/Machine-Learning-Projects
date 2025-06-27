import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def preprocess_data(input_filepath, output_filepath):
    """
    Loads a dataset, handles missing values, normalizes numerical features,
    and saves the cleaned dataset.
    """
    df = pd.read_csv(input_filepath)

    # Handle missing values
    # For 'Age', fill missing values with the median
    df['Age'].fillna(df['Age'].median(), inplace=True)

    # Drop 'Cabin' column if it exists (it doesn't in this dataset, but good practice)
    if 'Cabin' in df.columns:
        df.drop('Cabin', axis=1, inplace=True)

    # Convert categorical features to numerical using one-hot encoding
    # Based on the CSV, 'Sex' is the only categorical column that needs one-hot encoding
    df = pd.get_dummies(df, columns=['Sex'], drop_first=True)

    # Normalize numerical features ('Age', 'Fare')
    scaler = MinMaxScaler()
    df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

    # Save the cleaned dataset
    df.to_csv(output_filepath, index=False)
    print(f"Cleaned data saved to {output_filepath}")

if __name__ == "__main__":
    # This block is for standalone execution, not when imported as a module.
    pass


