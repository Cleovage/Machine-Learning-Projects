import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(input_filepath):
    """
    Loads a cleaned dataset and performs exploratory data analysis,
    including visualizations and summary statistics.
    """
    df = pd.read_csv(input_filepath)

    print("\n--- Dataset Info ---")
    df.info()

    print("\n--- Dataset Description ---")
    print(df.describe())

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    # Visualizations
    plt.figure(figsize=(12, 6))

    # Distribution of Age
    plt.subplot(1, 2, 1)
    sns.histplot(df["Age"], kde=True)
    plt.title("Distribution of Age")

    # Distribution of Fare
    plt.subplot(1, 2, 2)
    sns.histplot(df["Fare"], kde=True)
    plt.title("Distribution of Fare")
    plt.tight_layout()
    plt.savefig("age_fare_distribution.png")
    plt.show()

    plt.figure(figsize=(12, 6))
    # Survival by Sex
    plt.subplot(1, 2, 1)
    sns.barplot(x="Sex_male", y="Survived", data=df)
    plt.title("Survival Rate by Sex (0=Female, 1=Male)")

    # Survival by Pclass
    plt.subplot(1, 2, 2)
    sns.barplot(x="Pclass", y="Survived", data=df)
    plt.title("Survival Rate by Pclass")
    plt.tight_layout()
    plt.savefig("survival_by_sex_pclass.png")
    plt.show()

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig("correlation_heatmap.png")
    plt.show()

    print("\n--- Insights ---")
    print("1. Age and Fare distributions are shown.")
    print("2. Survival rates by Sex and Pclass are visualized.")
    print("3. A correlation heatmap provides insights into feature relationships.")

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    input_file = os.path.abspath(os.path.join(base_dir, "..", "task1_data_preprocessing", "cleaned_titanic.csv"))
    perform_eda(input_file)


