import os
import pandas as pd
from sklearn.datasets import load_breast_cancer

def prepare_data():
    # 1. Load Dataset
    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df['target'] = cancer.target

    # 2. Checks & Validations
    feature_count = len(cancer.feature_names)
    assert feature_count == 30, f"Expected 30 features, but got {feature_count}"
    assert df.isnull().sum().sum() == 0, "Null values found in the dataset!"

    # 3. Save CSV
    os.makedirs("data/raw", exist_ok=True)
    output_path = "data/raw/breast_cancer.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully saved to {output_path} with {df.shape[0]} rows and {df.shape[1]} columns.")

if __name__ == "__main__":
    prepare_data()