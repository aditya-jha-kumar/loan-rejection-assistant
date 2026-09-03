import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Project root (parent of src/), so paths work regardless of CWD
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = ROOT / "data" / "loan_dataset.csv"

# DATA LOADING

def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded dataset: {df.shape[0]} rows and {df.shape[1]} columns")
    return df

# DATA CLEANING

def clean_data(df):
    df = df.copy()

    # REMOVING IMPOSSIBLE AGE VALUES
    before = len(df)
    df = df[df["person_age"] <= 100]
    print(f"Removed {before - len(df)} rows with impossible age values")

    # REMOVING IMPOSSIBLE EMPLOYMENT LENGTH
    # Keep missing emp_length so it can be imputed; NaN <= 60 is False
    # and would otherwise drop those rows instead of filling them.
    emp = df["person_emp_length"]
    before_emp = len(df)
    df = df[emp.isna() | (emp <= 60)].copy()
    print(f"Removed {before_emp - len(df)} rows with impossible employment length")
    print(f"Dataset after outlier removal: {df.shape[0]} rows")

    # HANDLING MISSING VALUES
    # emp_length - numeric, to be filled with median because of outliers
    df["person_emp_length"] = df["person_emp_length"].fillna(df["person_emp_length"].median())

    # loan_int_rate — numeric, fill with median
    # Interest rate correlates with loan grade so median is safe
    df["loan_int_rate"] = df["loan_int_rate"].fillna(df["loan_int_rate"].median())

    # REMOVING DUPLICATES
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed > 0:
        print(f"Removed {removed} duplicate rows")

    # Verify — no missing values should remain
    remaining = df.isna().sum().sum()
    if remaining > 0:
        print(f"WARNING: {remaining} missing values still remain")
    else:
        print("Clean: no missing values")

    return df


# DATA ENCODING

def encode_features(df):
    df = df.copy()

    # loan_grade — Label Encoding encoding because A > B > C > D > E > F > G
    grade_map = {"A": 6, "B": 5, "C": 4, "D": 3, "E": 2, "F": 1, "G": 0}
    df["loan_grade"] = df["loan_grade"].map(grade_map)

    # cb_person_default_on_file — binary label encoding
    # Y (has defaulted before) = 1, N (clean record) = 0
    df["cb_person_default_on_file"] = df["cb_person_default_on_file"].map(
        {"Y": 1, "N": 0}
    )

    # person_home_ownership and loan_intent — One-Hot Encoded
    # No natural order between RENT/OWN/MORTGAGE or PERSONAL/MEDICAL etc.
    # drop_first=True avoids the dummy variable trap
    df = pd.get_dummies(
        df, columns = ["person_home_ownership", "loan_intent"], 
        drop_first=True
    )

    # Convert bool columns to int — XGBoost requires numeric types
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    print(f"Encoded: {df.shape[1]} columns, all numeric")
    print(f"Columns: {df.columns.tolist()}")

    return df
    
# SPLIT FEATURES AND TARGET

def split_features(df):
    X = df.drop("loan_status", axis=1).astype(float)
    y = df["loan_status"].astype(int)

    print(f"\nFeatures (X): {X.shape}")
    print(f"Target   (y): {y.shape}")
    print(f"\nClass distribution:")
    print(y.value_counts())
    print(f"\nImbalance ratio: {y.value_counts()[0] / y.value_counts()[1]:.1f}:1")

    return X, y

# TRAIN/TEST SPLIT

def get_train_test(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size = test_size,
        random_state = random_state,
        stratify = y
    )

    print(f"\nTraining set: {X_train.shape[0]} rows")
    print(f"Test set:     {X_test.shape[0]} rows")
    print(f"\nClass ratio in test set:")
    print(y_test.value_counts())

    return X_train, X_test, y_train, y_test

# FULL PIPELINE
def run_pipeline(path):
    df = load_data(path)
    df = clean_data(df)
    df = encode_features(df)
    X, y = split_features(df)
    X_train, X_test, y_train, y_test = get_train_test(X, y)

    return X_train, X_test, y_train, y_test, X, y

# QUICK TEST
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, X, y = run_pipeline(DEFAULT_DATA_PATH)
    print("\nPipeline complete. Ready for model training.")
    print(f"Final feature set: {X.columns.tolist()}")