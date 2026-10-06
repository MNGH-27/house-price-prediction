import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "median_house_value"


def load_data(path):
    return pd.read_csv(path)


def split_data(df):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
