def fill_missing_bedrooms(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    bedrooms_median = X_train["total_bedrooms"].median()

    X_train["total_bedrooms"] = X_train["total_bedrooms"].fillna(
        bedrooms_median
    )

    X_test["total_bedrooms"] = X_test["total_bedrooms"].fillna(
        bedrooms_median
    )

    return X_train, X_test

import pandas as pd


def encode_ocean_proximity(X_train, X_test):
    X_train = pd.get_dummies(
        X_train,
        columns=["ocean_proximity"],
        dtype=int
    )

    X_test = pd.get_dummies(
        X_test,
        columns=["ocean_proximity"],
        dtype=int
    )

    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    return X_train, X_test

def add_rooms_per_household(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train["rooms_per_household"] = (
        X_train["total_rooms"] / X_train["households"]
    )

    X_test["rooms_per_household"] = (
        X_test["total_rooms"] / X_test["households"]
    )

    return X_train, X_test

from sklearn.preprocessing import PolynomialFeatures


def add_geo_polynomial_features(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    geo_columns = ["latitude", "longitude"]

    poly = PolynomialFeatures(
        degree=2,
        include_bias=False
    )

    train_poly = poly.fit_transform(X_train[geo_columns])
    test_poly = poly.transform(X_test[geo_columns])

    feature_names = poly.get_feature_names_out(geo_columns)

    train_poly = pd.DataFrame(
        train_poly,
        columns=feature_names,
        index=X_train.index
    )

    test_poly = pd.DataFrame(
        test_poly,
        columns=feature_names,
        index=X_test.index
    )

    new_columns = [
        "latitude^2",
        "latitude longitude",
        "longitude^2"
    ]

    X_train = pd.concat(
        [X_train, train_poly[new_columns]],
        axis=1
    )

    X_test = pd.concat(
        [X_test, test_poly[new_columns]],
        axis=1
    )

    return X_train, X_test

def standardize_linear_features(X_train, X_test, columns):
    X_train = X_train.copy()
    X_test = X_test.copy()

    train_mean = X_train[columns].mean()
    train_std = X_train[columns].std()

    X_train[columns] = (
        X_train[columns] - train_mean
    ) / train_std

    X_test[columns] = (
        X_test[columns] - train_mean
    ) / train_std

    return X_train, X_test
