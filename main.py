from src.data import load_data, split_data
from src.preprocess import (
    fill_missing_bedrooms,
    encode_ocean_proximity,
    add_rooms_per_household,
)
from src.train import train_random_forest
from src.metrics import evaluate_regression


def main():
    df = load_data("data/housing.csv")

    X_train_raw, X_test_raw, y_train, y_test = split_data(df)

    X_train_clean, X_test_clean = fill_missing_bedrooms(
        X_train_raw,
        X_test_raw
    )

    X_train_encoded, X_test_encoded = encode_ocean_proximity(
        X_train_clean,
        X_test_clean
    )

    X_train_features, X_test_features = add_rooms_per_household(
        X_train_encoded,
        X_test_encoded
    )

    model = train_random_forest(
        X_train_features,
        y_train
    )

    predictions = model.predict(X_test_features)

    mae, rmse, r2 = evaluate_regression(
        y_test,
        predictions
    )

    print("Random Forest")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R²:", r2)


if __name__ == "__main__":
    main()
