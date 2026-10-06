from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model
