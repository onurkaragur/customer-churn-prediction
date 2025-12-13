import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from config import DATA_PROCESSED, PREPROCESSOR

def preprocess(df):
    # Feature/Target
    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"No": 0, "Yes": 1})

    num_features = X.select_dtypes(include=["int64", "float64"]).columns
    cat_features = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor.fit(X_train)
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    output = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }

    pickle.dump(output, open(DATA_PROCESSED, "wb"))
    pickle.dump(preprocessor, open(PREPROCESSOR, "wb"))

    return output, preprocessor