import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

DATA_RAW = os.path.join(BASE_DIR, "data/raw/telco-customer-churn.csv")
DATA_PROCESSED = os.path.join(BASE_DIR, "data/processed/preprocessed_data.pkl")

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_LOGREG = os.path.join(MODEL_DIR, "model_logreg.pkl")
MODEL_RF = os.path.join(MODEL_DIR, "model_rf.pkl")
MODEL_XGB = os.path.join(MODEL_DIR, "model_xgb.pkl")
PREPROCESSOR = os.path.join(MODEL_DIR, "preprocessor.pkl")