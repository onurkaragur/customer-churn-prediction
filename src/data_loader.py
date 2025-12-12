import pandas as pd
import pickle
from config import DATA_RAW, DATA_PROCESSED

def load_raw():
    return pd.read_csv(DATA_RAW)

def load_processed():
    return pickle.load(open(DATA_PROCESSED, "rb"))