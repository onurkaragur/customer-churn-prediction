import pandas as pd
from config import DATA_RAW, DATA_PROCESSED
from utils import load_pickle


def load_raw():
    return pd.read_csv(DATA_RAW)


def load_processed():
    return load_pickle(DATA_PROCESSED)