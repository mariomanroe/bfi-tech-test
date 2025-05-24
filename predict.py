import pickle
import pandas as pd

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def predict(input_dict):
    df = pd.DataFrame([input_dict])
    return model.predict(df)