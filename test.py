import joblib
import pandas as pd
# Load the trained model from the file
clf = joblib.load('trained_model.joblib')

# 8. Use the trained model to make predictions for new data
