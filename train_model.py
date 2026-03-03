import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

data = pd.read_csv("dataset.csv")

X = data[["experience_years"]]
y = data["salary"]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "salary_model.pkl")

print("Model trained and saved as salary_model.pkl")
