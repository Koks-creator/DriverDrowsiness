import os
import cv2
import numpy as np
import csv
import pandas as pd
from sklearn.model_selection import train_test_split  # pozwoli na podzielenie danych na do trenowania i do testow
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler  # standaryzuje dane
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pickle
from sklearn.metrics import accuracy_score


DATA_FILE = r"coordsYawn.csv"

df = pd.read_csv(DATA_FILE)
x = df.drop('class', axis=1)
y = df['class']
print(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)

pipelines = {
    # "lr": make_pipeline(StandardScaler(), LogisticRegression()),
    # "rc": make_pipeline(StandardScaler(), RidgeClassifier()),
    "rf": make_pipeline(StandardScaler(), RandomForestClassifier()),
    # "gb": make_pipeline(StandardScaler(), GradientBoostingClassifier())
}
# print(np.isnan(x))
fit_models = {}
print("Starting training")
for algo, pipeline in pipelines.items():
    print(f"Training with {algo}")
    model = pipeline.fit(x_train, y_train)
    fit_models[algo] = model
# print(fit_models)

for algo, model in fit_models.items():
    print(f"predict with {algo}")
    yhat = model.predict(x_test)
    print(algo, accuracy_score(y_test, yhat))

# print(y_test)

with open("res.pkl", "wb") as f:
    pickle.dump(fit_models['rf'], f) #stestuj lr i rc
