import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

# ==============================
# DATASET 1: UCI Heart Disease
# ==============================

df1 = pd.read_csv("heart.csv")

df1 = df1[
    [
        "age", "sex", "trestbps", "chol",
        "fbs", "restecg", "thalach",
        "exang", "target"
    ]
].copy()

# Rename to common schema
df1.columns = [
    "age", "sex", "bp", "chol",
    "sugar", "ecg", "heartrate",
    "exercise", "target"
]

# Add missing features (not present in UCI dataset)
df1["smoking"] = 0
df1["alcohol"] = 0


# ==============================
# DATASET 2: Heart Failure Prediction
# ==============================

df2 = pd.read_csv("heart(1).csv")

df2 = df2[
    [
        "Age", "Sex", "RestingBP", "Cholesterol",
        "FastingBS", "RestingECG", "MaxHR",
        "ExerciseAngina", "HeartDisease"
    ]
].copy()

df2.columns = [
    "age", "sex", "bp", "chol",
    "sugar", "ecg", "heartrate",
    "exercise", "target"
]

# Encode categorical variables
df2["sex"] = df2["sex"].map({"M": 1, "F": 0})
df2["exercise"] = df2["exercise"].map({"Y": 1, "N": 0})

# ECG: Normal = 0, abnormal = 1
df2["ecg"] = df2["ecg"].map({
    "Normal": 0,
    "ST": 1,
    "LVH": 1
})

# Add missing features
df2["smoking"] = 0
df2["alcohol"] = 0


# ==============================
# COMBINE DATASETS
# ==============================

df = pd.concat([df1, df2], ignore_index=True)

# ==============================
# HANDLE MISSING VALUES
# ==============================

# Numeric features → median
numeric_cols = ["age", "bp", "chol", "heartrate"]
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Binary / categorical → mode
binary_cols = ["sex", "sugar", "ecg", "exercise", "smoking", "alcohol"]
for col in binary_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# ==============================
# FEATURES & TARGET
# ==============================

FEATURES = [
    "age", "sex", "bp", "chol", "sugar",
    "ecg", "heartrate", "exercise",
    "smoking", "alcohol"
]

X = df[FEATURES]
y = df["target"]

# ==============================
# TRAIN / TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# MODEL TRAINING
# ==============================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# ==============================
# EVALUATION
# ==============================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 3))

# ==============================
# SAVE MODEL
# ==============================

joblib.dump(model, "heart_rf_model.pkl")
print("Model saved as heart_rf_model.pkl")

# ==============================
# PREDICTION FUNCTION
# ==============================

def predict_heart_disease(
    age, sex, bp, chol, sugar,
    ecg, heartrate, exercise,
    smoking=0, alcohol=0,
    model_path="heart_rf_model.pkl"
):
    model = joblib.load(model_path)

    data = pd.DataFrame([[
        age, sex, bp, chol, sugar,
        ecg, heartrate, exercise,
        smoking, alcohol
    ]], columns=FEATURES)

    prob = model.predict_proba(data)[0][1]
    return round(prob * 100, 2)


# ==============================
# SAMPLE RUN
# ==============================

if __name__ == "__main__":
    result = predict_heart_disease(
        age=55,
        sex=1,
        bp=140,
        chol=230,
        sugar=1,
        ecg=1,
        heartrate=150,
        exercise=1
    )

    print(f"Heart Disease Probability: {result}%")
