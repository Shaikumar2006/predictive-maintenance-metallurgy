import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib   # To save/load models

FEATURES = [
    'temperature', 'vibration', 'pressure',
    'temp_mean', 'vib_mean', 'pres_mean',
    'temp_std', 'vib_std', 'pres_std'
]

def train_models():
    df = pd.read_csv("features.csv")
    X = df[FEATURES]
    y_reg = df['RUL']
    fault_threshold = 2000
    y_cls = (df['RUL'] < fault_threshold).astype(int)
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X, y_reg)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y_cls)
    joblib.dump(reg, "regressor.pkl")
    joblib.dump(clf, "classifier.pkl")
    print("Models trained and saved.")

if __name__ == "__main__":
    train_models()
