import pandas as pd
import joblib

FEATURES = [
    'temperature', 'vibration', 'pressure',
    'temp_mean', 'vib_mean', 'pres_mean',
    'temp_std', 'vib_std', 'pres_std'
]

def predict_sample(input_dict):
    reg = joblib.load("regressor.pkl")
    clf = joblib.load("classifier.pkl")
    df = pd.DataFrame([input_dict])
    pred_rul = reg.predict(df[FEATURES])[0]
    pred_faulty = clf.predict(df[FEATURES])[0]
    return pred_rul, pred_faulty

# Example usage:
if __name__ == "__main__":
    # Simulate a sample input. In practice, replace with user input
    sample = {
        'temperature': 100, 'vibration': 20, 'pressure': 90,
        'temp_mean': 99, 'vib_mean': 19, 'pres_mean': 91,
        'temp_std': 0.2, 'vib_std': 0.05, 'pres_std': 0.3
    }
    rul, faulty = predict_sample(sample)
    print(f"Predicted RUL: {rul:.1f}, Fault Status: {'FAULTY' if faulty else 'Healthy'}")
