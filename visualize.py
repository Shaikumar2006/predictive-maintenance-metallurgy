import pandas as pd
import matplotlib.pyplot as plt
import joblib

FEATURES = [
    'temperature', 'vibration', 'pressure',
    'temp_mean', 'vib_mean', 'pres_mean',
    'temp_std', 'vib_std', 'pres_std'
]

def plot_sensor_trends():
    df = pd.read_csv("features.csv")
    plt.figure(figsize=(14, 5))
    plt.plot(df['cycle'], df['temperature'], label='Temperature', color='orange')
    plt.plot(df['cycle'], df['vibration'], label='Vibration', color='green')
    plt.plot(df['cycle'], df['pressure'], label='Pressure', color='blue')
    plt.xlabel('Cycle')
    plt.ylabel('Raw Value')
    plt.title('Raw Sensor Signals')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_feature_trends():
    df = pd.read_csv("features.csv")
    plt.figure(figsize=(14, 5))
    plt.plot(df['cycle'], df['temp_mean'], label='Temperature Mean')
    plt.plot(df['cycle'], df['vib_mean'], label='Vibration Mean')
    plt.plot(df['cycle'], df['pres_mean'], label='Pressure Mean')
    plt.xlabel('Cycle')
    plt.ylabel('Feature Value')
    plt.title('Rolling Mean Features')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_RUL():
    df = pd.read_csv("features.csv")
    plt.figure(figsize=(12, 4))
    plt.plot(df['cycle'], df['RUL'], color='purple', label='RUL')
    plt.xlabel('Cycle')
    plt.ylabel('Remaining Useful Life')
    plt.title('RUL Over Time')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_fault_status():
    df = pd.read_csv("features.csv")
    fault_threshold = 2000
    plt.figure(figsize=(14, 2))
    plt.plot(df['cycle'], (df['RUL'] < fault_threshold).astype(int), color='red', label='Faulty')
    plt.xlabel('Cycle')
    plt.ylabel('Fault Status')
    plt.yticks([0, 1], ['Healthy', 'Faulty'])
    plt.title('Fault Status Over Time')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model_file, model_type="regression"):
    importances = joblib.load(model_file).feature_importances_
    imp_df = pd.DataFrame({'feature': FEATURES, 'importance': importances}).sort_values('importance', ascending=True)
    plt.figure(figsize=(8, 5))
    plt.barh(imp_df['feature'], imp_df['importance'])
    plt.xlabel("Importance")
    plt.title(f"Feature Importances for {model_type.capitalize()} Model")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_sensor_trends()
    plot_feature_trends()
    plot_RUL()
    plot_fault_status()
    plot_feature_importance("regressor.pkl", "regression")
    plot_feature_importance("classifier.pkl", "classification")
