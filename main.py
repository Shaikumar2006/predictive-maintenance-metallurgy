import data_simulation
import feature_engineering
import model_training
import predict

def main():
    print("Simulating sensor data...")
    df_raw = data_simulation.simulate_sensor_data()
    df_raw.to_csv("simulated_sensor_data.csv", index=False)
    print("Raw data simulated and saved.")

    print("Engineering features...")
    df_feat = feature_engineering.add_features(df_raw)
    df_feat.to_csv("features.csv", index=False)
    print("Features engineered and saved.")

    print("Training models...")
    model_training.train_models()

    print("Making sample prediction...")
    # Use the first row as a sample (or set your own)
    sample = df_feat.iloc[0][predict.FEATURES].to_dict()
    rul, faulty = predict.predict_sample(sample)
    print(f"Predicted RUL: {rul:.1f}, Fault Status: {'FAULTY' if faulty else 'Healthy'}")

if __name__ == "__main__":
    main()
