import pandas as pd

def add_features(df, window=50):
    df['temp_mean'] = df['temperature'].rolling(window).mean()
    df['vib_mean'] = df['vibration'].rolling(window).mean()
    df['pres_mean'] = df['pressure'].rolling(window).mean()
    df['temp_std'] = df['temperature'].rolling(window).std()
    df['vib_std'] = df['vibration'].rolling(window).std()
    df['pres_std'] = df['pressure'].rolling(window).std()
    df = df.dropna()
    return df

if __name__ == "__main__":
    df = pd.read_csv("simulated_sensor_data.csv")
    df = add_features(df)
    df.to_csv("features.csv", index=False)
