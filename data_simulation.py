import numpy as np
import pandas as pd

def simulate_sensor_data(num_samples=10_000):
    np.random.seed(0)
    cycles = np.arange(num_samples)
    temperature = 70 + 0.01 * cycles + np.random.normal(0, 0.5, num_samples)
    vibration = 0.02 * cycles + np.random.normal(0, 0.1, num_samples)
    pressure = 100 - 0.005 * cycles + np.random.normal(0, 0.3, num_samples)
    fault_cycle = 8_000
    vibration[fault_cycle:] += 0.05 * (cycles[fault_cycle:] - fault_cycle)
    temperature[fault_cycle:] += 0.02 * (cycles[fault_cycle:] - fault_cycle)

    df = pd.DataFrame({
        'cycle': cycles,
        'temperature': temperature,
        'vibration': vibration,
        'pressure': pressure,
    })
    df['RUL'] = num_samples - df['cycle']
    return df

if __name__ == "__main__":
    data = simulate_sensor_data()
    data.to_csv("simulated_sensor_data.csv", index=False)
