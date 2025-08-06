# Predictive Maintenance for Metallurgical Equipment

## Project Summary
This project provides a step-by-step solution for Predictive Maintenance of metallurgical equipment (rolling mills, furnaces, etc.) using Python. It covers simulated sensor data generation, realistic fault modeling, comprehensive feature extraction, and Remaining Useful Life (RUL) prediction through machine learning and deep learning. It is inspired by MATLAB's Predictive Maintenance Toolbox, adapted for Python with an open, modular workflow.

## Workflow
1. Data Simulation
   - Simulate realistic sensor data using physics-based models (e.g., for rolling mills).
   - Multi-sensor support: temperature, vibration, pressure, current, etc.
2. Fault Modeling
   - Injects realistic faults (bearing wear, misalignment, lubrication loss, etc.).
   - Models progressive, sudden, and intermittent degradations.
3. Feature Extraction
   - Extracts time-domain, frequency-domain, statistical, and advanced features from sensor data.
4. Model Training & RUL Prediction
   - Supports classical ML (Random Forest, SVR), deep learning (LSTM, CNN), and ensemble methods.
   - Trains and evaluates models to predict Remaining Useful Life (RUL).

## Technologies Used
- NumPy, Pandas, Scikit-learn
- TensorFlow/Keras, (PyTorch optional)
- SciPy, PyWavelets, librosa
- Matplotlib, Seaborn, Plotly, Bokeh
- HDF5/PyTables, SQLite, YAML
- Jupyter, pytest, Black, flake8

## How to Use
1. Clone this repository and install dependencies (`requirements.txt`).
2. Simulate equipment sensor data with `data_simulation.py`.
3. Inject faults with `fault_modeling.py`.
4. Extract features with `feature_engineering.py`.
5. Train models and evaluate in `model_training.py`.
6. Make predictions with trained models using `predict.py`.

Example:
```python
from src.data_simulation import SensorSimulator
simulator = SensorSimulator(config)
sensor_data = simulator.generate_data()

from src.fault_modeling import FaultInjector
faulty_data = FaultInjector(fault_scenarios).apply_faults(sensor_data)

from src.feature_extraction import FeatureExtractor
features = FeatureExtractor().extract_features(faulty_data)

from src.models import RULPredictor
model = RULPredictor().train(features, targets)
prediction = model.predict(new_data)
```

---
## Author
- Name: Umar Shaik
- GitHub: Shaikumar2006
- Email: shaikmuhammadumar2006@gmail.com
- Year: 2025

MIT License.
