# Predictive Maintenance for Metallurgical Equipment

## Introduction

This project provides a comprehensive, step-by-step guide to building a **Predictive Maintenance solution for metallurgical equipment** (such as rolling mills and furnaces) using Python. The solution addresses the challenge of limited real-world sensor data by implementing simulated data generation, combined with advanced machine learning and deep learning techniques for equipment health monitoring and Remaining Useful Life (RUL) prediction.

The workflow is inspired by and adapted from MATLAB's Predictive Maintenance Toolbox, bringing these capabilities to the Python ecosystem with enhanced flexibility and customization options.

## Project Structure

```
predictive-maintenance-metallurgy/
├── data/
│   ├── raw/                    # Raw sensor data (simulated)
│   ├── processed/              # Cleaned and preprocessed data
│   └── features/               # Extracted features for ML models
├── src/
│   ├── data_simulation/        # Data simulation modules
│   │   ├── sensor_simulator.py
│   │   └── equipment_models.py
│   ├── fault_modeling/         # Fault injection and modeling
│   │   ├── fault_injector.py
│   │   └── degradation_models.py
│   ├── feature_extraction/     # Feature engineering
│   │   ├── time_domain.py
│   │   ├── frequency_domain.py
│   │   └── statistical_features.py
│   ├── models/                 # ML/DL models for RUL prediction
│   │   ├── classical_ml.py
│   │   ├── deep_learning.py
│   │   └── ensemble_methods.py
│   └── utils/                  # Utility functions
│       ├── data_preprocessing.py
│       ├── visualization.py
│       └── evaluation.py
├── notebooks/                  # Jupyter notebooks for experimentation
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_model_comparison.ipynb
├── config/                     # Configuration files
│   ├── simulation_config.yaml
│   └── model_config.yaml
├── tests/                      # Unit tests
├── requirements.txt            # Python dependencies
├── setup.py                   # Package setup
└── README.md                  # This file
```

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Shaikumar2006/predictive-maintenance-metallurgy.git
   cd predictive-maintenance-metallurgy
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package in development mode:**
   ```bash
   pip install -e .
   ```

### Quick Start

```python
from src.data_simulation import SensorSimulator
from src.fault_modeling import FaultInjector
from src.feature_extraction import FeatureExtractor
from src.models import RULPredictor

# Initialize components
simulator = SensorSimulator()
fault_injector = FaultInjector()
feature_extractor = FeatureExtractor()
rul_predictor = RULPredictor()

# Generate and process data
data = simulator.generate_sensor_data()
faulty_data = fault_injector.inject_faults(data)
features = feature_extractor.extract_features(faulty_data)

# Train model and predict RUL
model = rul_predictor.train(features)
rul_prediction = rul_predictor.predict(new_data)
```

## Usage

### 1. Data Simulation
Generate realistic sensor data for metallurgical equipment:

```python
from src.data_simulation import SensorSimulator

# Configure simulation parameters
config = {
    'equipment_type': 'rolling_mill',
    'sensors': ['temperature', 'vibration', 'pressure', 'current'],
    'sampling_rate': 1000,  # Hz
    'duration': 3600,       # seconds
    'noise_level': 0.05
}

# Generate data
simulator = SensorSimulator(config)
sensor_data = simulator.generate_data()
```

### 2. Fault Modeling
Inject realistic fault patterns into the simulated data:

```python
from src.fault_modeling import FaultInjector

# Define fault scenarios
fault_scenarios = {
    'bearing_wear': {'onset': 1800, 'severity': 'gradual'},
    'misalignment': {'onset': 2400, 'severity': 'sudden'},
    'lubrication_loss': {'onset': 3000, 'severity': 'progressive'}
}

# Apply fault injection
fault_injector = FaultInjector(fault_scenarios)
faulty_data = fault_injector.apply_faults(sensor_data)
```

### 3. Feature Extraction
Extract meaningful features from sensor data:

```python
from src.feature_extraction import FeatureExtractor

# Initialize feature extractor
extractor = FeatureExtractor()

# Extract various types of features
features = {
    'time_domain': extractor.time_domain_features(data),
    'frequency_domain': extractor.frequency_domain_features(data),
    'statistical': extractor.statistical_features(data),
    'wavelet': extractor.wavelet_features(data)
}
```

### 4. Model Training for RUL Prediction
Train machine learning models for RUL prediction:

```python
from src.models import RULPredictor
from sklearn.model_selection import train_test_split

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(features, rul_targets, test_size=0.2)

# Train models
predictor = RULPredictor()
models = {
    'random_forest': predictor.train_random_forest(X_train, y_train),
    'lstm': predictor.train_lstm(X_train, y_train),
    'cnn': predictor.train_cnn(X_train, y_train)
}

# Evaluate models
results = predictor.evaluate_models(models, X_test, y_test)
```

## Workflow

The complete predictive maintenance workflow consists of four main stages:

### 1. Data Simulation
- **Purpose**: Generate realistic sensor data to overcome limited real-world data availability
- **Components**: 
  - Equipment physics models (rolling mills, furnaces, etc.)
  - Multi-sensor simulation (temperature, vibration, pressure, electrical)
  - Noise and environmental factor modeling
- **Output**: Time-series sensor data with normal operating conditions

### 2. Fault Modeling
- **Purpose**: Introduce realistic fault patterns and degradation scenarios
- **Fault Types**:
  - Bearing wear and fatigue
  - Mechanical misalignment
  - Lubrication system failures
  - Thermal degradation
  - Electrical anomalies
- **Degradation Models**: Progressive, sudden, and intermittent fault patterns
- **Output**: Sensor data with labeled fault conditions and RUL targets

### 3. Feature Extraction
- **Time Domain Features**: RMS, peak-to-peak, crest factor, skewness, kurtosis
- **Frequency Domain Features**: FFT analysis, power spectral density, spectral centroid
- **Statistical Features**: Mean, variance, correlation coefficients
- **Advanced Features**: Wavelet coefficients, envelope analysis, trend analysis
- **Output**: Comprehensive feature matrix for machine learning

### 4. Model Training for RUL Prediction
- **Classical ML**: Random Forest, Support Vector Regression, Gradient Boosting
- **Deep Learning**: LSTM networks, CNN for signal processing, hybrid models
- **Ensemble Methods**: Model averaging, stacking, boosting
- **Evaluation**: RMSE, MAE, prognostic horizon, alpha-lambda metrics
- **Output**: Trained models capable of predicting equipment RUL

## Technologies Used

### Core Libraries
- **NumPy**: Numerical computing and array operations
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Classical machine learning algorithms
- **TensorFlow/Keras**: Deep learning framework for neural networks
- **PyTorch**: Alternative deep learning framework (optional)

### Signal Processing
- **SciPy**: Scientific computing and signal processing
- **PyWavelets**: Wavelet analysis
- **librosa**: Audio and signal analysis tools

### Visualization
- **Matplotlib**: Static plotting and visualization
- **Seaborn**: Statistical data visualization
- **Plotly**: Interactive plots and dashboards
- **Bokeh**: Interactive visualization library

### Data Management
- **HDF5/PyTables**: Efficient data storage
- **SQLite**: Lightweight database for metadata
- **YAML**: Configuration file management

### Development Tools
- **Jupyter**: Interactive development and experimentation
- **pytest**: Unit testing framework
- **Black**: Code formatting
- **flake8**: Code linting

## Contribution

We welcome contributions to improve and extend this predictive maintenance solution! Here's how you can contribute:

### Getting Started
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest tests/`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Contribution Guidelines
- **Code Style**: Follow PEP 8 guidelines, use Black for formatting
- **Documentation**: Add docstrings for new functions and classes
- **Testing**: Include unit tests for new features (aim for >80% coverage)
- **Commit Messages**: Use clear, descriptive commit messages
- **Issues**: Check existing issues before creating new ones

### Areas for Contribution
- [ ] Additional equipment models (compressors, turbines, etc.)
- [ ] New fault injection mechanisms
- [ ] Advanced feature extraction techniques
- [ ] Real-time prediction capabilities
- [ ] Web-based dashboard for monitoring
- [ ] Integration with industrial IoT platforms
- [ ] Performance optimization
- [ ] Documentation improvements

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v --cov=src

# Format code
black src/ tests/

# Lint code
flake8 src/ tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### MIT License Summary
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use
- ❌ Liability
- ❌ Warranty

---

## Acknowledgments

- Inspired by MATLAB's Predictive Maintenance Toolbox
- Thanks to the open-source community for the excellent Python libraries
- Industrial partners for domain expertise and validation

## Contact

For questions, suggestions, or collaborations:

- **Project Maintainer**: [Shaikumar2006](https://github.com/Shaikumar2006)
- **Issues**: [GitHub Issues](https://github.com/Shaikumar2006/predictive-maintenance-metallurgy/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Shaikumar2006/predictive-maintenance-metallurgy/discussions)

## Citation

If you use this project in your research or industrial applications, please cite:

```bibtex
@software{predictive_maintenance_metallurgy,
  author = {Shaikumar2006},
  title = {Predictive Maintenance for Metallurgical Equipment},
  url = {https://github.com/Shaikumar2006/predictive-maintenance-metallurgy},
  version = {1.0.0},
  year = {2025}
}
```

---

**⭐ Star this repository if you find it helpful!**

**🔧 Built with Python • 🏭 For Industrial Applications • 📊 Powered by Machine Learning**
