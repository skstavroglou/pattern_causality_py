# pattern_causality_py

[![PyPI version](https://img.shields.io/pypi/v/pattern-causality.svg)](https://badge.fury.io/py/pattern-causality)
[![Downloads](https://pepy.tech/badge/pattern-causality)](https://pepy.tech/project/pattern-causality)
[![Tests](https://github.com/skstavroglou/pattern_causality_py/actions/workflows/tests.yml/badge.svg)](https://github.com/skstavroglou/pattern_causality_py/actions/workflows/tests.yml)
[![Coverage](https://img.shields.io/badge/coverage-79%25-yellow.svg)](https://github.com/skstavroglou/pattern_causality_py)
[![License](https://img.shields.io/pypi/l/pattern-causality.svg)](https://github.com/skstavroglou/pattern_causality_py/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Python](https://img.shields.io/pypi/pyversions/pattern-causality.svg)](https://pypi.org/project/pattern-causality/)

## Overview

`pattern_causality` is a comprehensive Python library that implements the Pattern Causality algorithm for analyzing causal relationships in time series data. This package provides efficient tools for detecting and quantifying causality patterns between multiple time series, with a particular focus on nonlinear complex systems.

## Key Features

- **Efficient Causality Detection**: Robust analysis of causal relationships between pairs of time series
- **Parameter Optimization**: Automated identification of optimal embedding parameters
- **Cross-validation Support**: Statistical validation of causality results
- **Matrix Analysis**: Comprehensive computation of causality matrices for multiple time series
- **Effect Analysis**: Sophisticated tools for analyzing causal effects in complex systems

## Installation

### Via pip (Recommended)
```bash
pip install pattern-causality
```

### From Source
For the latest development version:
```bash
pip install git+https://github.com/skstavroglou/pattern_causality_py.git
```

## Usage Guide

### Loading Data
The package includes a pre-processed climate indices dataset for demonstration:

```python
from pattern_causality import load_data

# Load the included climate indices dataset
data = load_data()
print("Available climate indices:", data.columns.tolist())
```

### Basic Causality Analysis
Perform causality analysis between two time series:

```python
from pattern_causality import pc_lightweight

# Prepare data
data = load_data()
X = data['NAO'].values  # North Atlantic Oscillation
Y = data['AAO'].values  # Arctic Oscillation

# Perform pattern causality analysis
result = pc_lightweight(
    X=X, 
    Y=Y, 
    E=3,          # embedding dimension
    tau=1,        # time delay
    h=1,          # prediction horizon
    metric="euclidean",  # distance metric
    weighted=True        # use weighted causality
)
print("Causality Analysis Results:\n", result)
```

The `weighted` parameter determines the causality strength calculation method:
- `weighted=True`: Utilizes the error function (erf) to normalize causality strength
- `weighted=False`: Uses binary causality strength (1 for presence, 0 for absence)

### Parameter Optimization
Identify optimal parameters for your dataset:

```python
from pattern_causality import optimal_parameters_search

data = load_data()
result = optimal_parameters_search(
    Emax=5,       # maximum embedding dimension
    tau_max=5,    # maximum time delay
    metric="euclidean",
    dataset=data.drop(columns=['Date'])
)
print("Optimal Parameters:\n", result)
```

### Cross-Validation Analysis
Validate causality results across different sample sizes:

```python
from pattern_causality import pc_cross_validation

result = pc_cross_validation(
    X=data['NAO'].values,
    Y=data['AAO'].values,
    E=3,
    tau=1,
    metric="euclidean",
    h=1,
    weighted=True,
    numberset=[100, 200, 300, 400, 500]  # sample sizes
)
print("Cross-validation Results:\n", result)
```

### Multi-Series Analysis
Analyze causality patterns across multiple time series:

```python
from pattern_causality import pc_matrix

results = pc_matrix(
    dataset=data.drop(columns=['Date']),
    E=3,
    tau=1,
    metric="euclidean",
    h=1,
    weighted=True
)

print("Pattern Causality Matrix Results:")
print("Positive causality matrix:", results['positive'])
print("Negative causality matrix:", results['negative'])
print("Dark causality matrix:", results['dark'])
print("Variable names:", results['items'])
```

### Effect Analysis
Analyze causal effects between time series:

```python
from pattern_causality import pc_matrix, pc_effect

# Calculate causality matrix
matrix_results = pc_matrix(
    dataset=data.drop(columns=['Date']),
    E=3,
    tau=1,
    metric="euclidean",
    h=1,
    weighted=True
)

# Analyze effects
effects = pc_effect(matrix_results)
print("Causal Effects Analysis:")
print("Positive effects:", effects['positive'])
print("Negative effects:", effects['negative'])
print("Dark effects:", effects['dark'])
```

## Testing

The package includes comprehensive test coverage:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
python -m pytest tests/

# Run tests with coverage report
python -m pytest tests/ --cov=pattern_causality
```

Current test coverage: 79%

## Contributing

We welcome contributions! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Submit a pull request

For major changes, please open an issue first to discuss proposed modifications.

## Development Setup

1. Clone the repository
2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
3. Run tests:
   ```bash
   pytest
   ```

## References

- Stavroglou, S. K., Pantelous, A. A., Stanley, H. E., & Zuev, K. M.
  (2019). Hidden interactions in financial markets. _Proceedings of the
  National Academy of Sciences, 116(22)_, 10646-10651.

- Stavroglou, S. K., Pantelous, A. A., Stanley, H. E., & Zuev, K. M.
  (2020). Unveiling causal interactions in complex systems. _Proceedings
  of the National Academy of Sciences, 117(14)_, 7599-7605.

- Stavroglou, S. K., Ayyub, B. M., Kallinterakis, V., Pantelous, A. A.,
  & Stanley, H. E. (2021). A novel causal risk‐based decision‐making
  methodology: The case of coronavirus. _Risk Analysis, 41(5)_, 814-830.

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.