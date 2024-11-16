# pattern_causality_py

[![PyPI version](https://img.shields.io/pypi/v/pattern-causality.svg)](https://badge.fury.io/py/pattern-causality)
[![Tests](https://github.com/skstavroglou/pattern_causality_py/actions/workflows/tests.yml/badge.svg)](https://github.com/skstavroglou/pattern_causality_py/actions/workflows/tests.yml)
[![Coverage](https://img.shields.io/badge/coverage-79%25-yellow.svg)](https://github.com/skstavroglou/pattern_causality_py)
[![License](https://img.shields.io/pypi/l/pattern-causality.svg)](https://github.com/skstavroglou/pattern_causality_py/blob/main/LICENSE)

`pattern_causality` is a powerful Python library implementing the Pattern Causality algorithm for analyzing causal relationships in time series data. This package provides efficient tools for detecting and quantifying causality patterns between multiple time series, with particular emphasis on nonlinear complex systems.

## Key Features

- **Lightweight Analysis**: Fast causality detection between pairs of time series
- **Parameter Optimization**: Automated search for optimal embedding parameters
- **Cross-validation Support**: Robust validation of causality results
- **Matrix Analysis**: Efficient computation of causality matrices for multiple time series
- **Effect Analysis**: Advanced tools for analyzing causal effects in complex systems

## Installation

### Using pip (Recommended)
The easiest way to install the package is via pip:

```bash
pip install pattern-causality
```

### From Source
For the latest development version, you can install directly from GitHub:

```bash
pip install git+https://github.com/skstavroglou/pattern_causality_py.git
```

## Quick Start Guide

### Loading Sample Data
The package comes with a built-in climate indices dataset for testing and demonstration:

```python
from pattern_causality import load_data

# Load the included climate indices dataset
data = load_data()
print("Available climate indices:", data.columns.tolist())
```

### Basic Causality Analysis
Perform a basic causality analysis between two time series using the lightweight implementation:

```python
from pattern_causality import pc_lightweight

# Load data
data = load_data()

# Example using two climate indices
X = data['NAO'].values  # North Atlantic Oscillation
Y = data['AAO'].values  # Arctic Oscillation

# Run lightweight pattern causality analysis
# Parameters:
# - E: embedding dimension
# - tau: time delay
# - h: prediction horizon
# - metric: distance metric, default is "euclidean"
# - weighted: whether to use weighted causality, default is True
result = pc_lightweight(X=X, Y=Y, E=3, tau=1, h=1)
print("Causality strength:\n", result)
```

### Parameter Optimization
Find the optimal parameters for your specific dataset:

```python
from pattern_causality import optimal_parameters_search

data = load_data()
# Search for best parameters up to Emax and tau_max
result = optimal_parameters_search(
    Emax=5, 
    tau_max=5, 
    metric="euclidean", 
    dataset=data.drop(columns=['Date'])
)
print("Optimal parameters:\n", result)
```

### Cross-Validation Analysis
Validate your causality results using different sample sizes:

```python
from pattern_causality import pc_cross_validation

data = load_data()
X = data['NAO'].values
Y = data['AAO'].values

# Perform cross-validation with different sample sizes
cv_results = pc_cross_validation(
    X=X,
    Y=Y,
    E=3,
    tau=1,
    metric="euclidean",
    h=1,
    weighted=True,
    numberset=[100, 200, 300, 400, 500]  # Different sample sizes
)
print("Cross-validation results:\n", cv_results)
```

### Multi-Series Analysis
Analyze causality patterns between multiple time series simultaneously:

```python
from pattern_causality import pc_matrix

data = load_data()
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
print("\nNegative causality matrix:", results['negative'])
print("\nDark causality matrix:", results['dark'])
print("\nVariable names:", results['items'])
```

### Effect Analysis
Calculate and analyze the causal effects between different time series:

```python
from pattern_causality import pc_matrix, pc_effect

# Load data and calculate pc_matrix
data = load_data()
pc_matrix_results = pc_matrix(
    dataset=data.drop(columns=['Date']),
    E=3,
    tau=1,
    metric="euclidean",
    h=1,
    weighted=True
)

# Calculate effects
effects = pc_effect(pc_matrix_results)
print("Pattern Causality Effects:")
print("\nPositive effects:", effects['positive'])
print("\nNegative effects:", effects['negative'])
print("\nDark effects:", effects['dark'])
```
## Testing

This package includes a comprehensive test suite. To run the tests:

```bash
## Install test dependencies
pip install pytest pytest-cov
## Run tests
python -m pytest tests/
## Run tests with coverage report
python -m pytest tests/ --cov=pattern_causality
```


### Test Coverage

The test suite covers:
- Basic functionality tests
- Advanced functionality tests
- Utility function tests

Current test coverage: 79%

## Contributing
We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Development

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