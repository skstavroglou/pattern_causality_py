# pattern_causality_py

[![PyPI version](https://badge.fury.io/py/pattern-causality.svg)](https://badge.fury.io/py/pattern-causality)
[![PyPI Downloads](https://static.pepy.tech/badge/pattern-causality)](https://pepy.tech/project/pattern-causality)
[![Tests](https://github.com/skstavroglou/pattern_causality_py/actions/workflows/tests.yml/badge.svg)](https://github.com/skstavroglou/pattern_causality_py/actions/workflows/tests.yml)
[![Lint](https://github.com/skstavroglou/pattern_causality_py/actions/workflows/lint.yml/badge.svg)](https://github.com/skstavroglou/pattern_causality_py/actions/workflows/lint.yml)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

`pattern_causality` is a comprehensive Python library that implements the Pattern Causality algorithm for analyzing causal relationships in time series data. This package provides efficient tools for detecting and quantifying causality patterns between multiple time series, with a particular focus on nonlinear complex systems.

## Key Features

- **Efficient C++ Implementation**: Core algorithms implemented in C++ for maximum performance
- **Comprehensive Analysis Tools**: 
  - Basic pattern causality analysis
  - Multivariate time series analysis
  - Cross-validation capabilities
  - Parameter optimization
  - Effect metrics calculation
- **Built-in Dataset**: Includes climate indices dataset for demonstration
- **OpenMP Support**: Parallel processing for improved performance
- **Extensive Testing**: Comprehensive test suite with high coverage

## System Requirements

- Python 3.8 or later
- C++ compiler with C++11 support
- OpenMP support (for parallel processing)
- NumPy 1.19.0 or later
- Pandas 1.0.0 or later

## Changelog

### Version 1.0.3 (2024-02-15)
- Fixed integer type conversion issue in natureOfCausality function for Windows compatibility
- Improved type handling for array data in pattern causality calculations
- Enhanced cross-platform compatibility for integer types

### Version 1.0.2 (2024-02-15)
- Changed default behavior to use relative differences (relative=True by default)
- Added relative parameter to signaturespace for choosing between relative and absolute differences
- Enhanced documentation for the new parameter
- Improved backward compatibility with absolute difference mode (relative=False)

### Version 1.0.1 (2024-02-14)
- Fixed type conversion issue in natureOfCausality function
- Improved compatibility with different system architectures by using np.int_
- Enhanced stability for array data type handling
- Fixed Python 3.8 compatibility issue with numpy integer types

## Installation

### Via pip (Recommended)
```bash
pip install pattern-causality
```

### Via pip + git
```bash
pip install git+https://github.com/skstavroglou/pattern_causality_py.git
```

### From Source
#### Prerequisites

#### On Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y g++ python3-dev libomp-dev build-essential
```

#### On macOS:
```bash
brew install libomp
```

### Installing the Package

```bash
# Install required Python packages
python -m pip install numpy pandas

# Install pattern-causality
python -m pip install -e .
```

## Usage Examples

### Basic Usage

```python
from pattern_causality import pattern_causality, load_data

# Load the included climate indices dataset
data = load_data()

# Initialize pattern causality analyzer
pc = pattern_causality(verbose=True)

# Analyze causality between NAO and AAO indices
result = pc.pc_lightweight(
    X=data["NAO"].values,
    Y=data["AAO"].values,
    E=3,          # embedding dimension
    tau=1,        # time delay
    metric="euclidean",
    h=1,          # prediction horizon
    weighted=True, # use weighted calculations
    relative=True  # use relative differences (default)
)

print(result)
```

### Multivariate Analysis

```python
# Analyze causality patterns across multiple variables
matrix_result = pc.pc_matrix(
    dataset=data.drop(columns=["Date"]),
    E=3,
    tau=1,
    metric="euclidean",
    h=1,
    weighted=True,
    relative=True  # Using relative differences (default)
)

print("Pattern Causality Matrix Results:")
print(matrix_result)
```

### Parameter Optimization

```python
# Find optimal parameters
optimal_params = pc.optimal_parameters_search(
    Emax=5,
    tau_max=3,
    metric="euclidean",
    h=1,
    dataset=data.drop(columns=["Date"])
)

print("Optimal Parameters:")
print(optimal_params)
```

### Cross Validation

```python
# Perform cross-validation
cv_results = pc.pc_cross_validation(
    X=data["NAO"].values,
    Y=data["AAO"].values,
    E=3,
    tau=1,
    metric="euclidean",
    h=1,
    weighted=True,
    numberset=[100, 200, 300]
)

print("Cross-validation Results:")
print(cv_results)
```

## Development

### Setting Up Development Environment

1. Clone the repository:
```bash
git clone https://github.com/skstavroglou/pattern_causality_py.git
cd pattern_causality_py
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

3. Install development dependencies:
```bash
python -m pip install -e ".[dev]"
```

### Running Tests

```bash
# Run tests with coverage
python -m pytest tests/ --cov=pattern_causality -v
```

### Code Style

The project uses:
- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking

To check code style:
```bash
black .
isort .
flake8 .
mypy pattern_causality
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the test suite
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## References

- Stavroglou, S. K., Pantelous, A. A., Stanley, H. E., & Zuev, K. M. (2019). Hidden interactions in financial markets. _Proceedings of the National Academy of Sciences, 116(22)_, 10646-10651.
- Stavroglou, S. K., Pantelous, A. A., Stanley, H. E., & Zuev, K. M. (2020). Unveiling causal interactions in complex systems. _Proceedings of the National Academy of Sciences, 117(14)_, 7599-7605.
- Stavroglou, S. K., Ayyub, B. M., Kallinterakis, V., Pantelous, A. A., & Stanley, H. E. (2021). A novel causal risk‐based decision‐making methodology: The case of coronavirus. _Risk Analysis, 41(5)_, 814-830.

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.