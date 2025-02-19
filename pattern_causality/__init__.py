"""Pattern Causality Analysis Package.

This package provides tools for analyzing causal relationships in time series data.
"""

from importlib.metadata import version, metadata

# Get package metadata
__version__ = version("pattern-causality")
__author__ = metadata("pattern-causality").get("Author")
__email__ = metadata("pattern-causality").get("Author-email")
__license__ = metadata("pattern-causality").get("License")
__copyright__ = f"Copyright (c) 2024 {__author__}"

# Import core classes
from .pattern_causality import pattern_causality
from .datasets import load_data, get_dataset_info

# Import C++ extensions
try:
    from utils.databank import databank
    from utils.distancematrix import distancematrix
    from utils.fcp import fcp
    from utils.fillPCMatrix import fillPCMatrix
    from utils.natureOfCausality import natureOfCausality
    from utils.pastNNs import pastNNs
    from utils.patternhashing import patternhashing
    from utils.patternspace import patternspace
    from utils.predictionY import predictionY
    from utils.projectedNNs import projectedNNs
    from utils.signaturespace import signaturespace
    from utils.statespace import statespace
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import C++ extensions: {str(e)}")

__all__ = [
    "pattern_causality",
    "load_data",
    "get_dataset_info",
    "databank",
    "distancematrix",
    "fcp",
    "fillPCMatrix",
    "natureOfCausality",
    "pastNNs",
    "patternhashing",
    "patternspace",
    "predictionY",
    "projectedNNs",
    "signaturespace",
    "statespace",
]
