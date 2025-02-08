"""Pattern Causality Datasets Module.

This module provides access to built-in datasets for pattern causality analysis.
The datasets included are:
    - Climate_Indices: A dataset containing climate oscillation indices for pattern causality analysis
"""

from typing import Dict
import pandas as pd
import os


def load_data() -> pd.DataFrame:
    """Load the Climate Indices dataset included with the package.
    
    This function loads the built-in Climate_Indices.csv dataset, which contains
    climate oscillation indices data suitable for pattern causality analysis.
    
    Returns:
        pd.DataFrame: A DataFrame containing the climate indices data with the following columns:
            - Date: The date of the observation (YYYY-MM-DD)
            - AO: Arctic Oscillation index
            - AAO: Antarctic Oscillation index
            - NAO: North Atlantic Oscillation index
            - PNA: Pacific North American index
            
    Example:
        >>> from pattern_causality import load_data
        >>> data = load_data()
        >>> print(data.shape)
        (535, 5)
    """
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Climate_Indices.csv")
    return pd.read_csv(data_path)


def get_dataset_info() -> Dict[str, str]:
    """Get information about the built-in Climate Indices dataset.
    
    Returns:
        Dict[str, str]: A dictionary containing dataset information with keys:
            - description: General description of the dataset
            - source: Source of the data
            - citation: Citation information
            - variables: Description of the variables
            
    Example:
        >>> from pattern_causality import get_dataset_info
        >>> info = get_dataset_info()
        >>> print(info['description'])
    """
    return {
        "description": "Climate Oscillation Indices dataset for pattern causality analysis",
        "source": "NOAA Climate Prediction Center",
        "citation": (
            "Please cite the Pattern Causality package and the NOAA Climate "
            "Prediction Center when using this dataset."
        ),
        "variables": (
            "AO: Arctic Oscillation index - A climate pattern characterized by winds circulating "
            "counterclockwise around the Arctic.\n"
            "AAO: Antarctic Oscillation index - Also known as the Southern Annular Mode (SAM), "
            "describing the north-south movement of the westerly wind belt around Antarctica.\n"
            "NAO: North Atlantic Oscillation index - The atmospheric pressure difference between "
            "the Azores and Iceland.\n"
            "PNA: Pacific North American index - A climate pattern reflecting large-scale changes "
            "in atmospheric wave patterns over North America."
        )
    }
