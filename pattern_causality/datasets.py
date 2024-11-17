import pandas as pd
import pkg_resources


def load_data():
    """Load the example dataset included with the package."""
    data_path = pkg_resources.resource_filename(
        "pattern_causality", "data/Climate_Indices.csv"
    )
    return pd.read_csv(data_path)
