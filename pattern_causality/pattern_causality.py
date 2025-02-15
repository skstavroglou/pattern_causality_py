#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pattern Causality Analysis Package.

This module implements pattern causality analysis methods for time series data.
It provides tools for analyzing causal relationships between variables using pattern-based approaches.

The package includes methods for:
    - Basic pattern causality analysis
    - Multivariate time series analysis
    - Cross-validation and parameter optimization
    - Effect metrics calculation and visualization

Example:
    Basic usage example::

        >>> from pattern_causality import pattern_causality
        >>> pc = pattern_causality(verbose=True)
        >>> result = pc.pc_lightweight(X, Y, E=3, tau=1)
"""

from __future__ import annotations

# Standard library imports
import time
from typing import (
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from dataclasses import dataclass
from importlib.metadata import version, metadata

# Third-party imports
import numpy as np
import pandas as pd

# Local imports - using relative imports
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
    warnings.warn(f"Failed to import C++ modules: {str(e)}")
    # You might want to provide Python fallbacks here if available

# Package metadata
__version__ = version("pattern-causality")
__author__ = metadata("pattern-causality").get("Author")
__email__ = metadata("pattern-causality").get("Author-email")
__license__ = metadata("pattern-causality").get("License")
__copyright__ = f"Copyright (c) 2024 {__author__}"
__all__ = ['pattern_causality']

# Type aliases
T = TypeVar('T')
ArrayLike = Union[List[T], np.ndarray, pd.Series]
DatasetType = Union[pd.DataFrame, np.ndarray, List[T]]


@dataclass
class PCMatrixResult:
    """Data class for storing pattern causality matrix results.
    
    Attributes:
        positive (np.ndarray): Matrix of positive causality values
        negative (np.ndarray): Matrix of negative causality values
        dark (np.ndarray): Matrix of dark causality values
        items (list): List of variable names corresponding to matrix indices
    """
    positive: np.ndarray
    negative: np.ndarray
    dark: np.ndarray
    items: list


class pattern_causality:
    """Pattern Causality Analysis Class for Time Series Data.
    
    This class implements various pattern causality analysis methods for time series data.
    All methods return pandas DataFrames for consistency and ease of use.
    
    The class provides a comprehensive set of tools for analyzing causal relationships
    in time series data using pattern-based approaches.
    
    Attributes:
        verbose (bool): Whether to print detailed information during computation
        
    Methods:
        pc_lightweight: Basic pattern causality analysis for two time series
        pc_matrix: Calculate pattern causality matrix for multivariate time series
        pc_effect: Calculate effect metrics from pattern causality matrices
        pc_accuracy: Calculate pattern causality accuracy metrics
        pc_full_details: Detailed pattern causality analysis with time point information
        pc_cross_validation: Perform cross validation for pattern causality analysis
        optimal_parameters_search: Search for optimal E and tau parameters
        to_matrix: Convert flattened causality results to matrix format
        format_effects: Format effect results into matrices for visualization
        
    Note:
        All methods are designed to handle NaN values and invalid inputs gracefully.
        Error messages and warnings are provided when appropriate.
    """
    
    def __init__(self, verbose: bool = False):
        """Initialize pattern_causality class
        
        Args:
            verbose: Whether to print detailed information during computation
        """
        self.verbose = verbose

    @staticmethod
    def __version__():
        """Return the current version of the package"""
        from importlib.metadata import version
        return version("pattern-causality")
        
    def __repr__(self) -> str:
        """Return string representation of the class"""
        return f"pattern_causality(verbose={self.verbose})"
        
    def __str__(self) -> str:
        """Return string description of the class"""
        return "Pattern Causality Analysis Class for Time Series Data"

    def _print_if_verbose(self, message: str, verbose: bool = None) -> None:
        """
        Helper method to print messages when verbose is True
        
        Args:
            message: Message to print
            verbose: Override class-level verbose setting
        """
        verbose = self.verbose if verbose is None else verbose
        if verbose:
            print(message)

    def _calculate_basic_stats(self, X: Union[List, np.ndarray], Y: Union[List, np.ndarray]) -> Dict:
        """Calculate basic statistics for time series"""
        X, Y = np.array(X), np.array(Y)
        stats = {
            "X_mean": np.mean(X),
            "X_std": np.std(X),
            "Y_mean": np.mean(Y),
            "Y_std": np.std(Y),
            "correlation": np.corrcoef(X, Y)[0, 1],
            "X_length": len(X),
            "missing_values": np.sum(np.isnan(X)) + np.sum(np.isnan(Y))
        }
        return stats

    @staticmethod
    def _validate_input(X: Union[List, np.ndarray, pd.Series], 
                       Y: Union[List, np.ndarray, pd.Series]) -> tuple:
        """Validate and convert input time series to lists"""
        # Convert to numpy array first for type checking
        if isinstance(X, pd.Series):
            X = X.values
        elif isinstance(X, list):
            X = np.array(X)
            
        if isinstance(Y, pd.Series):
            Y = Y.values
        elif isinstance(Y, list):
            Y = np.array(Y)
            
        # Check if numeric
        if not np.issubdtype(X.dtype, np.number) or not np.issubdtype(Y.dtype, np.number):
            raise TypeError("All elements must be numeric")
            
        # Convert to list for processing
        return X.tolist(), Y.tolist()

    @staticmethod
    def _validate_dataset(dataset: Union[pd.DataFrame, np.ndarray, List]) -> pd.DataFrame:
        """Validate and convert dataset to DataFrame with numeric values"""
        if isinstance(dataset, np.ndarray):
            if not np.issubdtype(dataset.dtype, np.number):
                raise TypeError("All elements in array must be numeric")
            return pd.DataFrame(dataset)
        elif isinstance(dataset, list):
            arr = np.array(dataset).T
            if not np.issubdtype(arr.dtype, np.number):
                raise TypeError("All elements in list must be numeric")
            return pd.DataFrame(arr)
        elif isinstance(dataset, pd.DataFrame):
            if not all(dataset.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
                raise TypeError("All columns in DataFrame must be numeric")
            return dataset
        else:
            raise TypeError("dataset must be a DataFrame, numpy array, or list")

    @staticmethod
    def _validate_pc_matrix_result(result: Dict) -> bool:
        """Validate that the input is a result from pc_matrix"""
        required_keys = {"positive", "negative", "dark", "items"}
        
        if not all(key in result for key in required_keys):
            return False
            
        matrices = [result["positive"], result["negative"], result["dark"]]
        if not all(isinstance(m, np.ndarray) for m in matrices):
            return False
            
        shapes = [m.shape for m in matrices]
        if not all(len(shape) == 2 and shape[0] == shape[1] for shape in shapes):
            return False
            
        if not isinstance(result["items"], list) or len(result["items"]) != matrices[0].shape[0]:
            return False
            
        return True

    def pc_lightweight(self, 
                      X: Union[List, np.ndarray, pd.Series],
                      Y: Union[List, np.ndarray, pd.Series],
                      E: int,
                      tau: int,
                      metric: str = "euclidean",
                      h: int = 1,
                      weighted: bool = False,
                      relative: bool = True,
                      verbose: bool = None) -> pd.DataFrame:
        """
        Pattern Causality Lightweight implementation
        
        Args:
            X: Input time series (causal variable)
            Y: Input time series (affected variable)
            E: Embedding dimension
            tau: Time delay
            metric: Distance metric to use
            h: Prediction horizon
            weighted: Whether to use weighted calculations
            relative: Whether to use relative differences (default: False for absolute)
            verbose: Override class-level verbose setting
            
        Returns:
            DataFrame containing causality metrics
        """
        verbose = self.verbose if verbose is None else verbose
        start_time = time.time()
        
        X, Y = self._validate_input(X, Y)
        
        if verbose:
            stats = self._calculate_basic_stats(X, Y)
            self._print_if_verbose(f"\nInput Statistics:", verbose)
            self._print_if_verbose(f"X: mean={stats['X_mean']:.3f}, std={stats['X_std']:.3f}", verbose)
            self._print_if_verbose(f"Y: mean={stats['Y_mean']:.3f}, std={stats['Y_std']:.3f}", verbose)
            self._print_if_verbose(f"Correlation: {stats['correlation']:.3f}", verbose)
            self._print_if_verbose(f"Series length: {stats['X_length']}", verbose)
            if stats['missing_values'] > 0:
                self._print_if_verbose(f"Warning: {stats['missing_values']} missing values detected", verbose)
        
        # Initialize constants
        NNSPAN = E + 1
        CCSPAN = (E - 1) * tau
        hashedpatterns = patternhashing(E)
        
        if hashedpatterns is None or len(hashedpatterns) == 0:
            raise ValueError(f"Failed to generate hash patterns for E={E}")
            
        self._print_if_verbose(f"\nInitializing computation with E={E}, tau={tau}, h={h}", verbose)
        
        # Calculate shadow attractors
        self._print_if_verbose("Calculating state space and signatures...", verbose)
        Mx = statespace(X, E, tau)
        My = statespace(Y, E, tau)
        SMx = signaturespace(Mx, E, relative=relative)
        SMy = signaturespace(My, E, relative=relative)
        PSMx = patternspace(SMx, E)
        PSMy = patternspace(SMy, E)
        Dx = distancematrix(Mx, metric=metric)
        Dy = distancematrix(My, metric=metric)
        
        # Check time series length
        FCP = fcp(E, tau, h, X)
        al_loop_dur = range(FCP - 1, len(X) - (E - 1) * tau - h + 1)
        total_steps = len(al_loop_dur)
        
        self._print_if_verbose(f"\nProcessing time series...", verbose)
        self._print_if_verbose(f"Total time points to analyze: {total_steps}\n", verbose)
        
        # Initialize causality matrix
        predictedPCMatrix = databank("array", [3 ** (E - 1), 3 ** (E - 1), len(Y)])
        real_loop = None
        processed_points = 0
        valid_points = 0
        processable_points = 0  # Points that can be processed (no NaN, within bounds)
        
        # Main computation loop
        for i in al_loop_dur:
            processed_points += 1
            
            # Update progress every 10%
            progress_interval = max(1, total_steps // 10)
            if verbose and processed_points % progress_interval == 0:
                progress_percent = min(100, (processed_points/total_steps) * 100)  # Ensure we don't exceed 100%
                self._print_if_verbose(f"Progress: {processed_points}/{total_steps} points processed ({progress_percent:.1f}%)", verbose)
            
            if i + h >= len(My):
                continue
                
            # Check if point can be processed (no NaN values)
            if not np.any(np.isnan(Mx[i, :])) and not np.any(np.isnan(My[i + h, :])):
                processable_points += 1
                NNx = pastNNs(CCSPAN, NNSPAN, Mx, Dx, SMx, PSMx, i, h)
                
                if NNx is not None and not np.any(np.isnan(NNx["dists"])):
                    if not np.any(np.isnan(Dy[i, NNx["times"] + h])):
                        valid_points += 1
                        if real_loop is None:
                            real_loop = i
                        else:
                            real_loop = np.append(real_loop, i)
                            
                        projNNy = projectedNNs(My, Dy, SMy, PSMy, NNx["times"], i, h)
                        predicted_result = predictionY(E=E, projNNy=projNNy, zeroTolerance=E-1)
                        
                        # Get patterns and signatures
                        predictedSignatureY = predicted_result["predictedSignatureY"]
                        predictedPatternY = predicted_result["predictedPatternY"]
                        signatureX = SMx[i, :]
                        patternX = PSMx[i]
                        realSignatureY = SMy[i + h, :]
                        realPatternY = PSMy[i + h]
                        
                        # Calculate PC matrix values
                        pc = fillPCMatrix(
                            weighted=weighted,
                            predictedPatternY=predictedPatternY,
                            realPatternY=realPatternY,
                            predictedSignatureY=predictedSignatureY,
                            realSignatureY=realSignatureY,
                            patternX=patternX,
                            signatureX=signatureX
                        )
                        
                        # Find pattern indices
                        tolerance = 1e-10
                        hashedpatterns = np.array(hashedpatterns, dtype=np.float64)
                        patternX_val = np.float64(patternX.item())
                        predictedPatternY_val = np.float64(predictedPatternY)
                        
                        patternX_matches = np.where(np.abs(hashedpatterns - patternX_val) < tolerance)[0]
                        predictedPatternY_matches = np.where(np.abs(hashedpatterns - predictedPatternY_val) < tolerance)[0]
                        
                        if len(patternX_matches) > 0 and len(predictedPatternY_matches) > 0:
                            patternX_idx = patternX_matches[0]
                            predictedPatternY_idx = predictedPatternY_matches[0]
                            predictedPCMatrix[patternX_idx, predictedPatternY_idx, i] = pc["predicted"]
        
        # Print final progress update
        if verbose and processed_points > 0:
            self._print_if_verbose(f"Progress: {total_steps}/{total_steps} points processed (100.0%)\n", verbose)
        
        # Calculate causality metrics
        self._print_if_verbose("Calculating final causality metrics...", verbose)
        # Convert real_loop to integer type compatible with C++ NPY_LONG
        real_loop = np.array(real_loop, dtype=np.int_)
        causality = natureOfCausality(predictedPCMatrix, real_loop, hashedpatterns, X, weighted)
        
        # Calculate percentages
        totalCausPercent = 1 - np.nanmean(causality["noCausality"])
        mask = causality["noCausality"][real_loop] != 1
        
        if np.any(mask):
            valid_indices = real_loop[mask]
            valid_pos = causality["Positive"][valid_indices]
            valid_neg = causality["Negative"][valid_indices]
            valid_dark = causality["Dark"][valid_indices]
            
            valid_pos = valid_pos[~np.isnan(valid_pos)]
            valid_neg = valid_neg[~np.isnan(valid_neg)]
            valid_dark = valid_dark[~np.isnan(valid_dark)]
            
            posiCausPercent = np.mean(valid_pos) if len(valid_pos) > 0 else 0.0
            negaCausPercent = np.mean(valid_neg) if len(valid_neg) > 0 else 0.0
            darkCausPercent = np.mean(valid_dark) if len(valid_dark) > 0 else 0.0
            
            if weighted:
                total = posiCausPercent + negaCausPercent + darkCausPercent
                if total > 0:
                    posiCausPercent /= total
                    negaCausPercent /= total
                    darkCausPercent /= total
        else:
            posiCausPercent = negaCausPercent = darkCausPercent = 0.0
            
        end_time = time.time()
        if verbose:
            self._print_if_verbose(f"\nComputation completed in {end_time - start_time:.2f} seconds", verbose)
            self._print_if_verbose("\nProcessing Summary:", verbose)
            self._print_if_verbose(f"Total points analyzed: {total_steps}", verbose)
            self._print_if_verbose(f"Points with valid data: {processable_points}", verbose)
            self._print_if_verbose(f"Successfully processed: {valid_points}/{processable_points} ({(valid_points/processable_points)*100:.1f}%)", verbose)
            self._print_if_verbose("\nResults:", verbose)
            self._print_if_verbose(f"Total Causality: {totalCausPercent:.3f}", verbose)
            self._print_if_verbose(f"Positive Causality: {posiCausPercent:.3f}", verbose)
            self._print_if_verbose(f"Negative Causality: {negaCausPercent:.3f}", verbose)
            self._print_if_verbose(f"Dark Causality: {darkCausPercent:.3f}", verbose)
            
        return pd.DataFrame({
            "Total Causality": [totalCausPercent],
            "Positive Causality": [posiCausPercent],
            "Negative Causality": [negaCausPercent],
            "Dark Causality": [darkCausPercent]
        })

    def pc_matrix(self,
                  dataset: Union[pd.DataFrame, np.ndarray, List],
                  E: int,
                  tau: int,
                  metric: str = "euclidean",
                  h: int = 1,
                  weighted: bool = False,
                  relative: bool = True,
                  verbose: bool = None) -> pd.DataFrame:
        """
        Calculate pattern causality matrix for multivariate time series
        
        Args:
            dataset: Input dataset
            E: Embedding dimension
            tau: Time delay
            metric: Distance metric to use
            h: Prediction horizon
            weighted: Whether to use weighted calculations
            relative: Whether to use relative differences (default: False for absolute)
            verbose: Override class-level verbose setting
            
        Returns:
            pd.DataFrame: Flattened causality matrix where:
                         - Each row represents a pair of variables (from_var, to_var)
                         - Columns are ['from_var', 'to_var', 'positive', 'negative', 'dark']
                         - NaN values indicate self-causality (when from_var == to_var)
        """
        verbose = self.verbose if verbose is None else verbose
        start_time = time.time()
        
        dataset = self._validate_dataset(dataset)
        n_cols = dataset.shape[1]
        
        if verbose:
            self._print_if_verbose(f"\nAnalyzing dataset with {n_cols} variables", verbose)
            self._print_if_verbose(f"Parameters: E={E}, tau={tau}, h={h}", verbose)
            
            # Basic dataset statistics
            self._print_if_verbose("\nDataset Statistics:", verbose)
            for i in range(n_cols):
                col = dataset.iloc[:, i]
                self._print_if_verbose(f"Variable {i}: mean={col.mean():.3f}, std={col.std():.3f}", verbose)
        
        # Get variable names
        items = dataset.columns.tolist() if dataset.columns is not None else [f"Var_{i}" for i in range(n_cols)]
        
        # Initialize results list
        results = []
        
        total_pairs = n_cols * (n_cols - 1)
        processed_pairs = 0
        
        for i in range(n_cols):
            X = dataset.iloc[:, i].values.tolist()
            
            for j in range(n_cols):
                if i != j:
                    processed_pairs += 1
                    if verbose:
                        self._print_if_verbose(f"\nAnalyzing pair ({items[i]}, {items[j]}) - Progress: {processed_pairs}/{total_pairs}", verbose)
                    
                    if fcp(E, tau, h, X):
                        Y = dataset.iloc[:, j].values.tolist()
                        if fcp(E, tau, h, Y):
                            temp = self.pc_lightweight(
                                X=X,
                                Y=Y,
                                E=E,
                                tau=tau,
                                metric=metric,
                                h=h,
                                weighted=weighted,
                                relative=relative,
                                verbose=False
                            )
                            
                            # Store results in flattened format
                            results.append({
                                'from_var': items[i],
                                'to_var': items[j],
                                'positive': temp["Positive Causality"].values[0],
                                'negative': temp["Negative Causality"].values[0],
                                'dark': temp["Dark Causality"].values[0]
                            })
                            
                            if verbose:
                                self._print_if_verbose(f"Results for ({items[i]}, {items[j]}):", verbose)
                                self._print_if_verbose(f"  Positive: {results[-1]['positive']:.3f}", verbose)
                                self._print_if_verbose(f"  Negative: {results[-1]['negative']:.3f}", verbose)
                                self._print_if_verbose(f"  Dark: {results[-1]['dark']:.3f}", verbose)
                else:
                    # Add NaN values for self-causality
                    results.append({
                        'from_var': items[i],
                        'to_var': items[j],
                        'positive': np.nan,
                        'negative': np.nan,
                        'dark': np.nan
                    })
        
        if verbose:
            end_time = time.time()
            self._print_if_verbose(f"\nComputation completed in {end_time - start_time:.2f} seconds", verbose)
        
        # Create DataFrame from results
        result_df = pd.DataFrame(results)
        
        # Optional: Sort by from_var and to_var for consistency
        result_df = result_df.sort_values(['from_var', 'to_var']).reset_index(drop=True)
        
        return result_df

    def to_matrix(self, flat_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Convert flattened causality results to matrix format
        
        Args:
            flat_df: Flattened DataFrame from pc_matrix method
            
        Returns:
            Dictionary containing three matrices:
                - 'positive': Matrix of positive causality values
                - 'negative': Matrix of negative causality values
                - 'dark': Matrix of dark causality values
        """
        # Get unique variable names
        variables = sorted(list(set(flat_df['from_var'].unique()) | set(flat_df['to_var'].unique())))
        n = len(variables)
        
        # Initialize matrices
        matrices = {
            'positive': pd.DataFrame(np.nan, index=variables, columns=variables),
            'negative': pd.DataFrame(np.nan, index=variables, columns=variables),
            'dark': pd.DataFrame(np.nan, index=variables, columns=variables)
        }
        
        # Fill matrices
        for _, row in flat_df.iterrows():
            matrices['positive'].loc[row['from_var'], row['to_var']] = row['positive']
            matrices['negative'].loc[row['from_var'], row['to_var']] = row['negative']
            matrices['dark'].loc[row['from_var'], row['to_var']] = row['dark']
            
        return matrices

    def format_effects(self, effect_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Format effect results into matrices suitable for visualization
        
        This method transforms the effect results from pc_effect into a matrix format
        that is particularly suitable for visualization and analysis. The output matrices
        contain information about received and exerted effects, as well as their differences.
        
        Args:
            effect_df: DataFrame from pc_effect method containing causality effect metrics
            
        Returns:
            Dictionary containing three matrices:
                - 'positive': Matrix with columns [Received, Exerted, Difference]
                - 'negative': Matrix with columns [Received, Exerted, Difference]
                - 'dark': Matrix with columns [Received, Exerted, Difference]
                Each row represents a variable.
                
        Example:
            >>> pc = pattern_causality()
            >>> effects = pc.pc_effect(matrix_results)
            >>> effect_matrices = pc.format_effects(effects)
            >>> # Scatter plot for positive causality
            >>> plt.figure(figsize=(10, 6))
            >>> plt.scatter(effect_matrices['positive']['Received'], 
            ...           effect_matrices['positive']['Exerted'])
            >>> plt.xlabel('Received Effects')
            >>> plt.ylabel('Exerted Effects')
            >>> plt.title('Positive Causality: Received vs Exerted Effects')
            >>> plt.show()
            
        Raises:
            KeyError: If required columns are missing from effect_df
            ValueError: If effect_df has invalid structure
        """
        # Remove the 'Mean' row if it exists
        if 'Mean' in effect_df.index:
            effect_df = effect_df.drop('Mean')
            
        # Initialize the three matrices
        matrices = {
            'positive': pd.DataFrame(index=effect_df.index),
            'negative': pd.DataFrame(index=effect_df.index),
            'dark': pd.DataFrame(index=effect_df.index)
        }
        
        # Fill the matrices
        for causality_type in ['positive', 'negative', 'dark']:
            type_cap = causality_type.capitalize()
            matrices[causality_type]['Received'] = effect_df[f'{type_cap}_Received']
            matrices[causality_type]['Exerted'] = effect_df[f'{type_cap}_Exerted']
            matrices[causality_type]['Difference'] = effect_df[f'{type_cap}_Difference']
            
        return matrices

    def pc_effect(self, 
                  pcmatrix: pd.DataFrame,
                  verbose: bool = None) -> pd.DataFrame:
        """
        Calculate effect metrics from pattern causality matrices
        
        Args:
            pcmatrix: DataFrame from pc_matrix function (flattened format)
            verbose: Override class-level verbose setting
            
        Returns:
            pd.DataFrame: Effect metrics for each variable with columns:
                         [Positive/Negative/Dark]_[Received/Exerted/Difference]
        """
        verbose = self.verbose if verbose is None else verbose
        
        # Convert to matrix format first
        matrices = self.to_matrix(pcmatrix)
        
        if verbose:
            self._print_if_verbose("\nCalculating causality effects...", verbose)
            n_vars = len(matrices['positive'])
            self._print_if_verbose(f"Number of variables: {n_vars}", verbose)
        
        # Initialize results dictionary
        results = {}
        variables = matrices['positive'].index
        
        # Calculate metrics for each causality type
        for causality_type in ['positive', 'negative', 'dark']:
            matrix = matrices[causality_type].values
            
            # Calculate metrics
            received = np.nansum(matrix, axis=0) * 100  # Sum along rows (received effects)
            exerted = np.nansum(matrix, axis=1) * 100   # Sum along columns (exerted effects)
            diff = received - exerted
            
            # Store results
            results[f'{causality_type.capitalize()}_Received'] = received
            results[f'{causality_type.capitalize()}_Exerted'] = exerted
            results[f'{causality_type.capitalize()}_Difference'] = diff
            
            if verbose:
                self._print_if_verbose(f"\n{causality_type.capitalize()} Effects:", verbose)
                self._print_if_verbose(f"Mean Received: {np.nanmean(received):.2f}%", verbose)
                self._print_if_verbose(f"Mean Exerted: {np.nanmean(exerted):.2f}%", verbose)
                self._print_if_verbose(f"Mean Difference: {np.nanmean(diff):.2f}%", verbose)
                
                # Add detailed statistics for top effects
                if np.any(~np.isnan(received)):
                    self._print_if_verbose("\nTop Variables by Effect:", verbose)
                    self._print_if_verbose("Received Effects:", verbose)
                    sorted_idx = np.argsort(received)[-3:]
                    for idx in sorted_idx[::-1]:
                        self._print_if_verbose(f"  {variables[idx]}: {received[idx]:.2f}%", verbose)
                    
                    self._print_if_verbose("Exerted Effects:", verbose)
                    sorted_idx = np.argsort(exerted)[-3:]
                    for idx in sorted_idx[::-1]:
                        self._print_if_verbose(f"  {variables[idx]}: {exerted[idx]:.2f}%", verbose)
        
        # Create DataFrame with results
        result_df = pd.DataFrame(results, index=variables)
        
        # Add summary row
        summary = pd.DataFrame({
            col: np.nanmean(result_df[col])
            for col in result_df.columns
        }, index=['Mean'])
        
        result_df = pd.concat([result_df, summary])
        
        return result_df

    def pc_accuracy(self,
                   dataset: Union[pd.DataFrame, np.ndarray, List],
                   E: int,
                   tau: int,
                   metric: str,
                   h: int,
                   weighted: bool,
                   relative: bool = True) -> pd.DataFrame:
        """
        Calculate pattern causality accuracy metrics for a dataset.

        Args:
            dataset: Input dataset (DataFrame, numpy array, or list)
            E: Embedding dimension
            tau: Time delay
            metric: Distance metric to use
            h: Prediction horizon
            weighted: Whether to use weighted calculations
            relative: Whether to use relative differences (default: False for absolute)

        Returns:
            pd.DataFrame: Accuracy metrics with shape (1, 6)
                         Columns: ['E', 'tau', 'total', 'positive', 'negative', 'dark']
        """
        # Convert input to numpy array properly
        dataset = self._validate_dataset(dataset)
        n_cols = dataset.shape[1]
        
        couplingsTotal = databank("matrix", [n_cols, n_cols])
        couplingsPosi = databank("matrix", [n_cols, n_cols])
        couplingsNega = databank("matrix", [n_cols, n_cols])
        couplingsDark = databank("matrix", [n_cols, n_cols])

        # Calculate causality for each pair of variables
        for i in range(n_cols):
            for j in range(n_cols):
                if i != j:
                    X_list = dataset.iloc[:, i].values.tolist()
                    Y_list = dataset.iloc[:, j].values.tolist()

                    # Check if enough data points for causality calculation
                    if fcp(E, tau, h, X_list) and fcp(E, tau, h, Y_list):
                        # Calculate pattern causality
                        results = self.pc_lightweight(
                            X_list, Y_list, E, tau, metric, h, weighted, relative
                        )

                        # Store results
                        couplingsTotal[i, j] = results["Total Causality"].values[0]
                        couplingsPosi[i, j] = results["Positive Causality"].values[0]
                        couplingsNega[i, j] = results["Negative Causality"].values[0]
                        couplingsDark[i, j] = results["Dark Causality"].values[0]

        # Calculate mean metrics
        results = pd.DataFrame({
            'E': [E],
            'tau': [tau],
            'total': [np.nanmean(couplingsTotal)],
            'positive': [np.nanmean(couplingsPosi)],
            'negative': [np.nanmean(couplingsNega)],
            'dark': [np.nanmean(couplingsDark)]
        })

        return results

    def optimal_parameters_search(self,
                                Emax: int,
                                tau_max: int,
                                metric: str = "euclidean",
                                h: int = 1,
                                weighted: bool = False,
                                relative: bool = True,
                                dataset: Union[pd.DataFrame, np.ndarray, List] = None,
                                verbose: bool = None) -> pd.DataFrame:
        """
        Search for optimal parameters E and tau for pattern causality analysis.
        
        Args:
            Emax: Maximum embedding dimension (must be > 2)
            tau_max: Maximum time delay
            metric: Distance metric to use
            h: Prediction horizon
            weighted: Whether to use weighted calculations
            relative: Whether to use relative differences (default: False for absolute)
            dataset: Input dataset (DataFrame, numpy array, or list)
            verbose: Override class-level verbose setting
            
        Returns:
            DataFrame containing accuracy results for different parameter combinations
        """
        verbose = self.verbose if verbose is None else verbose
        start_time = time.time()
        
        if dataset is None:
            raise ValueError("Dataset must be provided")
            
        if Emax < 3:
            raise ValueError("Please enter the Emax with the number > 2")

        # Validate dataset
        dataset = self._validate_dataset(dataset)
        
        if verbose:
            self._print_if_verbose(f"\nStarting parameter search:", verbose)
            self._print_if_verbose(f"Dataset shape: {dataset.shape}", verbose)
            self._print_if_verbose(f"E range: 2 to {Emax}", verbose)
            self._print_if_verbose(f"tau range: 1 to {tau_max}", verbose)
        
        E_array = range(2, Emax + 1)
        tau_array = range(1, tau_max + 1)
        total_combinations = len(E_array) * len(tau_array)
        
        if verbose:
            self._print_if_verbose(f"Total parameter combinations to test: {total_combinations}", verbose)

        # Initialize matrices using databank
        tests_total = databank("matrix", [len(E_array), len(tau_array)])
        tests_posi = databank("matrix", [len(E_array), len(tau_array)])
        tests_nega = databank("matrix", [len(E_array), len(tau_array)])
        tests_dark = databank("matrix", [len(E_array), len(tau_array)])

        combinations_tested = 0
        best_score = -np.inf
        best_params = None

        # Main calculation loop
        for i, E in enumerate(E_array):
            for j, tau in enumerate(tau_array):
                combinations_tested += 1
                if verbose:
                    self._print_if_verbose(f"\nTesting combination {combinations_tested}/{total_combinations}", verbose)
                    self._print_if_verbose(f"Parameters: E={E}, tau={tau}", verbose)
                
                temp = self.pc_accuracy(
                    dataset=dataset, 
                    E=E, 
                    tau=tau, 
                    metric=metric, 
                    h=h, 
                    weighted=weighted,
                    relative=relative
                )

                # Store results
                total_score = temp["total"].values[0]
                tests_total[i, j] = total_score
                tests_posi[i, j] = temp["positive"].values[0]
                tests_nega[i, j] = temp["negative"].values[0]
                tests_dark[i, j] = temp["dark"].values[0]
                
                # Track best parameters
                if total_score > best_score:
                    best_score = total_score
                    best_params = (E, tau)
                    
                if verbose:
                    self._print_if_verbose(f"Results:", verbose)
                    self._print_if_verbose(f"  Total: {total_score:.3f}", verbose)
                    self._print_if_verbose(f"  Positive: {temp['positive'].values[0]:.3f}", verbose)
                    self._print_if_verbose(f"  Negative: {temp['negative'].values[0]:.3f}", verbose)
                    self._print_if_verbose(f"  Dark: {temp['dark'].values[0]:.3f}", verbose)

        # Process results
        accuracy_summary = []
        for i, E in enumerate(E_array):
            for j, tau in enumerate(tau_array):
                row_data = {
                    "E": E,
                    "tau": tau,
                    "Total": tests_total[i, j],
                    "of which Positive": tests_posi[i, j],
                    "of which Negative": tests_nega[i, j],
                    "of which Dark": tests_dark[i, j],
                }
                accuracy_summary.append(row_data)

        # Create final DataFrame without custom index
        accuracy_df = pd.DataFrame(accuracy_summary)

        end_time = time.time()
        time_taken = end_time - start_time
        
        if verbose:
            self._print_if_verbose(f"\nParameter search completed in {time_taken:.2f} seconds", verbose)
            self._print_if_verbose(f"Best parameters found: E={best_params[0]}, tau={best_params[1]}", verbose)
            self._print_if_verbose(f"Best total score: {best_score:.3f}", verbose)
            
            # Additional statistics
            self._print_if_verbose("\nParameter Search Statistics:", verbose)
            self._print_if_verbose(f"Mean total score: {np.mean(tests_total):.3f}", verbose)
            self._print_if_verbose(f"Std total score: {np.std(tests_total):.3f}", verbose)
            self._print_if_verbose(f"Score range: [{np.min(tests_total):.3f}, {np.max(tests_total):.3f}]", verbose)

        return accuracy_df

    def pc_full_details(self,
                       X: Union[List, np.ndarray, pd.Series],
                       Y: Union[List, np.ndarray, pd.Series],
                       E: int,
                       tau: int,
                       metric: str = "euclidean",
                       h: int = 1,
                       weighted: bool = False,
                       relative: bool = True,
                       verbose: bool = None) -> pd.DataFrame:
        """
        Pattern Causality Full Details implementation
        
        Args:
            X: Input time series (causal variable)
            Y: Input time series (affected variable)
            E: Embedding dimension
            tau: Time delay
            metric: Distance metric to use
            h: Prediction horizon
            weighted: Whether to use weighted calculations
            relative: Whether to use relative differences (default: False for absolute)
            verbose: Override class-level verbose setting
            
        Returns:
            pd.DataFrame: Detailed causality metrics for each time point
                         Columns: ['No Causality', 'Positive Causality', 
                                 'Negative Causality', 'Dark Causality']
                         Each row represents a time point. For weighted=True,
                         values are erf calculation results. For weighted=False,
                         exactly one column will have value 1 and others 0.
                         Points outside the valid range will be NaN.
        """
        verbose = self.verbose if verbose is None else verbose
        start_time = time.time()
        
        X, Y = self._validate_input(X, Y)
        
        if verbose:
            stats = self._calculate_basic_stats(X, Y)
            self._print_if_verbose(f"\nInput Statistics:", verbose)
            self._print_if_verbose(f"X: mean={stats['X_mean']:.3f}, std={stats['X_std']:.3f}", verbose)
            self._print_if_verbose(f"Y: mean={stats['Y_mean']:.3f}, std={stats['Y_std']:.3f}", verbose)
            self._print_if_verbose(f"Correlation: {stats['correlation']:.3f}", verbose)
            self._print_if_verbose(f"Series length: {stats['X_length']}", verbose)
        
        # Initialize constants
        NNSPAN = E + 1
        CCSPAN = (E - 1) * tau
        hashedpatterns = patternhashing(E)
        
        if hashedpatterns is None or len(hashedpatterns) == 0:
            raise ValueError(f"Failed to generate hash patterns for E={E}")
            
        if verbose:
            self._print_if_verbose(f"\nInitializing computation with E={E}, tau={tau}, h={h}", verbose)
        
        # Calculate shadow attractors
        Mx = statespace(X, E, tau)
        My = statespace(Y, E, tau)
        SMx = signaturespace(Mx, E, relative=relative)
        SMy = signaturespace(My, E, relative=relative)
        PSMx = patternspace(SMx, E)
        PSMy = patternspace(SMy, E)
        Dx = distancematrix(Mx, metric=metric)
        Dy = distancematrix(My, metric=metric)
        
        # Check time series length
        FCP = fcp(E, tau, h, X)
        al_loop_dur = range(FCP - 1, len(X) - (E - 1) * tau - h + 1)
        total_steps = len(al_loop_dur)
        
        if verbose:
            self._print_if_verbose(f"\nProcessing {total_steps} time points...", verbose)
        
        # Initialize causality matrix
        predictedPCMatrix = databank("array", [3 ** (E - 1), 3 ** (E - 1), len(Y)])
        real_loop = None
        processed_points = 0
        valid_points = 0
        
        # Main computation loop
        for i in al_loop_dur:
            processed_points += 1
            if verbose and processed_points % max(1, total_steps // 10) == 0:
                self._print_if_verbose(f"Progress: {processed_points}/{total_steps} points processed ({(processed_points/total_steps)*100:.1f}%)", verbose)
            
            if i + h >= len(My):
                continue
                
            if not np.any(np.isnan(Mx[i, :])) and not np.any(np.isnan(My[i + h, :])):
                NNx = pastNNs(CCSPAN, NNSPAN, Mx, Dx, SMx, PSMx, i, h)
                
                if NNx is not None and not np.any(np.isnan(NNx["dists"])):
                    if not np.any(np.isnan(Dy[i, NNx["times"] + h])):
                        valid_points += 1
                        if real_loop is None:
                            real_loop = i
                        else:
                            real_loop = np.append(real_loop, i)
                            
                        projNNy = projectedNNs(My, Dy, SMy, PSMy, NNx["times"], i, h)
                        predicted_result = predictionY(E=E, projNNy=projNNy, zeroTolerance=E-1)
                        
                        # Get patterns and signatures
                        predictedSignatureY = predicted_result["predictedSignatureY"]
                        predictedPatternY = predicted_result["predictedPatternY"]
                        signatureX = SMx[i, :]
                        patternX = PSMx[i]
                        realSignatureY = SMy[i + h, :]
                        realPatternY = PSMy[i + h]
                        
                        # Calculate PC matrix values
                        pc = fillPCMatrix(
                            weighted=weighted,
                            predictedPatternY=predictedPatternY,
                            realPatternY=realPatternY,
                            predictedSignatureY=predictedSignatureY,
                            realSignatureY=realSignatureY,
                            patternX=patternX,
                            signatureX=signatureX
                        )
                        
                        # Find pattern indices
                        tolerance = 1e-10
                        hashedpatterns = np.array(hashedpatterns, dtype=np.float64)
                        patternX_val = np.float64(patternX.item())
                        predictedPatternY_val = np.float64(predictedPatternY)
                        
                        patternX_matches = np.where(np.abs(hashedpatterns - patternX_val) < tolerance)[0]
                        predictedPatternY_matches = np.where(np.abs(hashedpatterns - predictedPatternY_val) < tolerance)[0]
                        
                        if len(patternX_matches) > 0 and len(predictedPatternY_matches) > 0:
                            patternX_idx = patternX_matches[0]
                            predictedPatternY_idx = predictedPatternY_matches[0]
                            predictedPCMatrix[patternX_idx, predictedPatternY_idx, i] = pc["predicted"]
        
        # Print final progress update
        if verbose:
            self._print_if_verbose(f"Progress: {total_steps}/{total_steps} points processed (100.0%)\n", verbose)
        
        # Calculate causality metrics
        causality = natureOfCausality(predictedPCMatrix, real_loop, hashedpatterns, X, weighted)
        
        # Create DataFrame with NaN values
        result_df = pd.DataFrame(
            np.full((len(X), 4), np.nan),
            columns=['No Causality', 'Positive Causality', 'Negative Causality', 'Dark Causality']
        )
        
        # Calculate valid range
        start_idx = FCP - 1
        end_idx = len(X) - (E - 1) * tau - h
        
        # Fill in causality values for the valid range
        if weighted:
            # For weighted=True, use the actual values from causality
            for i in range(len(X)):
                if i in real_loop:
                    result_df.loc[i, 'No Causality'] = causality['noCausality'][i]
                    result_df.loc[i, 'Positive Causality'] = causality['Positive'][i]
                    result_df.loc[i, 'Negative Causality'] = causality['Negative'][i]
                    result_df.loc[i, 'Dark Causality'] = causality['Dark'][i]
        else:
            # For weighted=False, use binary values (0 or 1)
            for i in range(len(X)):
                if i in real_loop:
                    if causality['noCausality'][i] == 1:
                        result_df.loc[i, 'No Causality'] = 1
                        result_df.loc[i, ['Positive Causality', 'Negative Causality', 'Dark Causality']] = 0
                    elif causality['Positive'][i] == 1:
                        result_df.loc[i, 'Positive Causality'] = 1
                        result_df.loc[i, ['No Causality', 'Negative Causality', 'Dark Causality']] = 0
                    elif causality['Negative'][i] == 1:
                        result_df.loc[i, 'Negative Causality'] = 1
                        result_df.loc[i, ['No Causality', 'Positive Causality', 'Dark Causality']] = 0
                    elif causality['Dark'][i] == 1:
                        result_df.loc[i, 'Dark Causality'] = 1
                        result_df.loc[i, ['No Causality', 'Positive Causality', 'Negative Causality']] = 0
        
        # Add summary row (counting only non-NaN values)
        summary = pd.DataFrame({
            'No Causality': [np.sum(~np.isnan(result_df['No Causality']) & (result_df['No Causality'] > 0))],
            'Positive Causality': [np.sum(~np.isnan(result_df['Positive Causality']) & (result_df['Positive Causality'] > 0))],
            'Negative Causality': [np.sum(~np.isnan(result_df['Negative Causality']) & (result_df['Negative Causality'] > 0))],
            'Dark Causality': [np.sum(~np.isnan(result_df['Dark Causality']) & (result_df['Dark Causality'] > 0))]
        }, index=['Total Points'])
        
        result_df = pd.concat([result_df, summary])
        
        if verbose:
            end_time = time.time()
            self._print_if_verbose(f"\nComputation completed in {end_time - start_time:.2f} seconds", verbose)
            self._print_if_verbose("\nCausality Summary:", verbose)
            self._print_if_verbose(f"Valid range: points {start_idx} to {end_idx}", verbose)
            self._print_if_verbose(f"No Causality Points: {int(summary['No Causality'].values[0])}", verbose)
            self._print_if_verbose(f"Positive Causality Points: {int(summary['Positive Causality'].values[0])}", verbose)
            self._print_if_verbose(f"Negative Causality Points: {int(summary['Negative Causality'].values[0])}", verbose)
            self._print_if_verbose(f"Dark Causality Points: {int(summary['Dark Causality'].values[0])}", verbose)
        
        return result_df

    def pc_cross_validation(self,
                          X: Union[List, np.ndarray, pd.Series],
                          Y: Union[List, np.ndarray, pd.Series],
                          E: int,
                          tau: int,
                          metric: str,
                          h: int,
                          weighted: bool,
                          numberset: Sequence[int],
                          relative: bool = True,
                          verbose: bool = None) -> pd.DataFrame:
        """
        Perform cross validation for pattern causality analysis
        
        Args:
            X: Input time series (causal variable)
            Y: Input time series (affected variable)
            E: Embedding dimension
            tau: Time delay
            metric: Distance metric to use
            h: Prediction horizon
            weighted: Whether to use weighted calculations
            numberset: Sequence of sample sizes to test
            relative: Whether to use relative differences (default: False for absolute)
            verbose: Override class-level verbose setting
            
        Returns:
            DataFrame containing cross validation results
        """
        verbose = self.verbose if verbose is None else verbose
        start_time = time.time()
        
        if not isinstance(numberset, (list, tuple, np.ndarray)):
            raise TypeError("Please enter the vector of the sample number.")
            
        X, Y = self._validate_input(X, Y)
        X = np.array(X)
        Y = np.array(Y)
        
        if max(numberset) > len(X):
            raise ValueError("The sample number is larger than the dataset.")
            
        if verbose:
            stats = self._calculate_basic_stats(X, Y)
            self._print_if_verbose(f"\nCross Validation Setup:", verbose)
            self._print_if_verbose(f"Total data points: {len(X)}", verbose)
            self._print_if_verbose(f"Sample sizes to test: {numberset}", verbose)
            self._print_if_verbose(f"Parameters: E={E}, tau={tau}, h={h}", verbose)
            self._print_if_verbose(f"\nInput Statistics:", verbose)
            self._print_if_verbose(f"X: mean={stats['X_mean']:.3f}, std={stats['X_std']:.3f}", verbose)
            self._print_if_verbose(f"Y: mean={stats['Y_mean']:.3f}, std={stats['Y_std']:.3f}", verbose)
            self._print_if_verbose(f"Correlation: {stats['correlation']:.3f}", verbose)
            
        numbers = np.sort(numberset)
        positive = databank("vector", [len(numberset)])
        negative = databank("vector", [len(numberset)])
        dark = databank("vector", [len(numberset)])
        
        total_samples = len(numbers)
        
        for i, n in enumerate(numbers):
            if verbose:
                self._print_if_verbose(f"\nProcessing sample size {n} ({i+1}/{total_samples})", verbose)
                
            sample_indices = np.random.choice(len(X), size=n, replace=False)
            sample_x = X[sample_indices]
            sample_y = Y[sample_indices]
            
            results = self.pc_lightweight(
                X=sample_x,
                Y=sample_y,
                E=E,
                tau=tau,
                metric=metric,
                h=h,
                weighted=weighted,
                relative=relative,
                verbose=False  # Suppress verbose output for individual calculations
            )
            
            positive[i] = results["Positive Causality"].values[0]
            negative[i] = results["Negative Causality"].values[0]
            dark[i] = results["Dark Causality"].values[0]
            
            if verbose:
                self._print_if_verbose(f"Results for n={n}:", verbose)
                self._print_if_verbose(f"  Positive: {positive[i]:.3f}", verbose)
                self._print_if_verbose(f"  Negative: {negative[i]:.3f}", verbose)
                self._print_if_verbose(f"  Dark: {dark[i]:.3f}", verbose)
        
        results_df = pd.DataFrame({
            "positive": positive,
            "negative": negative,
            "dark": dark
        }, index=numbers)
        
        if verbose:
            end_time = time.time()
            self._print_if_verbose(f"\nCross validation completed in {end_time - start_time:.2f} seconds", verbose)
            self._print_if_verbose("\nSummary Statistics:", verbose)
            self._print_if_verbose("Positive Causality:", verbose)
            self._print_if_verbose(f"  Mean: {np.mean(positive):.3f}", verbose)
            self._print_if_verbose(f"  Std: {np.std(positive):.3f}", verbose)
            self._print_if_verbose("Negative Causality:", verbose)
            self._print_if_verbose(f"  Mean: {np.mean(negative):.3f}", verbose)
            self._print_if_verbose(f"  Std: {np.std(negative):.3f}", verbose)
            self._print_if_verbose("Dark Causality:", verbose)
            self._print_if_verbose(f"  Mean: {np.mean(dark):.3f}", verbose)
            self._print_if_verbose(f"  Std: {np.std(dark):.3f}", verbose)
            
        return results_df

    @staticmethod
    def load_data(file_path: str, sep: str = ",", header: Union[int, None] = 0) -> pd.DataFrame:
        """Load data from a file into a pandas DataFrame.
        
        Args:
            file_path: Path to the data file
            sep: Separator used in the file (default: ",")
            header: Row number to use as column names (default: 0)
                   Use None if there is no header
            
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file cannot be parsed
        """
        try:
            data = pd.read_csv(file_path, sep=sep, header=header)
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {file_path}")
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")

    @staticmethod
    def get_supported_metrics() -> List[str]:
        """Return list of supported distance metrics
        
        Returns:
            List of supported metric names
        """
        return ["euclidean", "manhattan", "chebyshev", "minkowski"]
        
    def get_parameter_ranges(self) -> Dict[str, Tuple[int, int]]:
        """Return recommended parameter ranges
        
        Returns:
            Dictionary with parameter names and their recommended ranges
        """
        return {
            "E": (2, 10),  # Embedding dimension
            "tau": (1, 5), # Time delay
            "h": (1, 3)    # Prediction horizon
        }

