import numpy as np
import pandas as pd
import time
from typing import List, Union, Sequence, Dict, TypeVar, runtime_checkable, Protocol
from utils.statespace import statespace
from utils.patternhashing import patternhashing
from utils.projectedNNs import projectedNNs
from utils.predictionY import predictionY
from utils.fillPCMatrix import fillPCMatrix
from utils.fcp import fcp
from utils.signaturespace import signaturespace
from utils.distancematrix import distancematrix
from utils.patternspace import patternspace
from utils.pastNNs import pastNNs
from utils.databank import databank
from utils.natureOfCausality import natureOfCausality


def pc_lightweight(
    X: Union[List, np.ndarray, pd.Series],
    Y: Union[List, np.ndarray, pd.Series],
    E: int,
    tau: int,
    metric: str = "euclidean",
    h: int = 1,
    weighted: bool = False,
) -> pd.DataFrame:
    """
    Pattern Causality Lightweight implementation

    Args:
        X: Input time series (causal variable) - can be list, numpy array, or pandas Series
        Y: Input time series (affected variable) - can be list, numpy array, or pandas Series
        E: Embedding dimension
        tau: Time delay
        metric: Distance metric to use
        h: Prediction horizon
        weighted: Whether to use weighted calculations

    Returns:
        Dictionary containing causality metrics
    """
    # Convert inputs to lists if they are pandas Series or numpy arrays
    if isinstance(X, pd.Series):
        X = X.values.tolist()
    elif isinstance(X, np.ndarray):
        X = X.tolist()

    if isinstance(Y, pd.Series):
        Y = Y.values.tolist()
    elif isinstance(Y, np.ndarray):
        Y = Y.tolist()

    if not isinstance(X, list) or not isinstance(Y, list):
        raise TypeError("X and Y must be lists, numpy arrays, or pandas Series")

    # Initialize constants
    NNSPAN = E + 1  # Minimum number of nearest neighbors
    CCSPAN = (E - 1) * tau  # Remove common coordinate NNs
    hashedpatterns = patternhashing(E)  # Generate hash patterns
    
    # Convert hashedpatterns to numpy array if it's not already
    if not isinstance(hashedpatterns, np.ndarray):
        hashedpatterns = np.array(hashedpatterns)

    if hashedpatterns is None or len(hashedpatterns) == 0:
        raise ValueError(f"Failed to generate hash patterns for E={E}")

    #####################################
    ### STEP 1: THE SHADOW ATTRACTORS ###
    #####################################

    # A: State Space
    Mx = statespace(X, E, tau)
    My = statespace(Y, E, tau)

    # B: Signature Space
    SMx = signaturespace(Mx, E)
    SMy = signaturespace(My, E)

    # C: Pattern Space
    PSMx = patternspace(SMx, E)
    PSMy = patternspace(SMy, E)

    # D: Distance Matrix
    Dx = distancematrix(Mx, metric=metric)
    Dy = distancematrix(My, metric=metric)

    # Check if time series length is sufficient
    FCP = fcp(E, tau, h, X)

    # Calculate main loop duration
    # Note: We need to ensure we don't exceed array bounds when checking My[i + h]
    # Changed to include all valid points by adjusting the FCP offset
    al_loop_dur = range(FCP - 1, len(X) - (E - 1) * tau - h + 1)

    # Initialize causality matrix
    # Important: Keep the full length of Y for the matrix
    predictedPCMatrix = databank("array", [3 ** (E - 1), 3 ** (E - 1), len(Y)])

    real_loop = None

    # Process the main loop points
    for i in al_loop_dur:
        # Add bounds check to prevent index out of bounds
        if i + h >= len(My):
            continue
            
        # Check if current point and future point are valid
        if not np.any(np.isnan(Mx[i, :])) and not np.any(np.isnan(My[i + h, :])):
            # Get nearest neighbors
            NNx = pastNNs(CCSPAN, NNSPAN, Mx, Dx, SMx, PSMx, i, h)
            
            # Check if we have valid nearest neighbors
            if NNx is not None and not np.any(np.isnan(NNx["dists"])):
                # Check if future points of nearest neighbors are valid
                if not np.any(np.isnan(Dy[i, NNx["times"] + h])):
                    # Add to real_loop
                    if real_loop is None:
                        real_loop = i
                    else:
                        real_loop = np.append(real_loop, i)
                    projNNy = projectedNNs(My, Dy, SMy, PSMy, NNx["times"], i, h)
                    predicted_result = predictionY(
                        E=E, projNNy=projNNy, zeroTolerance=E - 1
                    )
                    predictedSignatureY = predicted_result["predictedSignatureY"]
                    predictedPatternY = predicted_result["predictedPatternY"]
                    signatureX = SMx[i, :]
                    patternX = PSMx[i]
                    realSignatureY = SMy[i + h, :]
                    realPatternY = PSMy[i + h]
                    pc = fillPCMatrix(
                        weighted=weighted,
                        predictedPatternY=predictedPatternY,
                        realPatternY=realPatternY,
                        predictedSignatureY=predictedSignatureY,
                        realSignatureY=realSignatureY,
                        patternX=patternX,
                        signatureX=signatureX,
                    )
                    
                    # Find indices using numpy where with tolerance
                    tolerance = 1e-10  # Use a smaller tolerance for floating point comparisons
                    
                    # Convert patterns to float64 for consistent comparison
                    hashedpatterns = np.array(hashedpatterns, dtype=np.float64)
                    patternX_val = np.float64(patternX.item())
                    predictedPatternY_val = np.float64(predictedPatternY)
                    
                    # Find matches using absolute tolerance
                    patternX_matches = np.where(np.abs(hashedpatterns - patternX_val) < tolerance)[0]
                    predictedPatternY_matches = np.where(np.abs(hashedpatterns - predictedPatternY_val) < tolerance)[0]
                    
                    if len(patternX_matches) == 0:
                        # Try to find closest match
                        closest_idx = np.argmin(np.abs(hashedpatterns - patternX_val))
                        closest_val = hashedpatterns[closest_idx]
                        raise ValueError(
                            f"Pattern X value {patternX_val} not found in hashedpatterns.\n"
                            f"This suggests a mismatch in pattern generation between R and C++."
                        )
                    if len(predictedPatternY_matches) == 0:
                        # Try to find closest match
                        closest_idx = np.argmin(np.abs(hashedpatterns - predictedPatternY_val))
                        closest_val = hashedpatterns[closest_idx]
                        raise ValueError(
                            f"Pattern Y value {predictedPatternY_val} not found in hashedpatterns.\n"
                            f"This suggests a mismatch in pattern generation between R and C++."
                        )
                    
                    patternX_idx = patternX_matches[0]
                    predictedPatternY_idx = predictedPatternY_matches[0]
                    predictedPCMatrix[patternX_idx, predictedPatternY_idx, i] = pc["predicted"]

    # Calculate causality metrics
    causality = natureOfCausality(
        predictedPCMatrix, real_loop, hashedpatterns, X, weighted
    )

    # Debug information
    print("\nDebug Information:")
    print("1. Raw causality values:")
    print("No Causality:", causality["noCausality"])
    print("Positive:", causality["Positive"])
    print("Negative:", causality["Negative"])
    print("Dark:", causality["Dark"])
    print("\n2. Real loop indices:", real_loop)

    # Calculate percentages, handling NA values
    totalCausPercent = 1 - np.nanmean(causality["noCausality"])
    print("\n3. Total Causality calculation:")
    print("Mean of noCausality:", np.nanmean(causality["noCausality"]))
    print("Total Causality Percent:", totalCausPercent)

    # For the other metrics, only consider cases where noCausality != 1
    mask = causality["noCausality"][real_loop] != 1
    print("\n4. Mask information:")
    print("Number of valid cases:", np.sum(mask))
    
    if np.any(mask):  # Only calculate if we have valid cases
        valid_indices = real_loop[mask]
        print("\n5. Valid indices:", valid_indices)
        
        # Calculate percentages only for valid cases
        valid_pos = causality["Positive"][valid_indices]
        valid_neg = causality["Negative"][valid_indices]
        valid_dark = causality["Dark"][valid_indices]
        
        print("\n6. Valid values before NaN removal:")
        print("Positive:", valid_pos)
        print("Negative:", valid_neg)
        print("Dark:", valid_dark)
        
        # Remove NaN values before calculating mean
        valid_pos = valid_pos[~np.isnan(valid_pos)]
        valid_neg = valid_neg[~np.isnan(valid_neg)]
        valid_dark = valid_dark[~np.isnan(valid_dark)]
        
        print("\n7. Valid values after NaN removal:")
        print("Positive:", valid_pos)
        print("Negative:", valid_neg)
        print("Dark:", valid_dark)
        
        # Calculate means
        posiCausPercent = np.mean(valid_pos) if len(valid_pos) > 0 else 0.0
        negaCausPercent = np.mean(valid_neg) if len(valid_neg) > 0 else 0.0
        darkCausPercent = np.mean(valid_dark) if len(valid_dark) > 0 else 0.0
        
        print("\n8. Initial percentages:")
        print("Positive:", posiCausPercent)
        print("Negative:", negaCausPercent)
        print("Dark:", darkCausPercent)

        if weighted:
            total = posiCausPercent + negaCausPercent + darkCausPercent
            if total > 0:
                posiCausPercent = posiCausPercent / total
                negaCausPercent = negaCausPercent / total
                darkCausPercent = darkCausPercent / total
                print("\n9. Weighted percentages:")
                print("Positive:", posiCausPercent)
                print("Negative:", negaCausPercent)
                print("Dark:", darkCausPercent)
    else:  # If no valid cases, set all to 0
        posiCausPercent = 0.0
        negaCausPercent = 0.0
        darkCausPercent = 0.0
        print("\n5. No valid cases found, setting all percentages to 0")

    # Create a DataFrame with the causality results
    results_df = pd.DataFrame(
        {
            "Total Causality": [totalCausPercent],
            "Positive Causality": [posiCausPercent],
            "Negative Causality": [negaCausPercent],
            "Dark Causality": [darkCausPercent],
        }
    )

    return results_df


def pc_accuracy(dataset, E, tau, metric, h, weighted):
    """
    Calculate pattern causality accuracy metrics for a dataset.

    Args:
        dataset: list of time series, numpy array, or pandas DataFrame
        E: embedding dimension
        tau: time delay
        metric: distance metric to use
        h: prediction horizon
        weighted: whether to use weighted calculations

    Returns:
        DataFrame with causality metrics
    """
    # Convert input to numpy array properly
    if isinstance(dataset, pd.DataFrame):
        dataset = dataset.values  # Convert DataFrame to numpy array
    elif isinstance(dataset, list):
        dataset = np.array(dataset).T  # Transpose to get correct shape
    elif not isinstance(dataset, np.ndarray):
        raise TypeError("dataset must be a list, numpy array, or pandas DataFrame")

    # Initialize storage matrices
    n_cols = dataset.shape[1]
    couplingsTotal = databank("matrix", [n_cols, n_cols])
    couplingsPosi = databank("matrix", [n_cols, n_cols])
    couplingsNega = databank("matrix", [n_cols, n_cols])
    couplingsDark = databank("matrix", [n_cols, n_cols])

    # Calculate causality for each pair of variables
    for i in range(n_cols):
        for j in range(n_cols):
            if i != j:
                # Convert numpy arrays back to lists before passing to functions
                X_list = dataset[:, i].tolist()
                Y_list = dataset[:, j].tolist()

                # Check if enough data points for causality calculation
                if fcp(E, tau, h, X_list) and fcp(E, tau, h, Y_list):
                    # Calculate pattern causality
                    results = pc_lightweight(
                        X_list, Y_list, E, tau, metric, h, weighted
                    )

                    # Store results
                    couplingsTotal[i, j] = results["Total Causality"].values[0]
                    couplingsPosi[i, j] = results["Positive Causality"].values[0]
                    couplingsNega[i, j] = results["Negative Causality"].values[0]
                    couplingsDark[i, j] = results["Dark Causality"].values[0]

    # Calculate mean metrics
    results = pd.DataFrame(
        {
            "E": [E],
            "tau": [tau],
            "total": [np.nanmean(couplingsTotal)],
            "positive": [np.nanmean(couplingsPosi)],
            "negative": [np.nanmean(couplingsNega)],
            "dark": [np.nanmean(couplingsDark)],
        }
    )

    return results


def optimal_parameters_search(
    Emax: int, tau_max: int, metric: str, dataset: Union[pd.DataFrame, np.ndarray, List]
) -> pd.DataFrame:
    """
    Search for optimal parameters E and tau for pattern causality analysis.

    Args:
        Emax: Maximum embedding dimension (must be > 2)
        tau_max: Maximum time delay
        metric: Distance metric to use
        dataset: Input dataset (DataFrame, numpy array, or list)

    Returns:
        DataFrame containing accuracy results for different parameter combinations
    """
    if Emax < 3:
        raise ValueError("Please enter the Emax with the number > 2")

    E_array = range(2, Emax + 1)
    tau_array = range(1, tau_max + 1)

    # Initialize matrices using databank
    tests_total = databank("matrix", [len(E_array), len(tau_array)])
    tests_posi = databank("matrix", [len(E_array), len(tau_array)])
    tests_nega = databank("matrix", [len(E_array), len(tau_array)])
    tests_dark = databank("matrix", [len(E_array), len(tau_array)])

    start_time = time.time()

    # Main calculation loop
    for i, E in enumerate(E_array):
        print(f"Testing | E: {E}")
        for j, tau in enumerate(tau_array):
            print(f"Testing | tau: {tau}")
            temp = pc_accuracy(
                dataset=dataset, E=E, tau=tau, metric=metric, h=0, weighted=False
            )

            # Store results
            tests_total[i, j] = temp["total"].values[0]
            tests_posi[i, j] = temp["positive"].values[0]
            tests_nega[i, j] = temp["negative"].values[0]
            tests_dark[i, j] = temp["dark"].values[0]

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

    # Create final DataFrame
    accuracy_df = pd.DataFrame(accuracy_summary)

    # Set index
    accuracy_df.index = [
        f"E = {row['E']} tau = {row['tau']}" for _, row in accuracy_df.iterrows()
    ]

    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Calculation duration: {time_taken:.2f} seconds")

    return accuracy_df


def zero_counter(vec: np.ndarray) -> int:
    """Count number of zeros in vector"""
    return np.sum(vec == 0)


def zero_filtering(vec: np.ndarray, threshold: int) -> bool:
    """Check if number of zeros is below threshold"""
    return zero_counter(vec) < threshold


def na_counter(vec: np.ndarray) -> int:
    """Count number of NaN values in vector"""
    return np.sum(np.isnan(vec))


def na_filtering(vec: np.ndarray, threshold: int) -> bool:
    """Check if number of NaN values is below threshold"""
    return na_counter(vec) < threshold


def pc_cross_validation(
    X: Union[List, np.ndarray, pd.Series],
    Y: Union[List, np.ndarray, pd.Series],
    E: int,
    tau: int,
    metric: str,
    h: int,
    weighted: bool,
    numberset: Sequence[int],
) -> pd.DataFrame:
    """
    Perform cross validation for pattern causality analysis.

    Args:
        X: Input time series (causal variable)
        Y: Input time series (affected variable)
        E: Embedding dimension
        tau: Time delay
        metric: Distance metric to use
        h: Prediction horizon
        weighted: Whether to use weighted calculations
        numberset: Sequence of sample sizes to test

    Returns:
        DataFrame containing cross validation results for different sample sizes
    """
    # Input validation
    if not isinstance(numberset, (list, tuple, np.ndarray)):
        raise TypeError("Please enter the vector of the sample number.")

    # Convert inputs to numpy arrays if needed
    if isinstance(X, (list, pd.Series)):
        X = np.array(X)
    if isinstance(Y, (list, pd.Series)):
        Y = np.array(Y)

    # Check sample sizes
    if max(numberset) > len(X):
        raise ValueError("The sample number is larger than the dataset.")

    # Sort sample sizes
    numbers = np.sort(numberset)

    # Initialize result vectors using databank
    positive = databank("vector", [len(numberset)])
    negative = databank("vector", [len(numberset)])
    dark = databank("vector", [len(numberset)])

    # Main calculation loop
    for i, n in enumerate(numbers):
        # Generate random samples
        sample_indices = np.random.choice(len(X), size=n, replace=False)
        sample_x = X[sample_indices]
        sample_y = Y[sample_indices]

        # Calculate pattern causality
        results = pc_lightweight(
            X=sample_x, Y=sample_y, E=E, tau=tau, metric=metric, h=h, weighted=weighted
        )

        # Store results
        positive[i] = results["Positive Causality"].values[0]
        negative[i] = results["Negative Causality"].values[0]
        dark[i] = results["Dark Causality"].values[0]

    # Create results DataFrame
    results_df = pd.DataFrame(
        {"positive": positive, "negative": negative, "dark": dark}, index=numbers
    )

    return results_df


def pc_matrix(
    dataset: Union[pd.DataFrame, np.ndarray, List],
    E: int,
    tau: int,
    metric: str,
    h: int,
    weighted: bool,
) -> Dict:
    """
    Calculate pattern causality matrix for multivariate time series.

    Args:
        dataset: Input dataset (DataFrame, numpy array, or list)
        E: Embedding dimension
        tau: Time delay
        metric: Distance metric to use
        h: Prediction horizon
        weighted: Whether to use weighted calculations

    Returns:
        Dictionary containing causality matrices and column names
    """
    # Convert input to DataFrame if needed
    if isinstance(dataset, np.ndarray):
        dataset = pd.DataFrame(dataset)
    elif isinstance(dataset, list):
        dataset = pd.DataFrame(np.array(dataset).T)
    elif not isinstance(dataset, pd.DataFrame):
        raise TypeError("dataset must be a DataFrame, numpy array, or list")

    n_cols = dataset.shape[1]

    # Initialize storage arrays using databank
    couplings_posi = databank("array", [n_cols, n_cols])
    couplings_nega = databank("array", [n_cols, n_cols])
    couplings_dark = databank("array", [n_cols, n_cols])

    start_time = time.time()

    # Main calculation loop
    for i in range(n_cols):
        print(f"CAUSE: {dataset.columns[i] if dataset.columns is not None else i}")

        # Get the i-th column as a list
        X = dataset.iloc[:, i].values.tolist()

        for j in range(n_cols):
            print(f"EFFECT: {dataset.columns[j] if dataset.columns is not None else j}")

            if i != j:
                # Check if enough data points for causality calculation
                if fcp(E, tau, h, X):
                    Y = dataset.iloc[:, j].values.tolist()
                    if fcp(E, tau, h, Y):
                        # Calculate pattern causality
                        temp = pc_lightweight(
                            X=X,
                            Y=Y,
                            E=E,
                            tau=tau,
                            metric=metric,
                            h=h,
                            weighted=weighted,
                        )

                        # Store results
                        couplings_posi[i, j] = temp["Positive Causality"].values[0]
                        couplings_nega[i, j] = temp["Negative Causality"].values[0]
                        couplings_dark[i, j] = temp["Dark Causality"].values[0]

    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Calculation duration: {time_taken:.2f} seconds")

    return {
        "positive": couplings_posi,
        "negative": couplings_nega,
        "dark": couplings_dark,
        "items": (
            dataset.columns.tolist()
            if dataset.columns is not None
            else list(range(n_cols))
        ),
    }

@runtime_checkable
class PCMatrixResult(Protocol):
    """Protocol for pc_matrix results"""

    positive: np.ndarray
    negative: np.ndarray
    dark: np.ndarray
    items: list


def validate_pc_matrix_result(result: Dict) -> bool:
    """Validate that the input is a result from pc_matrix"""
    required_keys = {"positive", "negative", "dark", "items"}

    # Check if all required keys exist
    if not all(key in result for key in required_keys):
        return False

    # Check if matrices have correct shape and type
    matrices = [result["positive"], result["negative"], result["dark"]]
    if not all(isinstance(m, np.ndarray) for m in matrices):
        return False

    # Check if all matrices are square and have same dimensions
    shapes = [m.shape for m in matrices]
    if not all(len(shape) == 2 and shape[0] == shape[1] for shape in shapes):
        return False

    # Check if items list length matches matrix dimensions
    if (
        not isinstance(result["items"], list)
        or len(result["items"]) != matrices[0].shape[0]
    ):
        return False

    return True


def pc_effect(pcmatrix: Dict[str, Union[np.ndarray, list]]) -> Dict[str, pd.DataFrame]:
    """
    Calculate effect metrics from pattern causality matrices.

    Args:
        pcmatrix: Dictionary containing causality matrices and items list
                 (must be output from pc_matrix function)

    Returns:
        Dictionary containing DataFrames for positive, negative, and dark effects

    Raises:
        ValueError: If input is not a valid pc_matrix result
    """
    # Validate input
    if not validate_pc_matrix_result(pcmatrix):
        raise ValueError(
            "Invalid input: must be output from pc_matrix function with keys "
            "'positive', 'negative', 'dark', and 'items', where matrices are square "
            "numpy arrays of the same size and items is a list of matching length"
        )
    # Extract matrices and convert to numpy arrays if needed
    matrices = {}
    for key in ["positive", "negative", "dark"]:
        if isinstance(pcmatrix[key], list):
            matrices[key] = np.array(pcmatrix[key])
        else:
            matrices[key] = pcmatrix[key].copy()

        # Replace NaN with 0 and multiply by 100
        matrices[key] = np.nan_to_num(matrices[key]) * 100

    # Get variable names
    items = pcmatrix["items"]

    # Initialize result dictionary
    results = {}

    # Calculate metrics for each type
    for key, data in matrices.items():
        # Calculate sums
        received = np.sum(data, axis=0)  # sum along columns
        exerted = np.sum(data, axis=1)  # sum along rows
        diff = received - exerted

        # Create DataFrame
        results[key] = pd.DataFrame(
            {"received": received, "exerted": exerted, "Diff": diff}, index=items
        )

    # Add items list to results
    results["items"] = items

    return results


def convert_signature_to_value(E: int, tau: int, Y: Union[List, np.ndarray], i: int, h: int, predicted_signature_Y: np.ndarray) -> np.ndarray:
    """
    Convert signature to actual values.
    
    Args:
        E: Embedding dimension
        tau: Time delay
        Y: Time series data
        i: Current index
        h: Prediction horizon
        predicted_signature_Y: Predicted signature
        
    Returns:
        Array of predicted values
    """
    predicted_Y = np.zeros(E)
    predicted_Y[0] = Y[i + h]
    for k in range(1, E):
        predicted_Y[k] = predicted_Y[k-1] + predicted_signature_Y[k-1]
    return predicted_Y

def convert_signature_to_value_out_of_sample(E: int, tau: int, Y_pred_last: float, i: int, h: int, predicted_signature_Y: np.ndarray) -> np.ndarray:
    """
    Convert signature to values for out of sample prediction.
    
    Args:
        E: Embedding dimension
        tau: Time delay
        Y_pred_last: Last predicted value
        i: Current index
        h: Prediction horizon
        predicted_signature_Y: Predicted signature
        
    Returns:
        Array of predicted values
    """
    predicted_Y = np.zeros(E)
    predicted_Y[0] = Y_pred_last
    for k in range(1, E):
        predicted_Y[k] = predicted_Y[k-1] + predicted_signature_Y[k-1]
    return predicted_Y

def pc_full_details(
    X: Union[List, np.ndarray, pd.Series],
    Y: Union[List, np.ndarray, pd.Series],
    E: int,
    tau: int,
    metric: str = "euclidean",
    h: int = 1,
    weighted: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Pattern Causality Full Details implementation.
    
    Args:
        X: Input time series (causal variable)
        Y: Input time series (affected variable)
        E: Embedding dimension
        tau: Time delay
        metric: Distance metric to use
        h: Prediction horizon
        weighted: Whether to use weighted calculations
        
    Returns:
        Dictionary containing causality arrays:
            - noCausality: Array of no causality values
            - Positive: Array of positive causality values
            - Negative: Array of negative causality values
            - Dark: Array of dark causality values
    """
    # Convert inputs to lists
    if isinstance(X, pd.Series):
        X = X.values.tolist()
    elif isinstance(X, np.ndarray):
        X = X.tolist()

    if isinstance(Y, pd.Series):
        Y = Y.values.tolist()
    elif isinstance(Y, np.ndarray):
        Y = Y.tolist()

    if not isinstance(X, list) or not isinstance(Y, list):
        raise TypeError("X and Y must be lists, numpy arrays, or pandas Series")

    # Initialize constants
    NNSPAN = E + 1  # Minimum number of nearest neighbors
    CCSPAN = (E - 1) * tau  # Remove common coordinate NNs
    hashedpatterns = patternhashing(E)

    # STEP 1: THE SHADOW ATTRACTORS
    # A: State Space
    Mx = statespace(X, E, tau)
    My = statespace(Y, E, tau)

    # B: Signature Space
    SMx = signaturespace(Mx, E)
    SMy = signaturespace(My, E)

    # C: Pattern Space
    PSMx = patternspace(SMx, E)
    PSMy = patternspace(SMy, E)

    # D: Distance Matrix
    Dx = distancematrix(Mx, metric=metric)
    Dy = distancematrix(My, metric=metric)

    # Check if time series length is sufficient
    FCP = fcp(E, tau, h, X)
    
    # Calculate main loop duration
    al_loop_dur = range(FCP, len(X) - (E - 1) * tau - h)

    # Initialize storage arrays
    predictedPCMatrix = databank("array", [3 ** (E - 1), 3 ** (E - 1), len(Y)])
    real_loop = []

    # Main computation loop
    for i in al_loop_dur:
        if not np.any(np.isnan(Mx[i])) and not np.any(np.isnan(My[i + h])):
            NNx = pastNNs(CCSPAN, NNSPAN, Mx, Dx, SMx, PSMx, i, h)
            
            if NNx and not np.any(np.isnan(NNx["dists"])):
                if not np.any(np.isnan(Dy[i, NNx["times"] + h])):
                    real_loop.append(i)
                    
                    projNNy = projectedNNs(My, Dy, SMy, PSMy, NNx["times"], i, h)

                    # Calculate predictions
                    predicted_result = predictionY(E, projNNy, zeroTolerance=E-1)
                    predictedSignatureY = predicted_result["predictedSignatureY"]
                    predictedPatternY = predicted_result["predictedPatternY"]
                    
                    # Get patterns and signatures
                    signatureX = SMx[i]
                    patternX = PSMx[i]
                    realSignatureY = SMy[i + h]
                    realPatternY = PSMy[i + h]

                    # Calculate PC matrix values
                    pc = fillPCMatrix(
                        weighted=weighted,
                        predictedPatternY=predictedPatternY,
                        realPatternY=realPatternY,
                        predictedSignatureY=predictedSignatureY,
                        realSignatureY=realSignatureY,
                        patternX=patternX,
                        signatureX=signatureX,
                    )
                    
                    # Store PC matrix values
                    predictedPCMatrix[
                        hashedpatterns.index(patternX),
                        hashedpatterns.index(predictedPatternY),
                        i
                    ] = pc["predicted"]

    # Calculate causality spectrum
    real_loop = np.array(real_loop)
    return natureOfCausality(predictedPCMatrix, real_loop, hashedpatterns, X, weighted)

def pc_into_the_dark(
    X: Union[List, np.ndarray, pd.Series],
    Y: Union[List, np.ndarray, pd.Series],
    E: int,
    tau: int,
    metric: str,
    spot: int,
    dark_horizon: int
) -> np.ndarray:
    """
    Pattern Causality prediction into the dark (future).
    
    Args:
        X: Input time series (causal variable)
        Y: Input time series (affected variable)
        E: Embedding dimension
        tau: Time delay
        metric: Distance metric
        spot: Starting point for prediction
        dark_horizon: Number of steps to predict into the future
        
    Returns:
        Array of predicted values
    """
    # Convert inputs to lists
    if isinstance(X, pd.Series):
        X = X.values.tolist()
    elif isinstance(X, np.ndarray):
        X = X.tolist()

    if isinstance(Y, pd.Series):
        Y = Y.values.tolist()
    elif isinstance(Y, np.ndarray):
        Y = Y.tolist()

    if not isinstance(X, list) or not isinstance(Y, list):
        raise TypeError("X and Y must be lists, numpy arrays, or pandas Series")

    # Initialize constants
    NNSPAN = E + 1
    CCSPAN = (E - 1) * tau
    hashedpatterns = patternhashing(E)

    # Calculate shadow attractors
    Mx = statespace(X, E, tau)
    My = statespace(Y, E, tau)
    SMx = signaturespace(Mx, E)
    SMy = signaturespace(My, E)
    PSMx = patternspace(SMx, E)
    PSMy = patternspace(SMy, E)
    Dx = distancematrix(Mx, metric=metric)
    Dy = distancematrix(My, metric=metric)

    # Initialize predicted values storage
    predictedValuesY = databank("matrix", [dark_horizon + 1, E])

    # First prediction to disentangle from Y
    if not np.any(np.isnan(Mx[spot])):
        NNx = pastNNs(CCSPAN, NNSPAN, Mx, Dx, SMx, PSMx, spot, h=0)
        if not np.any(np.isnan(Dy[spot, NNx["times"]])):
            projNNy = projectedNNs(My, Dy, SMy, PSMy, NNx["times"], spot, h=0)
            predictedSignatureY = predictionY(E, projNNy, zeroTolerance=E-1)["predictedSignatureY"]
            predictedValuesY[0] = convert_signature_to_value(E, tau, Y, spot, 0, predictedSignatureY)

    # Predict into the dark
    j = 1
    for h in range(1, dark_horizon + 1):
        if not np.any(np.isnan(Mx[spot])):
            NNx = pastNNs(CCSPAN, NNSPAN, Mx, Dx, SMx, PSMx, spot, h)
            if NNx and not np.any(np.isnan(NNx["dists"])):
                if not np.any(np.isnan(Dy[spot, NNx["times"] + h])):
                    projNNy = projectedNNs(My, Dy, SMy, PSMy, NNx["times"], spot, h)
                    predictedSignatureY = predictionY(E, projNNy, zeroTolerance=E-1)["predictedSignatureY"]
                    predictedValuesY[j] = convert_signature_to_value_out_of_sample(
                        E, tau, predictedValuesY[j-1, -1], spot, h, predictedSignatureY
                    )
                    j += 1

    return predictedValuesY

