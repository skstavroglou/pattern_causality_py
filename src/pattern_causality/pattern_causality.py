import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler

def artifact_formation(dataset, X_set=None, embedding_dimensions=None, time_delays=None):
    # If no names given, consider all columns except the 'Date' column
    if X_set is None:
        X_set = dataset.columns[:]

    # If no Es are given then consider the minimum viable
    if embedding_dimensions is None:
        embedding_dimensions = [2] * len(X_set)

    # If not taus are given then consider the minimum viable
    if time_delays is None:
        time_delays = [1] * len(X_set)

    # Check that lengths of names, embedding_dimensions and time_delays match
    if len(X_set) != len(embedding_dimensions) or len(X_set) != len(time_delays):
        raise ValueError("Length of names, embedding_dimensions, and time_delays must be the same!")

    artifact_df = pd.DataFrame()
    artifact_df.index = dataset.index
    # artifact_df.index.name = 'Date'

    for i, name in enumerate(X_set):
        m = embedding_dimensions[i]
        tau = time_delays[i]

        for j in range(m):
            col_name = f"{name}_e{j + 1}"
            artifact_df[col_name] = dataset[name].shift(j * tau)

    return artifact_df


def artifact_distance_matrix(artifact_df, metric, **kwargs):
    dist_matrix = None
    if metric == 'euclidean':
        dist_matrix = distance.cdist(artifact_df, artifact_df, 'euclidean')
    elif metric == 'minkowski':
        dist_matrix = distance.cdist(artifact_df, artifact_df, 'minkowski', **kwargs)
    elif metric == 'cityblock':
        dist_matrix = distance.cdist(artifact_df, artifact_df, 'cityblock')
    elif metric == 'seuclidean':
        artifact_adjusted_df = artifact_df.copy()
        artifact_adjusted_df = artifact_adjusted_df.replace(np.nan, 0)
        dist_matrix = distance.cdist(artifact_df, artifact_df, 'seuclidean',
                                     V=np.var(np.vstack((artifact_adjusted_df, artifact_adjusted_df)), axis=0))
    elif metric == 'sqeuclidean':
        dist_matrix = distance.cdist(artifact_df, artifact_df, 'sqeuclidean')
    elif metric == 'cosine':
        dist_matrix = distance.cdist(artifact_df, artifact_df, 'cosine')
    elif metric == 'correlation':
        dist_matrix = distance.cdist(artifact_df, artifact_df, 'correlation')
    elif metric == 'chebyshev':
        dist_matrix = distance.cdist(artifact_df, artifact_df, 'chebyshev')
    elif metric == 'canberra':
        dist_matrix = distance.cdist(artifact_df, artifact_df, 'canberra')
    elif metric == 'braycurtis':
        dist_matrix = distance.cdist(artifact_df, artifact_df, 'braycurtis')

    dist_matrix = pd.DataFrame(dist_matrix, index=artifact_df.index, columns=artifact_df.index)
    return dist_matrix


def replace_diagonals_with_nan_and_infinitify(distance_matrix, num_of_diagonals):
    """
    Replace elements of diagonals parallel to the main diagonal with NaN.
    It also counts the main diagonal, i.e. num_of_diagonals=1 will nan-ify only the main diagonal

    Parameters:
    - df: The input DataFrame.
    - num_of_diagonals: Number of diagonals to replace, starting from main diagonal and moving towards bottom-left.

    Returns:
    - DataFrame with replaced diagonals.
    """

    # Convert the DataFrame to numpy array
    arr = distance_matrix.values

    np.fill_diagonal(arr, np.nan)
    arr[np.triu_indices(arr.shape[0], 1)] = np.nan

    # Replace the diagonal elements
    for i in range(num_of_diagonals):
        np.fill_diagonal(arr[i:], np.nan)

    # Convert back to DataFrame
    df_out = pd.DataFrame(arr, index=distance_matrix.index, columns=distance_matrix.columns)

    df_out.fillna(np.inf, inplace=True)

    return df_out


def find_n_nearest_neighbors(distance_matrix, n):
    # Sort them
    nearest_neighbors_dists_df = distance_matrix.apply(lambda row: pd.Series(np.sort(row)), axis=1)
    nearest_neighbors_indices_df = distance_matrix.apply(lambda row: pd.Series(np.argsort(row)), axis=1)
    nearest_neighbors_labels_df = distance_matrix.apply(lambda row: pd.Series(distance_matrix.columns[np.argsort(row)]),
                                                        axis=1)

    # Nan-ify
    nearest_neighbors_dists_df.replace(np.inf, np.nan, inplace=True)
    nearest_neighbors_indices_df = pd.DataFrame(
        np.where(nearest_neighbors_dists_df.isna(), np.nan, nearest_neighbors_indices_df))
    # Convert datetime to string in nearest_neighbors_labels_df
    nearest_neighbors_labels_df = nearest_neighbors_labels_df.applymap(str)
    # Replace inf with nan in nearest_neighbors_dists_df
    nearest_neighbors_dists_df.replace(np.inf, np.nan, inplace=True)
    nearest_neighbors_labels_df = pd.DataFrame(
        np.where(nearest_neighbors_dists_df.isna(), np.nan, nearest_neighbors_labels_df))

    # Keep the n columns
    nearest_neighbors_dists_df = nearest_neighbors_dists_df.iloc[:, :n]
    nearest_neighbors_indices_df = nearest_neighbors_indices_df.iloc[:, :n]
    nearest_neighbors_labels_df = nearest_neighbors_labels_df.iloc[:, :n]

    return nearest_neighbors_dists_df, nearest_neighbors_indices_df, nearest_neighbors_labels_df


def get_nn_features(indices_df, values_df, horizon):
    indices_df_projected = indices_df + horizon

    # Create an empty DataFrame to hold the result
    result_df = pd.DataFrame(index=indices_df_projected.index)

    # Iterate through each column in values_df
    for value_column in values_df.columns:
        # Iterate through each column in indices_df_projected
        count = 1
        for index_column in indices_df_projected.columns:
            # Create a new column name by combining the value_column and index_column names
            new_column_name = f'{value_column}-NN-{count}'
            count = count + 1

            # Create an empty list to hold the result values for this column
            result_values = []

            # Iterate through each row in indices_df_projected
            for index_value in indices_df_projected[index_column]:
                # Check if the index_value is NaN
                if pd.isna(index_value):
                    # If it's NaN, append NaN to result_values
                    result_values.append(np.nan)
                else:
                    # If it's not NaN, convert index_value to an integer,
                    # and use it to index into values_df
                    result_values.append(values_df[value_column].iloc[int(index_value)])

            # Convert result_values to a Series and assign it to the result DataFrame
            result_df[new_column_name] = pd.Series(result_values, index=indices_df_projected.index)

    # Now result_df should be structured as desired

    return result_df


def pattern_causality(dataset, embedding_dimensions, time_delays, metric, horizon, nn_amount, Y_set, X_set=None):

    artifact = artifact_formation(dataset, X_set, embedding_dimensions, time_delays)
    distance_matrix = artifact_distance_matrix(artifact, metric=metric)
    distance_matrix_calibrated = replace_diagonals_with_nan_and_infinitify(distance_matrix,
                                                                           num_of_diagonals=horizon + nn_amount)

    nn_dists_state, nn_indices_state, nn_labels_state = find_n_nearest_neighbors(distance_matrix_calibrated, nn_amount)
    NNs = get_nn_features(nn_indices_state, dataset[Y_set], horizon)

    return NNs, nn_dists_state, distance_matrix_calibrated


def weights_from_distances(distances):
    # Applying exp(-x) to every element
    new_df = np.exp(-distances)

    # Calculate row sums
    row_sums = new_df.sum(axis=1)

    # Divide column-wise
    normalized_df = new_df.div(row_sums, axis=0)

    return normalized_df


def weighted_prediction(NNs, distances):
    weights = weights_from_distances(distances)
    y_est_constituents = NNs * weights.values
    y_est = y_est_constituents.sum(axis=1)
    return y_est
# %%