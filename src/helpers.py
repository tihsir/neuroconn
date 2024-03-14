import numpy as np
from scipy.signal import hilbert
from scipy.stats import pearsonr
import igraph
import networkx as nx

def compute_amplitude_coupling(data, labels):
    """
    Compute the amplitude coupling between all pairs of regions and extract additional information.

    Parameters:
    data - time series data for all regions (epochs x regions x time)
    labels - list of labels representing regions

    Returns:
    A dictionary with amplitude coupling values and additional information.
    """
    n_regions = data.shape[1]
    coupling_info = {}

    # Compute the envelope of the analytic signal for each region
    envelopes = np.abs(hilbert(data, axis=2))
    mean_envelopes = envelopes.mean(axis=0)
    signs = np.sign(mean_envelopes)

    for i in range(n_regions):
        for j in range(i + 1, n_regions):
            corr, _ = pearsonr(envelopes[:, i].ravel(), envelopes[:, j].ravel())

            # Standardize the correlation to a [0, 1] scale with 0.5 as no connectivity, <0.5 as negative connectivity, and >0.5 as positive connectivity
            standardized_corr = (corr + 1) / 2  # This shifts the [-1, 1] range to [0, 1]

            # Determine the nature of coupling based on the correlation and mean envelopes' signs
            if corr > 0:
                if signs[i] == signs[j] == 1:
                    nature_of_coupling = 'co-activation'
                elif signs[i] == signs[j] == -1:
                    nature_of_coupling = 'co-deactivation'
                else:
                    nature_of_coupling = 'complex-coupling'  # Different or zero signs, positive correlation
            else:
                nature_of_coupling = 'anti-correlation' if signs[i] != signs[
                    j] else 'complex-coupling'  # Same signs, negative correlation

            # Record the coupling information
            coupling_info[(labels[i].name, labels[j].name)] = {
                'correlation': corr,  # Original correlation value
                'standardized_correlation': standardized_corr,  # Standardized correlation value
                'nature_of_coupling': nature_of_coupling,
                'activation_magnitudes': (mean_envelopes[i], mean_envelopes[j])
            }

    return coupling_info


# Function to perform aggregated bootstrapping and find optimal alpha and upper threshold
    """
    Perform aggregated bootstrapping and determine optimal alpha and upper threshold for a set of windowed graphs.

    Parameters:
    - windowed_graphs (list): A list of networkx graphs representing data in consecutive windows.
    - num_iterations (int): Number of iterations for bootstrapping the aggregated edge weights (default is 1000).
    - percentile (int): Desired percentile for determining the upper threshold (default is 95).
    - alpha_start (float): Starting value for the range of alpha values to be tested (default is 0.001).
    - alpha_end (float): Ending value for the range of alpha values to be tested (default is 0.1).
    - num_alphas (int): Number of alpha values to be tested within the specified range (default is 100).

    Returns:
    - optimal_alpha (float): The optimal alpha value determined for the aggregated data.
    - upper_threshold (float): The upper threshold calculated using bootstrapping.
    """
def aggregated_bootstrapping_and_threshold(windowed_graphs, num_iterations=1000, percentile=95, alpha_start=0.001,
                                           alpha_end=0.1, num_alphas=100):
    # Aggregate edge weights from all windowed graphs
    all_edge_weights = np.concatenate(
        [np.array([data['weight'] for _, _, data in G.edges(data=True)]) for G in windowed_graphs])

    # Perform bootstrapping on aggregated edge weights
    bootstrap_weights = []
    for _ in range(num_iterations):
        random_weights = np.random.choice(all_edge_weights, size=len(all_edge_weights), replace=True)
        bootstrap_weights.extend(random_weights)

    # Determine upper threshold for aggregated data
    upper_threshold = np.percentile(bootstrap_weights, percentile)

    # Test range of alphas to determine optimal alpha for aggregated data
    alphas = np.linspace(alpha_start, alpha_end, num_alphas)
    avg_connectivities = []
    for alpha in alphas:
        connectivities = []
        for G in windowed_graphs:
            G_filtered = G.copy()
            for u, v, weight in G.edges(data='weight'):
                if weight > upper_threshold or (G_filtered[u][v]['weight'] ** 2 / sum(
                        [d['weight'] ** 2 for _, _, d in G_filtered.edges(u, data=True)]) < alpha):
                    G_filtered.remove_edge(u, v)
            connectivities.append(np.mean(
                nx.convert_matrix.to_numpy_array(G_filtered)[np.nonzero(nx.convert_matrix.to_numpy_array(G_filtered))]))
        avg_connectivities.append(np.mean(connectivities))

    optimal_alpha_idx = np.argmin(np.abs(np.diff(avg_connectivities)))
    return alphas[optimal_alpha_idx], upper_threshold


# Function to apply aggregated threshold and disparity filter to a graph
def apply_aggregated_filter(G, optimal_alpha, upper_threshold):
    G_filtered = G.copy()
    for u, v, data in G.edges(data=True):
        if data['weight'] > upper_threshold:
            G_filtered.remove_edge(u, v)

        elif data['weight'] ** 2 / sum(
                [d['weight'] ** 2 for _, _, d in G_filtered.edges(u, data=True)]) < optimal_alpha:
            G_filtered.remove_edge(u, v)

    return G_filtered

def compute_cross_correlation(data_window):
    """Compute cross-correlation for given data window."""
    # Reshape the data to be 2D

    data_2D = data_window.reshape(data_window.shape[0], -1)
    correlation_matrix = np.corrcoef(data_2D, rowvar=True)
    return correlation_matrix

    # Compute dPLI at the level of regions


def compute_dPLI(data):
    print('Computing dPLI')
    n_regions = data.shape[1]  # Compute for regions
    dPLI_matrix = np.zeros((n_regions, n_regions))
    print(data)
    analytic_signal = hilbert(data)
    phase_data = np.angle(analytic_signal)
    for i in range(n_regions):
        for j in range(n_regions):
            if i != j:
                phase_diff = phase_data[:, i] - phase_data[:, j]
                dPLI_matrix[i, j] = np.abs(
                    np.mean(np.exp(complex(0, 1) * phase_diff)))
    return dPLI_matrix

# dPLI_matrix = compute_dPLI(label_time_courses) --> computing static, fc for the entire dataset


def disparity_filter(G, alpha=0.01):
    disparities = {}
    for i, j, data in G.edges(data=True):
        weight_sum_square = sum(
            [d['weight']**2 for _, _, d in G.edges(i, data=True)])
        disparities[(i, j)] = data['weight']**2 / weight_sum_square

    G_filtered = G.copy()
    for (i, j), disparity in disparities.items():
        if disparity < alpha:
            G_filtered.remove_edge(i, j)
    return G_filtered


def graph_to_matrix(graph, size):
    matrix = np.zeros((size, size))
    for i, j, data in graph.edges(data=True):
        matrix[i, j] = data['weight']
        matrix[j, i] = data['weight']  # Ensure symmetry
    return matrix


def threshold_matrix(matrix):
    G_temp = nx.convert_matrix.from_numpy_array(matrix)
    G_temp_thresholded = disparity_filter(G_temp)

    matrix_thresholded = np.zeros_like(matrix)
    for i, j, data in G_temp_thresholded.edges(data=True):
        matrix_thresholded[i, j] = data['weight']
        matrix_thresholded[j, i] = data['weight']
    return matrix_thresholded


def threshold_graph_by_density(G, density=0.1, directed=False):
    if density < 0 or density > 1:
        raise ValueError("Density value must be between 0 and 1.")
    num_edges_desired = int(G.number_of_edges() * density)
    sorted_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'],
                          reverse=True)
    if directed:
        G_thresholded = nx.DiGraph()
    else:
        G_thresholded = nx.Graph()
    G_thresholded.add_edges_from(sorted_edges[:num_edges_desired])
    return G_thresholded

# Convert dPLI to PLI


def dpli_to_pli(dpli_matrix):
    return 2 * np.abs(dpli_matrix - 0.5)


def compute_disparity(G):
    """
    Compute the disparity Y(i,j) for each edge in the graph.
    """
    disparities = {}
    for i, j, data in G.edges(data=True):
        weight_sum_square = sum(
            [d['weight']**2 for _, _, d in G.edges(i, data=True)])
        disparities[(i, j)] = data['weight']**2 / weight_sum_square
    return disparities

# Compute wPLI at the level of regions
def compute_wPLI(data):
    n_regions = data.shape[1]
    wPLI_matrix = np.zeros((n_regions, n_regions))

    # Compute the phase of the analytic signal
    analytic_signal = hilbert(data)
    phase_data = np.angle(analytic_signal)

    for i in range(n_regions):
        for j in range(i+1, n_regions):  # Only compute for upper triangle
            phase_diff = phase_data[i] - phase_data[j]
            imag_part = np.abs(np.imag(np.exp(1j * phase_diff)))
            wPLI_matrix[i, j] = np.mean(imag_part) / np.mean(np.abs(np.exp(1j * phase_diff)))
            wPLI_matrix[j, i] = wPLI_matrix[i, j]  # Symmetric matrix

    return wPLI_matrix