# This script is used to build the a two-step thresholding procedure (bootstrapping + disparity filter)
# The output should be a list of thresholded amplitude coupling and phase-coupling (wPLI) matrices for each window per participant
​
#import libraries
import numpy as np
import networkx as nx
from scipy.signal import hilbert
from scipy.stats import pearsonr
from itertools import combinations
import mne
import os
​
​
# Set your output directory
output_dir = r'C:\Users\cerna\Downloads'  # Replace with your desired output directory
subj = '101'  # Replace so that it loops over all participants
​
# Load the label time courses
label_time_courses_file = os.path.join(output_dir, f"{subj}_label_time_courses.npy") # Make sure it loops over all participants
label_time_courses = np.load(label_time_courses_file)
​
# Load labels from the atlas
labels = mne.read_labels_from_annot('fsaverage', parc='Schaefer2018_100Parcels_7Networks_order', subjects_dir=r'C:\Users\cerna\mne_data\MNE-fsaverage-data')
​
# Compute wPLI at the level of regions
def compute_wPLI(data):
    n_regions = data.shape[1]
    wPLI_matrix = np.zeros((n_regions, n_regions))
​
    # Compute the phase of the analytic signal
    analytic_signal = hilbert(data)
    phase_data = np.angle(analytic_signal)
​
    for i in range(n_regions):
        for j in range(i+1, n_regions):  # Only compute for upper triangle
            phase_diff = phase_data[i] - phase_data[j]
            imag_part = np.abs(np.imag(np.exp(1j * phase_diff)))
            wPLI_matrix[i, j] = np.mean(imag_part) / np.mean(np.abs(np.exp(1j * phase_diff)))
            wPLI_matrix[j, i] = wPLI_matrix[i, j]  # Symmetric matrix
​
    return wPLI_matrix
​
​
def compute_amplitude_coupling(data, labels):
    """
    Compute the amplitude coupling between all pairs of regions and extract additional information.
​
    Parameters:
    data - time series data for all regions (epochs x regions x time)
    labels - list of labels representing regions
​
    Returns:
    A dictionary with amplitude coupling values and additional information.
    """
    n_regions = data.shape[1]
    coupling_info = {}
​
    # Compute the envelope of the analytic signal for each region
    envelopes = np.abs(hilbert(data, axis=2))
    mean_envelopes = envelopes.mean(axis=0)
    signs = np.sign(mean_envelopes)
​
    for i in range(n_regions):
        for j in range(i + 1, n_regions):
            corr, _ = pearsonr(envelopes[:, i].ravel(), envelopes[:, j].ravel())
​
            # Standardize the correlation to a [0, 1] scale with 0.5 as no connectivity, <0.5 as negative connectivity, and >0.5 as positive connectivity
            standardized_corr = (corr + 1) / 2  # This shifts the [-1, 1] range to [0, 1]
​
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
​
            # Record the coupling information
            coupling_info[(labels[i].name, labels[j].name)] = {
                'correlation': corr,  # Original correlation value
                'standardized_correlation': standardized_corr,  # Standardized correlation value
                'nature_of_coupling': nature_of_coupling,
                'activation_magnitudes': (mean_envelopes[i], mean_envelopes[j])
            }
​
    return coupling_info
​
​
# Function to perform aggregated bootstrapping and find optimal alpha and upper threshold
def aggregated_bootstrapping_and_threshold(windowed_graphs, num_iterations=1000, percentile=95, alpha_start=0.001,
                                           alpha_end=0.1, num_alphas=100):
    # Aggregate edge weights from all windowed graphs
    all_edge_weights = np.concatenate(
        [np.array([data['weight'] for _, _, data in G.edges(data=True)]) for G in windowed_graphs])
​
    # Perform bootstrapping on aggregated edge weights
    bootstrap_weights = []
    for _ in range(num_iterations):
        random_weights = np.random.choice(all_edge_weights, size=len(all_edge_weights), replace=True)
        bootstrap_weights.extend(random_weights)
​
    # Determine upper threshold for aggregated data
    upper_threshold = np.percentile(bootstrap_weights, percentile)
​
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
​
    optimal_alpha_idx = np.argmin(np.abs(np.diff(avg_connectivities)))
    return alphas[optimal_alpha_idx], upper_threshold
​
​
# Function to apply aggregated threshold and disparity filter to a graph
def apply_aggregated_filter(G, optimal_alpha, upper_threshold):
    G_filtered = G.copy()
    for u, v, data in G.edges(data=True):
        if data['weight'] > upper_threshold:
            G_filtered.remove_edge(u, v)
​
        elif data['weight'] ** 2 / sum(
                [d['weight'] ** 2 for _, _, d in G_filtered.edges(u, data=True)]) < optimal_alpha:
            G_filtered.remove_edge(u, v)
​
    return G_filtered
​
# Sampling rate and window parameters
sampling_rate = 512  # in Hz
window_length_seconds = 1  # desired window length in seconds (non-overlapping window)
​
# Convert time to samples
window_length_samples = int(window_length_seconds * sampling_rate)
​
# Calculate total duration in samples
total_duration_samples = int(label_time_courses.shape[0] * label_time_courses.shape[2] / sampling_rate)
​
# Compute metrics for each window
num_windows = int(total_duration_samples / window_length_samples)
windowed_wpli_matrices = []
windowed_amplitude_coupling = []
​
# Loop over all windows (non-overlapping)
for win_idx in range(num_windows):
    start_sample = win_idx * window_length_samples
    end_sample = start_sample + window_length_samples
​
    if end_sample > total_duration_samples:
        break
​
    # Extract the data for the current window
    windowed_data = label_time_courses[:, :, start_sample:end_sample]
​
    # Sub-divide the window into two halves
    first_half = windowed_data[:, :, :window_length_samples // 2]
    second_half = windowed_data[:, :, window_length_samples // 2:]
​
    # Perform computations on each half
    for half_data in [first_half, second_half]:
        # wPLI computation
        wpli_result = compute_wPLI(half_data)
        windowed_wpli_matrices.append(wpli_result)
​
        # Amplitude coupling computation
        amp_coupling_result = compute_amplitude_coupling(half_data, labels)
        windowed_amplitude_coupling.append(amp_coupling_result)
​
# Convert each windowed dPLI, wPLI, and region correlation matrix to a graph
windowed_wpli_graphs = [nx.convert_matrix.from_numpy_array(matrix, create_using=nx.Graph) for matrix in windowed_wpli_matrices]
windowed_region_correlations = [nx.convert_matrix.from_numpy_array(matrix, create_using=nx.Graph) for matrix in windowed_amplitude_coupling]
​
# Perform bootstrapping and find optimal alpha and upper threshold for dPLI, wPLI, and region correlations
optimal_alpha_wpli, upper_threshold_wpli = aggregated_bootstrapping_and_threshold(windowed_wpli_graphs, num_iterations=1000, percentile=95)
optimal_alpha_corr, upper_threshold_corr = aggregated_bootstrapping_and_threshold(windowed_region_correlations, num_iterations=1000, percentile=95)
​
# Apply the aggregated filter to each windowed graph for wPLI
thresholded_wpli_matrices = []
for G_wPLI in windowed_wpli_graphs:
    G_wPLI_thresholded = apply_aggregated_filter(G_wPLI, optimal_alpha_wpli, upper_threshold_wpli)
    wpli_matrix_thresholded = nx.convert_matrix.to_numpy_array(G_wPLI_thresholded)
    thresholded_wpli_matrices.append(wpli_matrix_thresholded)
​
# Apply the aggregated filter to each windowed graph for region correlations
thresholded_region_correlations = []
for G_corr in windowed_region_correlations:
    G_corr_thresholded = apply_aggregated_filter(G_corr, optimal_alpha_corr, upper_threshold_corr)
    corr_matrix_thresholded = nx.convert_matrix.to_numpy_array(G_corr_thresholded)
    thresholded_region_correlations.append(corr_matrix_thresholded)
​
# thresholded_dpli_matrices, thresholded_wpli_matrices, and thresholded_region_correlations contains the thresholded matrices for each window
​
# Averaging across windows for each metric
average_wpli = np.mean(np.array(thresholded_wpli_matrices), axis=0)
average_region_correlations = np.mean(np.array(thresholded_region_correlations), axis=0)
​
# Now you have average_dpli, average_wpli, and average_region_correlations
# which are the average matrices across all windows for each metric.
​
#######################################################################################################################
# The following code is commented out for future analysis
# ----------------------------------------
​
# In case we want to maitain the temporal resolution and explore the influence of fluctuations in these metrics
​
# # Function to calculate change scores
# def calculate_change_scores(metric_matrices):
#     change_scores = [np.abs(metric_matrices[i+1] - metric_matrices[i]) for i in range(len(metric_matrices)-1)]
#     return np.array(change_scores)
​
# # Calculate change scores for each metric
# change_scores_wpli = calculate_change_scores(windowed_wpli_matrices)
# change_scores_region_corr = calculate_change_scores(windowed_region_correlations)
​
# # Define a threshold for significant change
# threshold = np.percentile(change_scores_dpli, 95)  # example using 95th percentile
​
# # Repeat for wPLI and region correlations
# threshold_wpli = np.percentile(change_scores_wpli, 95)
# significant_change_points_wpli = np.where(change_scores_wpli > threshold_wpli)[0]
​
# threshold_region_corr = np.percentile(change_scores_region_corr, 95)
# significant_change_points_region_corr = np.where(change_scores_region_corr > threshold_region_corr)[0]
​
# # Visualize significant changes
# import matplotlib.pyplot as plt
​
# plt.figure(figsize=(12, 6))
# plt.subplot(311)
# plt.plot(change_scores_dpli)
# plt.title("Change Scores for dPLI")
# plt.ylabel("Change Score")
# plt.vlines(significant_change_points_dpli, ymin=np.min(change_scores_dpli), ymax=np.max(change_scores_dpli), color='red')
​
# plt.subplot(312)
# plt.plot(change_scores_wpli)
# plt.title("Change Scores for wPLI")
# plt.ylabel("Change Score")
# plt.vlines(significant_change_points_wpli, ymin=np.min(change_scores_wpli), ymax=np.max(change_scores_wpli), color='red')
​
# plt.subplot(313)
# plt.plot(change_scores_region_corr)
# plt.title("Change Scores for Region Correlations")
# plt.ylabel("Change Score")
# plt.xlabel("Window Index")
# plt.vlines(significant_change_points_region_corr, ymin=np.min(change_scores_region_corr), ymax=np.max(change_scores_region_corr), color='red')
​
# plt.tight_layout()
# plt.show()


















# Plot the ICA components as topographies in descending order of component probabilities (run until #blinks)
print("Plotting the ICA components as topographies in descending order...")
n_components_actual = ica.n_components_

# Selecting ICA components automatically using ICLabel
ic_labels = label_components(original_EEG, ica, method="iclabel")
component_labels = ic_labels["labels"]  # Extract the labels
component_probabilities = ic_labels["y_pred_proba"]  # Extract the probabilities

# Combine indices, labels, and probabilities into a single list
components = list(zip(range(n_components_actual), component_labels, component_probabilities))

# Sort components in descending order based on probabilities
sorted_components = sorted(components, key=lambda x: x[2], reverse=True)

# Initialize a counter for numbering the components
component_number = 0

for i in range(0, n_components_actual, 62):
    # Determine the range of components to plot
    component_range = sorted_components[i:min(i + 62, n_components_actual)]

    # Extract the indices for plotting
    component_indices = [comp[0] for comp in component_range]

    # Plot the components
    fig = ica.plot_components(picks=component_indices, ch_type='eeg', inst=original_EEG)

    # Set titles for each axis based on the sorted labels, probabilities, and numbering
    for ax, comp in zip(fig.axes, component_range):
        label, prob = comp[1], comp[2]
        # Change the color of the probability if it is at or above 70%
        prob_str = f"{prob:.2f}"
        if prob >= 0.70:
            ax.set_title(f"{component_number}: {label}  ({prob_str})",
                         color='red')  # Underline and change color
        else:
            ax.set_title(f"{component_number}: {label} ({prob_str})")
        component_number += 1  # Increment the component number for the next plot

    # blinks
    ica.plot_overlay(original_EEG, exclude=[0], picks="eeg")