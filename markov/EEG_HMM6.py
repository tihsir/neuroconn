
# Integrating a Hidden Markov Model for brain state indentification

# Step 1: Compute the orthogonalized envelope of the analytic signal

import os
import itertools
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import mne
from mne_connectivity import symmetric_orth
from hmmlearn import hmm
from scipy.signal import hilbert  # For Hilbert transform
from scipy.signal import resample, butter, lfilter # For downsampling
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import networkx as nx
import seaborn as sns
from scipy.optimize import fminbound

# Set your output directory
output_dir = r'C:\Users\cerna\Downloads\label_time_courses'  # Your output directory
subj = '101'  # Subject ID

# Load the label time courses
label_time_courses_file = os.path.join(output_dir, f"{subj}_label_time_courses.npy")
label_time_courses = np.load(label_time_courses_file)

# Original and target sampling frequencies
original_sampling_freq = 513
target_sampling_freq = 250

# Calculate the downsampling factor
downsampling_factor = int(np.floor(original_sampling_freq / target_sampling_freq))

# Downsampling with Anti-Aliasing Filtering
def downsample_with_filtering(data, original_fs, target_fs):
    """Downsamples data with an anti-aliasing filter."""
    # Design an anti-aliasing lowpass filter
    nyq_rate = original_fs / 2.0
    cutoff_freq = target_fs / 2.0  # New Nyquist frequency
    normalized_cutoff = cutoff_freq / nyq_rate
    b, a = butter(4, normalized_cutoff, btype='low')

    # Apply the filter
    filtered_data = lfilter(b, a, data, axis=2)

    # Calculate new number of samples
    duration = data.shape[2] / original_fs
    new_num_samples = int(duration * target_fs)

    # Resample data
    downsampled_data = resample(filtered_data, new_num_samples, axis=2)

    return downsampled_data

# Apply downsampling
downsampled_label_time_courses = downsample_with_filtering(label_time_courses, original_sampling_freq, target_sampling_freq)

# Verify the new shape
print(downsampled_label_time_courses.shape)  # Expected shape: (num_labels, num_channels, new_num_samples)

# Load labels from the atlas
labels = mne.read_labels_from_annot('fsaverage', parc='Schaefer2018_100Parcels_7Networks_order', subjects_dir=r'C:\Users\cerna\mne_data\MNE-fsaverage-data')

def apply_orthogonalization(downsampled_label_time_courses):
    # Calculate the analytic signal for each epoch/sample
    analytic_signal = hilbert(downsampled_label_time_courses, axis=2)  # Apply Hilbert transform along the correct axis

    # Extract the amplitude envelope from the analytic signal
    amplitude_envelope = np.abs(analytic_signal)  # Calculate the amplitude envelope

    # Collinearity Check using QR-Decomposition
    Q, R = np.linalg.qr(amplitude_envelope.reshape(-1, amplitude_envelope.shape[-1]).T)  # Reshape amplitude_envelope and transpose for QR
    rank = np.linalg.matrix_rank(R)

    # Ensure 'rank' is an integer and perform the comparison
    if isinstance(rank, np.integer) and rank < amplitude_envelope.shape[-1]:  # Use the last dimension of amplitude_envelope for comparison
        print("Warning: Signals appear to be collinear.")
        non_orthogonal_label_pairs = []  # Find combinations of non-orthogonal signals
        tol = 1e-8  # Tolerance for near-zero values in R
        for i in range(R.shape[0]):
            for j in range(i + 1, R.shape[1]):
                if abs(R[i, j]) > tol:
                    non_orthogonal_label_pairs.append((i, j))
        print("Non-orthogonal label pairs:", non_orthogonal_label_pairs)

    orthogonalized_data = symmetric_orth(amplitude_envelope)

    # Check for NaNs or Infs in your data and add regularization if needed
    if np.any(np.isnan(orthogonalized_data)) or np.any(np.isinf(orthogonalized_data)):
        raise ValueError("Data contains NaNs or infinite values")

    # Optional regularization
    regularized_data = orthogonalized_data + 1e-6 * np.random.randn(*orthogonalized_data.shape)

    return regularized_data

# Orthogonalization
orthogonalized_data = apply_orthogonalization(downsampled_label_time_courses)

#-----------------------------------------------------------------------------------------------------------------------

# Step 2: Determine the Optimal Number of States for the HMM

# Compute the variance of your features to set a variance floor later
feature_variances = np.var(orthogonalized_data, axis=0)

# Choose a small fraction (e.g., 1% or 0.1%) of the maximum variance as the variance floor
fraction_of_max_variance = 0.05  # Adjust as needed
variance_floor = fraction_of_max_variance * np.max(feature_variances)

# Handle NaNs and Infs in features (using masking)
features = np.mean(orthogonalized_data, axis=2)
features = np.ma.masked_invalid(features).filled(0)

# Reshape orthogonalized_data using array.reshape (-1, 1)
# New shape will be (samples/epochs, labels * sampling frequency)
reshaped_data = orthogonalized_data.reshape(-1, 1)

# Reduce dimensionality to speed up HMM fitting
pca = PCA(n_components=0.95)  # Retain 95% of the variance

# Fit PCA to the normalized data
pca_data = pca.fit_transform(reshaped_data)

# Standardize the PCA-transformed data
scaler = StandardScaler()
pca_data = scaler.fit_transform(pca_data)

# Initialize lists to store AIC and BIC values
aics = []
bics = []

# Define the range of state numbers to test based on previous literature
state_numbers = range(3, 16)


for n_states in state_numbers:
    # Initialize the HMM model with diagonal covariance
    model = hmm.GaussianHMM(n_components=n_states, n_iter=50, covariance_type='full', tol=1e-7, verbose=False,
                            params='st', init_params='stmc')  # Add smoothing parameter

    # Fit the model using the PCA-transformed data
    model.fit(pca_data)

    # Calculate AIC and BIC for the current model
    log_likelihood = model.score(pca_data)
    n_params = n_states * (2 * pca_data.shape[1] - 1)  # Adjusted for diagonal covariance
    aic = 2 * n_params - 2 * log_likelihood
    bic = np.log(pca_data.shape[0]) * n_params - 2 * log_likelihood

    # Store the AIC and BIC values
    aics.append(aic)
    bics.append(bic)

# Determine the optimal number of states based on the lowest AIC and BIC
optimal_states_aic = state_numbers[np.argmin(aics)]
optimal_states_bic = state_numbers[np.argmin(bics)]

# Plot AIC and BIC values
plt.figure(figsize=(10, 5))
plt.plot(state_numbers, aics, label='AIC')
plt.plot(state_numbers, bics, label='BIC')
plt.xlabel('Number of States')
plt.ylabel('AIC/BIC Value')
plt.title('AIC/BIC Values for Different Number of States')
plt.legend()
plt.show()

# Take the average of the optimal states based on AIC and BIC
optimal_states = 4#int((optimal_states_aic + optimal_states_bic) / 2) # Optimal states often called K in the literature
print(f"Optimal number of states based on AIC/BIC: {optimal_states}")

#-----------------------------------------------------------------------------------------------------------------------

# Step 3 Alternative: Variational Bayesian HMM for State Number Estimation


#-----------------------------------------------------------------------------------------------------------------------

# Step 3: Train the HMM with the Optimal Number of States

# Replace NaNs and Infs using masking
features = np.mean(orthogonalized_data, axis=2)
features = np.ma.masked_invalid(features).filled(0)

# Standardized features
scaler = StandardScaler()

# Fit and transform the reshaped data only once
features = scaler.fit_transform(features)

# HMM Fitting with simpler covariance type and fewer states
model = hmm.GaussianHMM(n_components=optimal_states, n_iter=50, covariance_type='full', tol=1e-7, verbose=False,
                        params='st', init_params='stmc')
model.fit(features)

# State Decoding (use lagged_data)
state_sequence = model.predict(features)  # Find the most likely state sequence

# Probability calculation for each time step
state_probs = model.predict_proba(features)

# Visualize the state probabilities over time---------------------------------------------------------------------------

# Create a time array assuming equal time intervals and starting at 0
time_points = np.linspace(0, 60, state_probs.shape[0])  # Replace 60 by the actual total time in seconds

# Create a stacked area plot
plt.figure(figsize=(15, 5))

# Define the colors for each state
colors = ['red', 'orange', 'blue', 'green', 'purple', 'yellow']  # Define more colors if you have more states

# The number of state levels should be derived from optimal_states
state_labels = [f'State {i+1}' for i in range(optimal_states)]

# Create a stackplot
stacks = plt.stackplot(time_points, state_probs.T, colors=colors, edgecolor='none')

# Create a legend
legend_patches = [Patch(color=color, label=label) for color, label in zip(colors, state_labels)]
plt.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.05, 1), title='States')

# Add labels and title
plt.xlabel('Time (s)')
plt.ylabel('Probability of state activation')
plt.title('State Probabilities Over Time')

# Customize the y-axis limits and ticks to match the probability range
plt.ylim(0, 1)
plt.yticks(np.linspace(0, 1, 11))

# Add grid for better readability
#plt.grid(True)

# Show the plot
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make room for the legend
plt.show()

#-----------------------------------------------------------------------------------------------------------------------

# Step 4: Calculate temporal features/dynamics

# Compute fractional occupancy: fraction of time spent in each state
fractional_occupancy = np.array([np.sum(state_sequence == i) / len(state_sequence) for i in range(optimal_states)])

# Compute transition probabilities
transition_counts = np.zeros((optimal_states, optimal_states))
for (i, j) in zip(state_sequence[:-1], state_sequence[1:]):
    transition_counts[i, j] += 1
transition_probabilities = transition_counts / np.sum(transition_counts, axis=1, keepdims=True)


# Compute mean lifetime (dwell time) in each state: average time spent in each state before transitioning
mean_lifetime = np.zeros(optimal_states)
for i in range(optimal_states):
    # Identify the indices where state changes
    change_indices = np.where(np.diff(state_sequence == i, prepend=False, append=False))[0]
    # Calculate segment lengths by differencing indices of changes; add 1 because diff loses 1
    segment_lengths = np.diff(change_indices) + 1
    # Compute mean segment length for state i
    mean_lifetime[i] = np.mean(segment_lengths) if len(segment_lengths) > 0 else 0

# Print the calculated temporal features
print("Fractional Occupancy:", fractional_occupancy)
print("Transition Probabilities:\n", transition_probabilities)
print("Mean Lifetime:", mean_lifetime)

# Visualizations for temporal features----------------------------------------------------------------------------------

# Pie chart to visualize fractional occupancy
labels_FO = [f'State {i}' for i in range(len(fractional_occupancy))]
plt.figure(figsize=(8, 8))
plt.pie(fractional_occupancy, labels=labels_FO, autopct='%1.1f%%', startangle=140)
plt.title('Fractional Occupancy of Each State')
plt.show()

# Convert the mean lifetime from samples to milliseconds
# by dividing by the sampling frequency and multiplying by 1000
mean_lifetime_ms = mean_lifetime / target_sampling_freq * 1000  # Conversion to milliseconds

# Bar plot to visualize mean lifetime in each state (in milliseconds)
plt.figure(figsize=(8, 6))
plt.bar(range(optimal_states), mean_lifetime_ms, color='skyblue', tick_label=[f'State {i}' for i in range(optimal_states)])
plt.title('Mean Lifetime in Each State')
plt.xlabel('State')
plt.ylabel('Mean Lifetime (ms)')
plt.xticks(rotation=45)  # Rotate state labels for better readability
plt.show()


# Visualization of the Transition Probabilities Matrix as a Heatmap
sns.heatmap(transition_probabilities, annot=True, cmap='Blues', fmt='.2f')
plt.title('Transition Probability Matrix')
plt.xlabel('To State')
plt.ylabel('From State')
plt.show()

# Visualize transition probabilities as a directed graph --> Need to fix the bidirectional edges
G = nx.DiGraph()

# Add nodes with labels for each state
for i in range(optimal_states):
    G.add_node(i, label=f'State {i}')

# Add edges with weights for transition probabilities
for i in range(optimal_states):
    for j in range(optimal_states):
        if transition_probabilities[i, j] > 0:  # Only add edge if probability > 0
            G.add_edge(i, j, weight=transition_probabilities[i, j])

# Position nodes using the spring layout
pos = nx.spring_layout(G)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=700)

# Separate out the self-loops and bidirectional edges
self_loops = [(i, j) for i, j in G.edges() if i == j]
bidirectional_edges = [(i, j) for i, j in G.edges() if (j, i) in G.edges() and i < j]
normal_edges = [(i, j) for i, j in G.edges() if (j, i) not in G.edges() and (i, j) not in self_loops]

# Draw self-loops with a specific style
for i, j in self_loops:
    nx.draw_networkx_edges(G, pos, edgelist=[(i, j)], arrowstyle='->',
                           connectionstyle='arc3,rad=0.2',
                           arrowsize=20, edge_color='red', width=2)

# Draw bidirectional edges with different curvatures
for i, j in bidirectional_edges:
    # Forward direction
    nx.draw_networkx_edges(G, pos, edgelist=[(i, j)], arrowstyle='->',
                           connectionstyle='arc3,rad=0.1',
                           arrowsize=20, edge_color='blue', width=2 * G[i][j]['weight'])

    # Reverse direction
    nx.draw_networkx_edges(G, pos, edgelist=[(j, i)], arrowstyle='->',
                           connectionstyle='arc3,rad=-0.1',
                           arrowsize=20, edge_color='blue', width=2 * G[j][i]['weight'])

# Draw normal edges
for i, j in normal_edges:
    nx.draw_networkx_edges(G, pos, edgelist=[(i, j)], arrowstyle='->',
                           arrowsize=20, edge_color='blue', width=2 * G[i][j]['weight'])

# Draw node labels
nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')

# Draw edge labels with transition probabilities
edge_labels = {(i, j): f'{G[i][j]["weight"]:.2f}' for i, j in G.edges()}
for (i, j), label in edge_labels.items():
    x, y = pos[i]
    dx, dy = pos[j]
    if i == j:
        plt.text(x, y + 0.1, s=label, bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'),
                 horizontalalignment='center')
    else:
        label_pos = 0.5
        if (j, i) in bidirectional_edges:
            label_pos = 0.3 if i < j else 0.7  # Offset the label position for clarity
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(i, j): label}, font_color='black', label_pos=label_pos)

plt.axis('off')
plt.title('Transition Probabilities Between States')
plt.show()

#-----------------------------------------------------------------------------------------------------------------------

# Step 5: Calculate spatial features

# Function to calculate functional connectivity matrices for each state window
def calculate_functional_connectivity(orthogonalized_data, state_sequence, optimal_states):
    correlation_matrices = {}
    positive_correlations = {}
    negative_correlations = {}

    for state in range(optimal_states):
        state_indices = np.where(state_sequence == state)[0]

        block_starts = np.where(np.diff(state_indices) > 1)[0] + 1
        block_starts = np.insert(block_starts, 0, 0)
        block_ends = np.append(block_starts[1:] - 1, len(state_indices) - 1)
        state_blocks = zip(state_indices[block_starts], state_indices[block_ends] + 1)

        for i, (start_index, end_index) in enumerate(state_blocks):
            state_data = orthogonalized_data[:, :, start_index:end_index]

            # Select the time and sample dimensions for correlation
            reshaped_data = state_data.swapaxes(1, 2).reshape(102, -1)

            # Impute NaNs if necessary
            if np.isnan(state_data).any():  # Assuming orthogonalized_data is used here
                reshaped_data = np.nan_to_num(reshaped_data)

            corr_matrix = np.corrcoef(reshaped_data)  # Or use NaN-robust correlation
            correlation_matrices[(state, i)] = corr_matrix

            upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
            positive_correlations[(state, i)] = corr_matrix[upper_tri_indices][corr_matrix[upper_tri_indices] > 0]
            negative_correlations[(state, i)] = corr_matrix[upper_tri_indices][corr_matrix[upper_tri_indices] < 0]

    return correlation_matrices, positive_correlations, negative_correlations

# Calculate functional connectivity matrices for the synthetic data
correlation_matrices, positive_correlations, negative_correlations = calculate_functional_connectivity(
    orthogonalized_data, state_sequence, optimal_states)


# Visualization for spatial features-------------------------------------------------------------------------------------

# Visualize the distribution of positive and negative correlations for each state
all_correlations_by_state = {}
for state in range(optimal_states):
    all_correlations_by_state[state] = (np.concatenate([pos_corr for key, pos_corr in positive_correlations.items() if key[0] == state]),
                                        np.concatenate([neg_corr for key, neg_corr in negative_correlations.items() if key[0] == state]))

# Visualize the distribution of correlations summarized across windows
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
for state, (pos_corr, _) in all_correlations_by_state.items():
    sns.kdeplot(pos_corr, label=f'State {state}', fill=True)
plt.title('Positive Correlations (Summarized)')
plt.xlabel('Correlation Value')
plt.ylabel('Density')
plt.legend()

plt.subplot(1, 2, 2)
for state, (_, neg_corr) in all_correlations_by_state.items():
    sns.kdeplot(neg_corr, label=f'State {state}', fill=True)
plt.title('Negative Correlations (Summarized)')
plt.xlabel('Correlation Value')
plt.ylabel('Density')
plt.legend()

plt.tight_layout()
plt.show()

# Visualize a circular plot for each state (one/state)
for state in range(optimal_states):
    plt.figure(figsize=(8, 8))
    G = nx.from_numpy_array(correlation_matrices[(state, 0)])
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=3000, edge_cmap=plt.cm.Blues, font_weight='bold')
    plt.title(f'Functional Connectivity for State {state}')
    plt.show()


# Visualize a correlation matrix for each state (one/state)
for state in range(optimal_states):
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrices[(state, 0)], cmap='coolwarm', center=0, annot=False)
    plt.title(f'Correlation Matrix for State {state}')
    plt.show()

#-----------------------------------------------------------------------------------------------------------------------

# Step 6: Thresholding the Functional Connectivity Matrices using an Orthogonalized Minimum Spanning Tree

import time
import cProfile

# Adaptive thresholding approach based on bootstrapping and alpha thresholding
def aggregated_bootstrapping_and_alpha_threshold(windowed_graphs, num_iterations=10, alpha_start=0.001, alpha_end=0.1, num_alphas=100): # will start with 10 iterations to estimate how long it would be with 1k iterations
    # Aggregate edge weights from all windowed graphs
    all_edge_weights = np.concatenate([np.array([data['weight'] for _, _, data in G.edges(data=True)]) for G in windowed_graphs])

    # Perform bootstrapping on aggregated edge weights
    bootstrap_weights = []
    for _ in range(num_iterations):
        random_weights = np.random.choice(all_edge_weights, size=len(all_edge_weights), replace=True)
        bootstrap_weights.extend(random_weights)

    # Calculate the median from the bootstrapped weights
    bootstrap_median = np.median(bootstrap_weights)

    # Test range of alphas to determine optimal alpha for aggregated data
    alphas = np.linspace(alpha_start, alpha_end, num_alphas)
    avg_connectivities = []
    for alpha in alphas:
        connectivities = []
        for G in windowed_graphs:
            G_filtered = G.copy()
            for u, v, weight in G.edges(data='weight'):
                normalized_weight = weight / bootstrap_median
                if normalized_weight ** 2 / sum([d['weight'] ** 2 for _, _, d in G_filtered.edges(u, data=True)]) < alpha:
                    G_filtered.remove_edge(u, v)
            connectivities.append(np.mean(nx.convert_matrix.to_numpy_array(G_filtered)[np.nonzero(nx.convert_matrix.to_numpy_array(G_filtered))]))
        avg_connectivities.append(np.mean(connectivities))

    optimal_alpha_idx = np.argmin(np.abs(np.diff(avg_connectivities)))
    return alphas[optimal_alpha_idx], bootstrap_median

def apply_alpha_filter(G, optimal_alpha, bootstrap_median):
    G_filtered = G.copy()
    for u, v, data in G.edges(data=True):
        normalized_weight = data['weight'] / bootstrap_median
        if normalized_weight ** 2 / sum([d['weight'] ** 2 for _, _, d in G_filtered.edges(u, data=True)]) < optimal_alpha:
            G_filtered.remove_edge(u, v)

    return G_filtered

# Function to apply thresholding to the functional connectivity matrices
def threshold_functional_connectivity(correlation_matrices):
    thresholded_correlation_matrices = {}
    thresholded_positive_correlations = {}
    thresholded_negative_correlations = {}

    # Convert correlation matrices to NetworkX graphs
    windowed_graphs = [nx.from_numpy_array(corr_matrix) for corr_matrix in correlation_matrices.values()]

    # Determine the optimal alpha using aggregated bootstrapping
    optimal_alpha, bootstrap_median = aggregated_bootstrapping_and_alpha_threshold(windowed_graphs)

    for key, G in zip(correlation_matrices.keys(), windowed_graphs):
        # Apply alpha thresholding to each graph
        G_filtered = apply_alpha_filter(G, optimal_alpha, bootstrap_median)
        filtered_matrix = nx.convert_matrix.to_numpy_array(G_filtered)

        # Save the thresholded correlation matrix
        thresholded_correlation_matrices[key] = filtered_matrix

        # Recalculate and save positive and negative correlations based on the thresholded matrix
        upper_tri_indices = np.triu_indices_from(filtered_matrix, k=1)
        pos_corrs = filtered_matrix[upper_tri_indices][filtered_matrix[upper_tri_indices] > 0]
        neg_corrs = filtered_matrix[upper_tri_indices][filtered_matrix[upper_tri_indices] < 0]

        # Store the recalculated positive and negative correlations
        state = key[0]  # Assuming the key structure is (state, window)
        if state not in thresholded_positive_correlations:
            thresholded_positive_correlations[state] = pos_corrs
            thresholded_negative_correlations[state] = neg_corrs
        else:
            thresholded_positive_correlations[state] = np.concatenate((thresholded_positive_correlations[state], pos_corrs))
            thresholded_negative_correlations[state] = np.concatenate((thresholded_negative_correlations[state], neg_corrs))

    return thresholded_correlation_matrices, thresholded_positive_correlations, thresholded_negative_correlations

# Profile start time
start_time = time.time()

# Applying thresholding to the functional connectivity matrices
thresholded_correlation_matrices, thresholded_positive_correlations, thresholded_negative_correlations = threshold_functional_connectivity(correlation_matrices)

# Profile end time
end_time = time.time()
print(f"Thresholding took {end_time - start_time:.2f} seconds")

# Run with profiling
cProfile.run("threshold_functional_connectivity(correlation_matrices)")

#-----------------------------------------------------------------------------------------------------------------------

# Visualize correlation matrices for each state (one/state) after thresholding
for state in range(optimal_states):
    plt.figure(figsize=(8, 6))
    sns.heatmap(thresholded_correlation_matrices[(state, 0)], cmap='coolwarm', center=0, annot=False)
    plt.title(f'Correlation Matrix for State {state}')
    plt.show()

#-----------------------------------------------------------------------------------------------------------------------

# Step 6: Within/Between-Network Connectivity

# Initialize a dictionary to hold the network assignments
networks = {
    'Visual': [],
    'Somatomotor': [],
    'DorsalAttention': [],
    'VentralAttention': [],
    'Limbic': [],
    'Frontoparietal': [],
    'Default': []
}

# Function to extract the network name from the label name
def get_network_name(label_name):
    if 'Vis' in label_name:
        return 'Visual'
    elif 'SomMot' in label_name:
        return 'Somatomotor'
    elif 'DorsAttn' in label_name:
        return 'DorsalAttention'
    elif 'SalVentAttn' in label_name or 'VentAttn' in label_name:
        return 'VentralAttention'
    elif 'Limbic' in label_name:
        return 'Limbic'
    elif 'Cont' in label_name or 'Frontoparietal' in label_name:
        return 'Frontoparietal'
    elif 'Default' in label_name:
        return 'Default'
    else:
        return None

# Assign each region to its network
for i, label in enumerate(labels):
    network_name = get_network_name(label.name)
    if network_name:
        networks[network_name].append(i)  # Append the index of the region

# Print the networks and their assigned regions' indices for verification
for network, regions in networks.items():
    print(f"{network}: {regions}")

# Calculate within-network and between-network connectivity for each state window
def calculate_network_connectivity(correlation_matrices, networks, labels):
    within_network_connectivity = {}
    between_network_connectivity = {}

    # Iterate over each state window
    for (state, window), corr_matrix in correlation_matrices.items():
        within_network_connectivity[(state, window)] = []
        between_network_connectivity[(state, window)] = []

        # Calculate within-network connectivity
        for network_name, regions in networks.items():
            network_corr_matrix = corr_matrix[np.ix_(regions, regions)]
            upper_tri_indices = np.triu_indices_from(network_corr_matrix, k=1)

            for i, j in zip(*upper_tri_indices):
                region1 = labels[regions[i]].name
                region2 = labels[regions[j]].name
                corr_value = network_corr_matrix[i, j]
                within_network_connectivity[(state, window)].append(
                    f"[{network_name}]: {region1} - {corr_value:.2f} - {region2}")

        # Calculate between-network connectivity
        for net1, net2 in itertools.combinations(networks.keys(), 2):
            regions1, regions2 = networks[net1], networks[net2]
            between_corr_matrix = corr_matrix[np.ix_(regions1, regions2)]

            for i, j in itertools.product(range(len(regions1)), range(len(regions2))):
                region1 = labels[regions1[i]].name
                region2 = labels[regions2[j]].name
                corr_value = between_corr_matrix[i, j]
                between_network_connectivity[(state, window)].append(
                    f"[{net1}, {net2}]: {region1} - {corr_value:.2f} - {region2}")

    return within_network_connectivity, between_network_connectivity

# Calculate connectivity with correlation values
within_network_conn_values, between_network_conn_values = calculate_network_connectivity(
    correlation_matrices, networks, labels)


























