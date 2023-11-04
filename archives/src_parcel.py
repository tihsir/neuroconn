# Source-to-parcel analysis

# Import necessary libraries
import os
import glob
import numpy as np
import pandas as pd
import mne
from mne.datasets import fetch_fsaverage
from nilearn import datasets
from nilearn.image import get_data
from scipy.signal import hilbert
import matplotlib
matplotlib.use('Qt5Agg')  # Setting the backend BEFORE importing pyplot
import matplotlib.pyplot as plt
from mne_connectivity.viz import plot_connectivity_circle
from mne.viz import circular_layout
from scipy.stats import pearsonr
import networkx as nx

# Set your output directory
output_dir = r'../data/out/'  # Replace with your desired output directory
subj = '101'  # Replace with your subject ID

# Fetch the Schaefer atlas with 100 parcels
schaefer_atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100)

# Load the source space for both hemispheres
fs_dir = '../data/in/fsaverage'
fname = os.path.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
src = mne.read_source_spaces(fname, patch_stats=False, verbose=None)

# Load inverse solution file paths for both left and right hemispheres
inverse_solution_files_lh = glob.glob(os.path.join(output_dir, f"{subj}_inversesolution_epoch*.fif-lh.stc"))
inverse_solution_files_rh = glob.glob(os.path.join(output_dir, f"{subj}_inversesolution_epoch*.fif-rh.stc"))

# Calculate the total number of inverse solution files for both hemispheres
total_files_lh = len(inverse_solution_files_lh)
total_files_rh = len(inverse_solution_files_rh)

# Calculate the batch size for both hemispheres
batch_size_lh = total_files_lh // (total_files_lh // 10)  # Change 10 to your desired batch size
batch_size_rh = total_files_rh // (total_files_rh // 10)  # Change 10 to your desired batch size

# Ensure batch size is a multiple of 10 (or your desired batch size) for both hemispheres
while total_files_lh % batch_size_lh != 0:
    batch_size_lh -= 1

while total_files_rh % batch_size_rh != 0:
    batch_size_rh -= 1

# Initialize lists to store source estimates for both hemispheres
stcs_lh = []
stcs_rh = []

# Load data in batches for both hemispheres
for i in range(0, total_files_lh, batch_size_lh):
    batch_files_lh = inverse_solution_files_lh[i:i + batch_size_lh]
    batch_files_rh = inverse_solution_files_rh[i:i + batch_size_rh]

    for file_path_lh, file_path_rh in zip(batch_files_lh, batch_files_rh):
        try:
            stc_epoch_lh = mne.read_source_estimate(file_path_lh)
            stc_epoch_rh = mne.read_source_estimate(file_path_rh)
            stcs_lh.append(stc_epoch_lh)
            stcs_rh.append(stc_epoch_rh)
        except Exception as e:
            print(f"Error loading files {file_path_lh} or {file_path_rh}: {e}")

# Load labels from the atlas
labels = mne.read_labels_from_annot('fsaverage', parc='Schaefer2018_100Parcels_7Networks_order', subjects_dir=r'C:\Users\cerna\mne_data\MNE-fsaverage-data')

# Extract label time courses for both hemispheres
label_time_courses = [] # Initialize a list to store label time courses
for idx, (stc_lh, stc_rh) in enumerate(zip(stcs_lh, stcs_rh)):
    try:
        label_tc_lh = stc_lh.extract_label_time_course(labels, src=src, mode='mean_flip')
        label_tc_rh = stc_rh.extract_label_time_course(labels, src=src, mode='mean_flip')
        label_time_courses.extend([label_tc_lh, label_tc_rh])
    except Exception as e:
        print(f"Error extracting label time courses for iteration {idx}: {e}")
else:  # This block will execute if the for loop completes without encountering a break statement
    print("All time courses have been successfully extracted!")

# Convert label_time_courses to a NumPy array
label_time_courses_np = np.array(label_time_courses)

# If you prefer to save as a .csv file
# Convert to DataFrame and save as .csv
#label_time_courses_df = pd.DataFrame(label_time_courses_np)
#label_time_courses_df.to_csv(os.path.join(output_dir, f"{subj}_label_time_courses.csv"), index=False)

# Save the label time courses as a .npy file
output_dir = r'C:\Users\cerna\Downloads'  # Replace with your desired output directory
label_time_courses_file = os.path.join(output_dir, f"{subj}_label_time_courses.npy")
np.save(label_time_courses_file, label_time_courses_np)

########################################################################################################################
# VISUALIZATIONS

# Plotting Time Courses
#random_idx = np.random.randint(len(label_time_courses))
#random_time_course = label_time_courses[random_idx]

#plt.figure(figsize=(10, 6))
#plt.plot(random_time_course)
#plt.title(f'Time Course for Randomly Selected Region: {random_idx}')
#plt.xlabel('Time')
#plt.ylabel('Amplitude')
#plt.show()

# Connectivity Visualization for left hemisphere
#num_regions = len(label_time_courses[0])
#connectivity_matrix = np.zeros((num_regions, num_regions))

#for i in range(num_regions):
#    for j in range(num_regions):
#        connectivity_matrix[i, j], _ = pearsonr(label_time_courses[0][i], label_time_courses[0][j])

#plt.figure(figsize=(10, 10))
#plt.imshow(connectivity_matrix, cmap='viridis', origin='lower')
#plt.title('Connectivity Matrix')
#plt.xlabel('Region')
#plt.ylabel('Region')
#plt.colorbar(label='Pearson Correlation')
#plt.show()

# Average connectivity matrix across all epochs and hemispheres:

# Initialize connectivity matrix
#num_epochs_hemispheres = len(label_time_courses)
#num_regions = label_time_courses[0].shape[0]
#all_connectivity_matrices = np.zeros((num_epochs_hemispheres, num_regions, num_regions))

# Compute average connectivity for each epoch and hemisphere
#for k in range(num_epochs_hemispheres):
#    for i in range(num_regions):
#        for j in range(num_regions):
#            all_connectivity_matrices[k, i, j], _ = pearsonr(label_time_courses[k][i], label_time_courses[k][j])

# Average across all epochs and hemispheres
#avg_connectivity_matrix = np.mean(all_connectivity_matrices, axis=0)

# Visualization
#plt.figure(figsize=(10, 10))
#plt.imshow(avg_connectivity_matrix, cmap='viridis', origin='lower')
#plt.title('Average Connectivity Matrix')
#plt.xlabel('Region')
#plt.ylabel('Region')
#plt.colorbar(label='Pearson Correlation')
#plt.show()

########################################################################################################################
# All-to-all connectivity analysis

# Set your output directory
output_dir = r'C:\Users\cerna\Downloads'  # Replace with your desired output directory
subj = '101'  # Replace with your subject ID

# Load the label time courses
label_time_courses_file = os.path.join(output_dir, f"{subj}_label_time_courses.npy")
label_time_courses = np.load(label_time_courses_file)

# Load labels from the atlas
labels = mne.read_labels_from_annot('fsaverage', parc='Schaefer2018_100Parcels_7Networks_order', subjects_dir=r'C:\Users\cerna\mne_data\MNE-fsaverage-data')

# Group labels by network
networks = {}
for label in labels:
    # Extract network name from label name (assuming format: 'NetworkName_RegionName')
    network_name = label.name.split('_')[0]
    if network_name not in networks:
        networks[network_name] = []
    networks[network_name].append(label)

# Organize regions by their network affiliations and extract the desired naming convention
ordered_regions = []
network_labels = []  # This will store the network each region belongs to

for label in labels:
    # Extract the desired naming convention "PFCl_1-lh" from the full label name
    parts = label.name.split('_')
    region_name = '_'.join(parts[2:])
    ordered_regions.append(region_name)

    # Extract the network name and store it in network_labels
    network_name = parts[2]
    network_labels.append(network_name)

# Compute dPLI at the level of regions
def compute_dPLI(data):
    n_regions = data.shape[1]  # Compute for regions
    dPLI_matrix = np.zeros((n_regions, n_regions))
    analytic_signal = hilbert(data)
    phase_data = np.angle(analytic_signal)
    for i in range(n_regions):
        for j in range(n_regions):
            if i != j:
                phase_diff = phase_data[:, i] - phase_data[:, j]
                dPLI_matrix[i, j] = np.abs(np.mean(np.exp(complex(0, 1) * phase_diff)))
    return dPLI_matrix

#dPLI_matrix = compute_dPLI(label_time_courses) --> computing static, fc for the entire dataset, if needed

# Time-resolved dPLI computation
sampling_rate = 512  # in Hz
window_length_seconds = 1
step_size_seconds = 0.5

# Total duration in samples
num_epochs_per_hemisphere = label_time_courses.shape[0] / 2  # Assuming the structure is the same as label_time_courses in Code 2
duration_per_epoch = label_time_courses.shape[2] / sampling_rate
total_duration_samples = int(num_epochs_per_hemisphere * duration_per_epoch * sampling_rate)

window_length_samples = int(window_length_seconds * sampling_rate)
step_size_samples = int(step_size_seconds * sampling_rate)

num_windows = int((total_duration_samples - window_length_samples) / step_size_samples) + 1
windowed_dpli_matrices = []
windowed_cross_correlation_matrices = []

for win_idx in range(num_windows):
    start_sample = win_idx * step_size_samples
    end_sample = start_sample + window_length_samples

    # Check if end_sample exceeds the total number of samples
    if end_sample > total_duration_samples:
        break

    # Calculate epoch and sample indices
    start_epoch = start_sample // label_time_courses.shape[2]
    start_sample_in_epoch = start_sample % label_time_courses.shape[2]
    end_epoch = end_sample // label_time_courses.shape[2]
    end_sample_in_epoch = end_sample % label_time_courses.shape[2]

    # Extract data across epochs
    if start_epoch == end_epoch:
        windowed_data = label_time_courses[start_epoch, :, start_sample_in_epoch:end_sample_in_epoch]
    else:
        first_part = label_time_courses[start_epoch, :, start_sample_in_epoch:]
        samples_needed_from_second_epoch = window_length_samples - first_part.shape[1]
        second_part = label_time_courses[end_epoch, :, :samples_needed_from_second_epoch]
        windowed_data = np.concatenate((first_part, second_part), axis=1)

    # dPLI computation
    dpli_result = compute_dPLI(windowed_data)
    windowed_dpli_matrices.append(dpli_result)

# Check the number of windows in the list
num_of_windows = len(windowed_dpli_matrices)
print(f"Total number of windows: {num_of_windows}")

# Calculating the average connectivity across all windows (or choose a specific window)
#chosen_window = 0 # Choose a window
dPLI_matrix = windowed_dpli_matrices # can do windowed_dpli_matrices[chosen_window] to only choose one window
G_dPLI_list = [nx.convert_matrix.from_numpy_array(matrix) for matrix in windowed_dpli_matrices]

# Determining the optimal alpha value for disparity filter before applying thresholding

# Disparity filter
def disparity_filter(G, alpha=0.01):
    disparities = {}
    for i, j, data in G.edges(data=True):
        weight_sum_square = sum([d['weight']**2 for _, _, d in G.edges(i, data=True)])
        disparities[(i, j)] = data['weight']**2 / weight_sum_square

    G_filtered = G.copy()
    for (i, j), disparity in disparities.items():
        if disparity < alpha:
            G_filtered.remove_edge(i, j)
    return G_filtered

# Determining alpha
alphas = np.linspace(0.001, 0.1, 100)  # Example range of alphas to test
avg_connectivities = []

for alpha in alphas:
    avg_conn_for_alpha = []
    for dpli_matrix in windowed_dpli_matrices:
        G_dPLI = nx.convert_matrix.from_numpy_array(dpli_matrix)
        G_dPLI_thresholded = disparity_filter(G_dPLI, alpha=alpha)

        # Convert the thresholded graph back to a matrix
        dPLI_matrix_thresholded = nx.convert_matrix.to_numpy_array(G_dPLI_thresholded)

        # Compute the average functional connectivity of the thresholded matrix
        avg_conn = np.mean(dPLI_matrix_thresholded)
        avg_conn_for_alpha.append(avg_conn)

    # Compute the average of averages for this alpha
    avg_connectivities.append(np.mean(avg_conn_for_alpha)) # AKA, for this alpha, what's the typical (or average) connectivity value across all windows?

# Find the alpha that gives the most stable average connectivity
optimal_alpha = alphas[np.argmin(np.diff(avg_connectivities))]

print(f"Optimal alpha: {optimal_alpha}")

# Thresholding the connectivity matrix
G_dPLI_thresholded_list = [disparity_filter(G, alpha=optimal_alpha) for G in G_dPLI_list]  # Use the optimal alpha value here

########################################################################################################################
# Next --> fc between groups and directed minimum spanning tree to examine nature of the fc difference between groups

