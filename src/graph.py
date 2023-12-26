# This script is used to build the a two-step thresholding procedure (bootstrapping + disparity filter)
# The output should be a list of thresholded dPLI matrices for each window per participant
from sklearn.decomposition import PCA

from scipy import signal
import sklearn as sk
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.stats import zscore

from helpers import aggregated_bootstrapping_and_threshold, apply_aggregated_filter, compute_dPLI

# defining input and output directory
files_in = '../data/in/subjects/'
files_out = '../data/out/subjects/'


# loading list of subject names from txt file
names = open("../batch/names.txt", "r")
subject_list = names.read().split('\n')
modes = ['EC', 'EO']


all_graphs = []

for subject in subject_list[:2]:
    for mode in modes:
        tc_file = files_out + subject +'/'+mode +'/'+subject+'_label_time_courses.npy'
        label_time_courses = np.load(tc_file)
        sampling_rate = 512  # in Hz
        window_length_seconds = 1  # desired window length in seconds
        step_size_seconds = 0.5  # desired step size in seconds

        # Convert time to samples
        window_length_samples = int(window_length_seconds * sampling_rate)
        step_size_samples = int(step_size_seconds * sampling_rate)

        # Calculate total duration in samples
        num_epochs_per_hemisphere = label_time_courses.shape[0] / 2
        duration_per_epoch = label_time_courses.shape[2] / sampling_rate
        total_duration_samples = int(num_epochs_per_hemisphere * duration_per_epoch * sampling_rate)

        # Compute dPLI for each window
        num_windows = int((total_duration_samples - window_length_samples) / step_size_samples) + 1
        windowed_dpli_matrices = []

        for win_idx in range(num_windows):
            start_sample = win_idx * step_size_samples
            end_sample = start_sample + window_length_samples

            if end_sample > total_duration_samples:
                break

            start_epoch = start_sample // label_time_courses.shape[2]
            start_sample_in_epoch = start_sample % label_time_courses.shape[2]
            end_epoch = end_sample // label_time_courses.shape[2]
            end_sample_in_epoch = end_sample % label_time_courses.shape[2]

            if start_epoch == end_epoch:
                windowed_data = label_time_courses[start_epoch, :, start_sample_in_epoch:end_sample_in_epoch]
            else:
                first_part = label_time_courses[start_epoch, :, start_sample_in_epoch:]
                samples_needed_from_second_epoch = window_length_samples - first_part.shape[1]
                second_part = label_time_courses[end_epoch, :, :samples_needed_from_second_epoch]
                windowed_data = np.concatenate((first_part, second_part), axis=1)

            dpli_result = compute_dPLI(windowed_data)
            windowed_dpli_matrices.append(dpli_result)
            all_graphs.append(dpli_result)

        all_graphs.append(windowed_dpli_matrices)

print(all_graphs)

alpha, thresh = aggregated_bootstrapping_and_threshold(all_graphs)

print(alpha, thresh)











#     # Convert each windowed dPLI matrix to a graph
#         windowed_graphs = [nx.convert_matrix.from_numpy_array(matrix, create_using=nx.DiGraph) for matrix in windowed_dpli_matrices]

#         # Perform aggregated bootstrapping and find optimal alpha and upper threshold
#         optimal_alpha, upper_threshold = aggregated_bootstrapping_and_threshold(windowed_graphs, num_iterations=1000, percentile=95)

#         # Apply the aggregated filter to each windowed graph
#         thresholded_dpli_matrices = []
#         for G_dPLI in windowed_graphs:
#             G_dPLI_thresholded = apply_aggregated_filter(G_dPLI, optimal_alpha, upper_threshold)
#             dpli_matrix_thresholded = nx.convert_matrix.to_numpy_array(G_dPLI_thresholded)
#             thresholded_dpli_matrices.append(dpli_matrix_thresholded)

#         # thresholded_dpli_matrices contains the thresholded dPLI matrices for each window
