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

from markovian_helpers import downsample_with_filtering, apply_orthogonalization, VariationalHMM

optimal_states_arr = []

# defining input and output directory
files_in = '../data/in/subjects/'
files_out = '../data/out/subjects/'


# loading list of subject names from txt file
names = open("./names.txt", "r")
subject_list = names.read().split('\n')
modes = ['EC', 'EO']
for subject in subject_list:
    for mode in modes:

        #defining input and output directories for each subject and mode
        dir_in = files_in + subject + '/' + mode + '/'
        dir_out = files_out + subject +"/" + mode +'/'

         orthogonalized_data = np.load( dir_out + "orth.npy")


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
        pca = PCA(n_components=0.99)  # Retain 99% of the variance

        # Fit PCA to the normalized data
        pca_data = pca.fit_transform(reshaped_data)

        # Standardize the PCA-transformed data
        scaler = StandardScaler()
        pca_data = scaler.fit_transform(pca_data)

        # Define the range of hidden states to explore
        state_numbers = range(3, 16)


        # Initialize lists to store models and Free Energies
        elbos = []
        models = []
        free_energies = []

        # Loop through different numbers of hidden states
        for n in state_numbers:
            # Create a Variational HMM model
            model = VariationalHMM(n, pca_data)

            # Fit the model and return q (posterior distribution) after fitting
            q = model.fit()

            # Calculate ELBOs (Evidence Lower Bound) and store them: Helps to analyze convergence if needed
            elbo = model.elbo(q)
            elbos.append(elbo)

            # Calculate and store Free Energy: -ELBO
            free_energy = model.free_energy(q)  # Assuming 'q' is your final posterior
            free_energies.append(free_energy)
            models.append(model)

        # Plotting
        plt.bar(state_numbers, free_energies)
        plt.xlabel("Number of States")
        plt.ylabel("Free Energy")
        plt.show()

        # Find the optimal number of states
        optimal_states = state_numbers[np.argmin(free_energies)]
        print(f"Optimal number of states based on Varitional Bayes for Subject {subject}: {optimal_states}")
        optimal_states_arr.append({subject:optimal_states})

