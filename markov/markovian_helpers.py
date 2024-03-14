
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


