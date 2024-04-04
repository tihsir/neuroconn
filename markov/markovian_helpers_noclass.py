
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
from scipy.stats import multivariate_normal
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import networkx as nx
import seaborn as sns
from scipy.optimize import fminbound

from numba import njit, jit
from numba.experimental import jitclass

# Downsampling with Anti-Aliasing Filtering
@njit
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

@njit
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




###################### BREAKING DOWN THE VARIATIONAL HMM CLASS INTO INDIVIDUAL FUNCTIONS


@njit
def elbo(n_states, data, init_distrib, trans_distrib, emission_means, emission_covs, q):
    expected_log_likelihood = 0
    entropy_q = 0
    kl_divergence = 0

    for t in range(len(data)):
        for i in range(n_states):
            expected_log_likelihood += q[i, t] * _emission_logprob(data[t], emission_means, emission_covs, i)

    for t in range(len(data)):
        for i in range(n_states):
            entropy_q -= q[i, t] * np.log(q[i, t] + 1e-10)

    for t in range(len(data)):
        for i in range(n_states):
            kl_divergence += q[i, t] * np.log(np.maximum(q[i, t], 1e-10) / q[i, t])

    return expected_log_likelihood + entropy_q - kl_divergence


# Free energy = -ELBO (expected log-likelihood - Kl divergence)
@njit
def free_energy(n_states, data, init_distrib, trans_distrib, emission_means, emission_covs, q):
    return -elbo(n_states, data, init_distrib, trans_distrib, emission_means, emission_covs, q)


@njit
def normalize_logprobs(log_probs):
    max_log_prob = np.max(log_probs)
    return log_probs - max_log_prob - np.log(np.sum(np.exp(log_probs - max_log_prob)))


def fit(n_states, data, max_iter=100):
    init_distrib = np.ones(n_states) / n_states
    trans_distrib = np.ones((n_states, n_states)) / n_states
    emission_means = np.random.randn(n_states, data.shape[1])
    emission_covs = np.ones((n_states, data.shape[1]))
    q = None

    for _ in range(max_iter):
        print("loop")
        q = e_step(n_states, data, init_distrib, trans_distrib, emission_means, emission_covs, q)
        init_distrib, trans_distrib, emission_means, emission_covs = m_step(n_states, data, init_distrib, trans_distrib,
                                                                             emission_means, emission_covs, q)
        print(f"ELBO: {elbo(n_states, data, init_distrib, trans_distrib, emission_means, emission_covs, q)}")
    return q


@njit(forceobj = True)
def e_step(n_states, data, init_distrib, trans_distrib, emission_means, emission_covs, q):
    log_likelihoods = np.zeros((n_states, len(data)))

    for t in range(len(data)):
        for i in range(n_states):
            log_lik = _emission_logprob(data[t], emission_means, emission_covs, i)
            log_likelihoods[i, t] = log_lik + (np.log(init_distrib[i]) if t == 0
                                               else np.logaddexp.reduce(
                    log_likelihoods[:, t - 1] + np.log(trans_distrib[:, i])))

    log_betas = np.zeros((n_states, len(data)))
    log_betas[:, -1] = 0

    for t in list(range(len(data) - 1, 0, -1)):
        for i in range(n_states):
            for j in range(n_states):
                log_betas[i, t] = np.logaddexp(log_betas[i, t], log_betas[j, t + 1] + np.log(
                    trans_distrib[i, j]) + _emission_logprob(data[t + 1], emission_means, emission_covs, j))

    q = np.zeros((n_states, len(data)))
    for t in range(len(data)):
        q[:, t] = np.exp(log_likelihoods[:, t] + log_betas[:, t] - normalize_logprobs(
            log_likelihoods[:, t] + log_betas[:, t]))

    return q
@njit
def _emission_logprob(x, emission_means, emission_covs, state_idx):
    return multivariate_normal.logpdf(x, mean=emission_means[state_idx],cov=np.diag(emission_covs[state_idx]))

@njit
def m_step(n_states, data, init_distrib, trans_distrib, emission_means, emission_covs, q):
    expected_transitions = np.zeros((n_states, n_states))

    for t in range(len(data) - 1):
        for i in range(n_states):
            for j in range(n_states):
                expected_transitions[i, j] += q[i, t] * trans_distrib[i, j] * _emission_logprob(
                    data[t + 1], emission_means, emission_covs, j) * q[j, t + 1]
    trans_distrib = expected_transitions / expected_transitions.sum(axis=1, keepdims=True)

    for i in range(n_states):
        init_distrib[i] = q[i, 0]

    for i in range(n_states):
        emission_means[i] = np.sum([q[i, t] * data[t] for t in range(len(data))], axis=0) / np.sum(
            q[i])

    for i in range(n_states):
        diff = data - emission_means[i]
        emission_covs = np.array([np.eye(data.shape[1]) for _ in range(n_states)])

    return init_distrib, trans_distrib, emission_means, emission_covs




            
            ### OLD HMMM class





# @jitclass
class VariationalHMM:
    
    def __init__(self, n_states, data):
        self.n_states = n_states
        self.data = data

        # Initialize priors (adjust if you have prior knowledge)
        self.init_distrib = np.ones(n_states) / n_states
        self.trans_distrib = np.ones((n_states, n_states)) / n_states
        self.emission_means = np.random.randn(n_states, data.shape[1])
        self.emission_covs = np.ones((n_states, data.shape[1]))
    
    @njit
    def elbo(self, q):  # Evidence Lower Bound

        # Expected log-likelihood: This measures how well the model with its approximate posterior distribution q explains the observed data
        expected_log_likelihood = 0
        for t in range(len(self.data)):
            for i in range(self.n_states):
                expected_log_likelihood += q[i, t] * self._emission_logprob(self.data[t], i)

        # Entropy of the approximate posterior: Measures the uncertainty in our approximate posterior distribution q
        entropy_q = 0
        for t in range(len(self.data)):
            for i in range(self.n_states):
                entropy_q -= q[i, t] * np.log(q[i, t] + 1e-10)

        # KL Divergence (between approximate and true posterior): This term penalizes the divergence of our approximation q from the true, intractable posterior distribution over model parameters
        kl_divergence = 0
        for t in range(len(self.data)):
            for i in range(self.n_states):
                kl_divergence += q[i, t] * np.log(np.maximum(q[i, t], 1e-10) / q[i, t])  # Using q as an approximation

                return expected_log_likelihood + entropy_q - kl_divergence

    # Free energy = -ELBO (expected log-likelihood - Kl divergence)
    @njit
    def free_energy(self, q):
        return -self.elbo(q)
  
    @staticmethod
    @njit(forceobj=True, nopython=False)
    def fit(self, max_iter=100):
        q = None  # Initialize q to None
        for _ in range(max_iter):
            print("loop")
            # E-step: Update q(z) using current model parameters
            q = self._e_step()

            # M-step: Update model parameters using current q(z)
            self._m_step(q)

            print(f"ELBO: {self.elbo(q)}")  # Commented out to avoid cluttering the output
        return q  # Return the final q after fitting

    @njit
    def normalize_logprobs(log_probs):
        max_log_prob = np.max(log_probs)
        return log_probs - max_log_prob - np.log(np.sum(np.exp(log_probs - max_log_prob)))

    @staticmethod
    @njit(forceobj = True)
    def _e_step(self):
        log_likelihoods = np.zeros((self.n_states, len(self.data)))

        # Forward Pass (Calculate alphas in log domain)
        for t in range(len(self.data)):
            for i in range(self.n_states):
                log_lik = self._emission_logprob(self.data[t], i)  # Log of emission probability
                log_likelihoods[i, t] = log_lik + (np.log(self.init_distrib[i]) if t == 0
                                                   else np.logaddexp.reduce(
                    log_likelihoods[:, t - 1] + np.log(self.trans_distrib[:, i])))

        # Backward Pass (Calculate betas in log domain)
        log_betas = np.zeros((self.n_states, len(self.data)))
        log_betas[:, -1] = 0  # Initialize in log-domain

        for t in list(range(len(self.data) - 1, 0, -1)):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    log_betas[i, t] = np.logaddexp(log_betas[i, t], log_betas[j, t + 1] + np.log(
                        self.trans_distrib[i, j]) + self._emission_logprob(self.data[t + 1], j))


        # Store approximate posteriors for all time steps
        q = np.zeros((self.n_states, len(self.data)))
        for t in range(len(self.data)):
            q[:, t] = np.exp(log_likelihoods[:, t] + log_betas[:, t] - self.normalize_logprobs(
                log_likelihoods[:, t] + log_betas[:, t]))

        return q
    @njit
    def _emission_logprob(self, x, state_idx):
        # Calculation of log p(x|z) based on Gaussian emission
        return multivariate_normal.logpdf(x, mean=self.emission_means[state_idx],
                                          cov=np.diag(self.emission_covs[state_idx]))
    @njit
    def _m_step(self, q):
        # Update initial distribution
        self.init_distrib = q[:, 0]  # Posteriors at t=0

        # Update transition probabilities
        expected_transitions = np.zeros((self.n_states, self.n_states))
        for t in range(len(self.data) - 1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    expected_transitions[i, j] += q[i, t] * self.trans_distrib[i, j] * self._emission_logprob(
                        self.data[t + 1], j) * q[j, t + 1]
        self.trans_distrib = expected_transitions / expected_transitions.sum(axis=1, keepdims=True)

        # Update emission means
        for i in range(self.n_states):
            self.emission_means[i] = np.sum([q[i, t] * self.data[t] for t in range(len(self.data))], axis=0) / np.sum(
                q[i])

        # Update emission covariances
        for i in range(self.n_states):
            diff = self.data - self.emission_means[i]
            # Corrected line: Initialize emission covariances to identity matrices
            self.emission_covs = np.array([np.eye(self.data.shape[1]) for _ in range(self.n_states)])
