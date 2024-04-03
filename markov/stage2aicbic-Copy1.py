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
optimal_states = 7#int((optimal_states_aic + optimal_states_bic) / 2) # Optimal states often called K in the literature
print(f"Optimal number of states based on AIC/BIC: {optimal_states}")

# Save the optimal number of states to a file for later use
output_dir = r'C:\Users\cerna\Downloads'  # Replace with your desired output directory
optimal_states_file = os.path.join(output_dir, f"{subj}_optimal_states.npy")
np.save(optimal_states_file, optimal_states)