# Modify labels based on user input
        while True:
            modify = input(
                "\nDo you want to modify any label? (yes/no): ").strip().lower()
            if modify == 'yes':
                component_nums = input(
                    "Enter the component numbers you want to modify (comma-separated): ").split(',')
                new_labels = input(
                    "Enter the new labels for these components (comma-separated): ").split(',')

                for comp_num, new_label in zip(component_nums, new_labels):
                    component_labels[int(comp_num.strip())] = new_label.strip()
            else:
                break

        log.write("Final labels: \n")
        for idx, label in enumerate(component_labels):
            log.write(f"Component {idx}: {label} \n")

        # Extract labels and reconstruct raw data
        labels = ic_labels["labels"]
        probabilities = ic_labels["y_pred_proba"]

        # Exclude components based on label and probability
        exclude_idx = [idx for idx, (label, prob) in enumerate(zip(labels, probabilities))
                       if label not in ["brain", "other"] or prob < 0.70]
        log.write(f"Excluding these ICA components: {exclude_idx} \n")

        # Copy the original interpolated EEG data
        reconst_EEG = original_EEG.copy()

        # Apply the ICA transformation, excluding certain components
        ica.apply(reconst_EEG, exclude=exclude_idx)

        # Plot the original data and set the window title
        fig = original_EEG.plot(show_scrollbars=False)

        # Plot the reconstructed data and set the window title
        fig = reconst_EEG.plot(show_scrollbars=False)

        # Save the preprocessed data
        output_dir = r'../data/out'
        ICA_file = output_path + subject + '_ICA.fif'
        reconst_EEG.save(ICA_file, overwrite=True)

#################################################################################

# Interpolate bad channels

# Replace NaN or inf values in channel locations with zero
new_chs = original_EEG.info['chs'].copy()
for ch in new_chs:
    ch['loc'] = np.nan_to_num(ch['loc'], nan=0.0, posinf=0.0, neginf=0.0)

new_info = mne.create_info(
    [ch['ch_name'] for ch in new_chs], original_EEG.info['sfreq'], ch_types='eeg')
original_EEG = mne.io.RawArray(original_EEG.get_data(), new_info)
original_EEG.set_montage(mne.channels.make_dig_montage(
    ch_pos={ch['ch_name']: ch['loc'][:3] for ch in new_chs}))
# Set the bad channels back to the original list
original_EEG.info['bads'] = original_bads

# Repeat for cropped_EEG
new_chs = cropped_EEG.info['chs'].copy()
for ch in new_chs:
    ch['loc'] = np.nan_to_num(ch['loc'], nan=0.0, posinf=0.0, neginf=0.0)

new_info = mne.create_info(
    [ch['ch_name'] for ch in new_chs], cropped_EEG.info['sfreq'], ch_types='eeg')
cropped_EEG = mne.io.RawArray(cropped_EEG.get_data(), new_info)
cropped_EEG.set_montage(mne.channels.make_dig_montage(
    ch_pos={ch['ch_name']: ch['loc'][:3] for ch in new_chs}))
# Set the bad channels back to the original list
cropped_EEG.info['bads'] = original_bads

# Pick types and interpolate bads
original_EEG_data = original_EEG.copy().pick_types(
    meg=False, eeg=True, exclude=[])
original_EEG_data_interp = original_EEG_data.copy().interpolate_bads(reset_bads=False)

cropped_EEG_data = cropped_EEG.copy().pick_types(meg=False, eeg=True, exclude=[])
cropped_EEG_data_interp = cropped_EEG_data.copy().interpolate_bads(reset_bads=False)

# Plot the data before and after interpolation
for title, data in zip(["cropped orig.", "cropped interp."], [cropped_EEG_data, cropped_EEG_data_interp]):
    with mne.viz.use_browser_backend("matplotlib"):
        fig = data.plot(butterfly=True, color="#00000022", bad_color="r")
    fig.subplots_adjust(top=0.9)
    fig.suptitle(title, size="xx-large", weight="bold")

# Save the interpolated data
output_dir = r'../data/out'
preprocessed_file = os.path.join(output_dir, subj + '_interpolated.fif')
original_EEG_data_interp.save(preprocessed_file, overwrite=True)

#################################################################################

# EPOCHING

# Define epoch parameters
name = subj + '_eventchan'  # --> change for each condition
# Latency rate/Sampling rate
epoch_no = np.floor(reconst_EEG.get_data(
).shape[1] / reconst_EEG.info['sfreq'])

# Create a list of onset times for your events
onsets = np.arange(0, reconst_EEG.get_data(
).shape[1] / reconst_EEG.info['sfreq'], 1)

# Create a list of event durations (all zeros if the events are instantaneous)
durations = np.zeros_like(onsets)

# Create a list of event descriptions
descriptions = ['Event'] * len(onsets)

# Create an Annotations object
annotations = mne.Annotations(onsets, durations, descriptions)

# Add the annotations to the Raw object
reconst_EEG.set_annotations(annotations)

# Now you can extract the events from the annotations
events, event_id = mne.events_from_annotations(reconst_EEG)

# Define epoching parameters
name = subj + '_epoch'  # --> change the name of condition
codes = ['1']
tmin = -0.5  # Start of the epoch (in seconds)
tmax = 0.5  # End of the epoch (in seconds)

# Create epochs without rejection to keep all data
epochs_all = mne.Epochs(reconst_EEG, events, event_id=event_id,
                        tmin=tmin, tmax=tmax, proj=True, baseline=None, preload=True)

# Apply z-score normalization and keep track of which epochs exceed the threshold
zscore_threshold = 6
to_drop = []

temp_data = np.zeros_like(epochs_all._data)

for i in range(len(epochs_all)):
    temp_data[i] = zscore(epochs_all._data[i], axis=1)
    if np.any(np.abs(temp_data[i]) > zscore_threshold):
        to_drop.append(i)

# Now we can drop the epochs that exceeded the threshold
epochs_all.drop(to_drop)

# Resample and decimate the epochs
current_sfreq = epochs_all.info['sfreq']
# Hz chaging this according to  https://doi.org/10.1046/j.1440-1819.2000.00729.x
desired_sfreq = 512

# Apply the resampling
epochs_all.resample(desired_sfreq, npad='auto')

# Get the data from all epochs
data_all = epochs_all.get_data()

# Plot the data
epochs_all.plot()

# Save the filtered data
# Replace with your desired output directory
output_dir = r'../data/out'
preprocessed_file = os.path.join(output_dir, subj + '_epoched.fif')
epochs_all.save(preprocessed_file, overwrite=True)
