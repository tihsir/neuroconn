# Import necessary Python modules
from sklearn.decomposition import PCA
from mne_icalabel import label_components
from mne.preprocessing import ICA
import copy  # This is a Python module that allows you to copy objects without changing the original object
from scipy import signal
import sklearn as sk
import matplotlib.pyplot as plt
import os
import matplotlib
import mne
import numpy as np
from scipy.stats import zscore
matplotlib.use('Agg')  # disable plotting
mne.viz.set_browser_backend('matplotlib', verbose=None)
mne.set_config('MNE_BROWSER_BACKEND', 'matplotlib')


# defining input and output directory
files_in = '../data/in/subjects/'
files_out = '../data/out/subjects/'


# loading list of subject names from txt file
names = open("./names.txt", "r")
subject_list = names.read().split('\n')
modes = ['EC', 'EO']
# Read the custom montage
montage_path = r"../data/in/MFPRL_UPDATED_V2.sfp"
montage = mne.channels.read_custom_montage(montage_path)

# Define the map of channel names using the provided keys
ch_map = {'Ch1': 'Fp1', 'Ch2': 'Fz', 'Ch3': 'F3', 'Ch4': 'F7', 'Ch5': 'LHEye', 'Ch6': 'FC5',
          # Setting FPz as GND so it matches montage
          'Ch7': 'FC1', 'Ch8': 'C3', 'Ch9': 'T7', 'Ch10': 'GND', 'Ch11': 'CP5', 'Ch12': 'CP1',
          'Ch13': 'Pz', 'Ch14': 'P3', 'Ch15': 'P7', 'Ch16': 'O1', 'Ch17': 'Oz', 'Ch18': 'O2',
          'Ch19': 'P4', 'Ch20': 'P8', 'Ch21': 'Rmastoid', 'Ch22': 'CP6', 'Ch23': 'CP2', 'Ch24': 'Cz',
          'Ch25': 'C4', 'Ch26': 'T8', 'Ch27': 'RHEye', 'Ch28': 'FC6', 'Ch29': 'FC2', 'Ch30': 'F4',
          'Ch31': 'F8', 'Ch32': 'Fp2', 'Ch33': 'AF7', 'Ch34': 'AF3', 'Ch35': 'AFz', 'Ch36': 'F1',
          'Ch37': 'F5', 'Ch38': 'FT7', 'Ch39': 'FC3', 'Ch40': 'FCz', 'Ch41': 'C1', 'Ch42': 'C5',
          'Ch43': 'TP7', 'Ch44': 'CP3', 'Ch45': 'P1', 'Ch46': 'P5', 'Ch47': 'Lneck', 'Ch48': 'PO3',
          'Ch49': 'POz', 'Ch50': 'PO4', 'Ch51': 'Rneck', 'Ch52': 'P6', 'Ch53': 'P2', 'Ch54': 'CPz',
          'Ch55': 'CP4', 'Ch56': 'TP8', 'Ch57': 'C6', 'Ch58': 'C2', 'Ch59': 'FC4', 'Ch60': 'FT8',
          'Ch61': 'F6', 'Ch62': 'F2', 'Ch63': 'AF4', 'Ch64': 'RVEye'}

for subject in subject_list:
    for mode in modes:

        print(subject, mode)
        # defining paths for current subject
        input_path = files_in+subject + '/' + mode + '/'
        output_path = files_out + subject + '/' + mode + '/'

        log_file = output_path+'log3.txt'
        log = open(log_file, "w")

        # Reading in label file
        labelf = open(output_path+'labels.txt', "r")
        labels = labelf.read().split('\n')
        probf = open(output_path+'probs.txt', "r")
        probabilities = probf.read().split('\n')

        # Exclude components based on label and probability
        exclude_idx = [idx for idx, (label, prob) in enumerate(zip(labels, probabilities))
                       if label not in ["brain", "other"] or prob < 0.70]
        log.write(f"Excluding these ICA components: {exclude_idx} \n")

        # Copy the original interpolated EEG data
        original_EEG = mne.io.read_raw_fif(
            output_path + subject + '_badchannels.fif', preload=True)
        reconst_EEG = original_EEG

        n_components = 0.99  # Choose number of ICA components based on PCA

        # redifining ICA
        ica = ICA(
            n_components=n_components,
            max_iter="auto",
            method="infomax",
            random_state=97,
            fit_params=dict(extended=True),
        )

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

        # Interpolate bad channels

        # Replace NaN or inf values in channel locations with zero
        new_chs = original_EEG.info['chs'].copy()
        for ch in new_chs:
            ch['loc'] = np.nan_to_num(
                ch['loc'], nan=0.0, posinf=0.0, neginf=0.0)

        new_info = mne.create_info(
            [ch['ch_name'] for ch in new_chs], original_EEG.info['sfreq'], ch_types='eeg')
        original_EEG = mne.io.RawArray(original_EEG.get_data(), new_info)
        original_EEG.set_montage(mne.channels.make_dig_montage(
            ch_pos={ch['ch_name']: ch['loc'][:3] for ch in new_chs}))
        # Set the bad channels back to the original list
        original_EEG.info['bads'] = EEG = mne.io.read_raw_fif(
            output_path + subject + '_maprenamed&nfiltered.fif', preload=True).info["bads"]

        # Pick types and interpolate bads
        original_EEG_data = original_EEG.copy().pick_types(
            meg=False, eeg=True, exclude=[])
        original_EEG_data_interp = original_EEG_data.copy().interpolate_bads(reset_bads=False)

        # Plot the data before and after interpolation
        for title, data in zip(["full orig.", "full interp."], [original_EEG_data, original_EEG_data_interp]):
            with mne.viz.use_browser_backend("matplotlib"):
                fig = data.plot(
                    butterfly=True, color="#00000022", bad_color="r")
            fig.subplots_adjust(top=0.9)
            fig.suptitle(title, size="xx-large", weight="bold")
        plt.savefig(output_path + 'interpolated.png')
        # Save the interpolated data
        original_EEG_data_interp.save(
            output_path + '_interpolated.fif', overwrite=True)

#################################################################################

        # EPOCHING

        # Define epoch parameters
        name = subject + '_eventchan'  # --> change for each condition
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
        name = subject + '_epoch'  # --> change the name of condition
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
        fig = epochs_all.plot()
        plt.savefig(output_path + 'epochs.png')
        # Save the filtered data
        # Replace with your desired output directory
        epochs_all.save(output_path + '_epoched.fif', overwrite=True)
        exit(R)
