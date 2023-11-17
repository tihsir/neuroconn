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
        # defining paths for current subject
        input_path = files_in+subject + '/' + mode + '/'
        output_path = files_out + subject + '/' + mode + '/'

        print(input_path, output_path)

        log_file = output_path+'log2.txt'
        log = open(log_file, "w")

        # loading in files savef from 1filter.py
        EEG = mne.io.read_raw_fif(
            output_path + subject + '_maprenamed&nfiltered.fif', preload=True)

        # MARKING BAD CHANNELS

        # This can be used to plot the data with the bad channels marked.
        # Uncomment the two lines of code below to see the plot
        # Replace 'regexp=" ."' with the tentative bad channels
        picks = mne.pick_channels_regexp(EEG.ch_names, regexp="AF.|FT.")
        plot_obj = EEG.plot(order=picks, n_channels=len(picks))

        # Change list of bad channels
        original_bads = copy.deepcopy(EEG.info["bads"])
        EEG.info["bads"].append("AF7")  # add a single channel
        # add a single channel to the original_bads list
        original_bads.append("AF7")
        # EEG_csd.info["bads"].extend(["EEG 051", "EEG 052"])  # add a list of channels
        # original_bads["bads"].extend(["EEG 051", "EEG 052"])  # add a list of channels

        # Print the bad channels to double check
        log.write('Writing Bad Channels to Double check \n')
        log.write(str(EEG.info['bads']) + '\n')
        log.write(str(original_bads) + '\n \n \n')

        # Save the data with the bad channels marked
        # Replace with your desired output directory
        bad_channel_file = output_path + subject + '_badchannels.fif'
        EEG.save(bad_channel_file, overwrite=True)

        # ICA (Independent Component Analysis)

        # Keep a reference to the original, uncropped data
        original_EEG = EEG

        # Crop a copy of the data to three seconds for easier plotting
        cropped_EEG = EEG.copy().crop(tmin=0, tmax=3).load_data()

        # Fit average re-reference to the data
        original_EEG.set_eeg_reference('average')

        # Drop channels #10 and #21 (mastoids) before ICA
        original_EEG.drop_channels(['Rmastoid'])

        # Determine the number of PCA components
        data = original_EEG.get_data().T
        pca = PCA()
        pca.fit(data)

        # Plot the explained variance ratio
        plt.figure(figsize=(10, 5))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Explained Variance by PCA Components')
        plt.grid(True)
        plt.imsave(output_path + subject + '_PCA_variance.png')

        # Define ICA parameters
        n_components = 0.99  # Choose number of ICA components based on PCA
        ica = ICA(
            n_components=n_components,
            max_iter="auto",
            method="infomax",
            random_state=97,
            fit_params=dict(extended=True),
        )

        # Pick only EEG channels
        picks_eeg = mne.pick_types(original_EEG.info, meg=False,
                                   eeg=True, eog=False, stim=False, emg=False, exclude='bads')

        # Fit ICA using only EEG channels
        ica.fit(original_EEG, picks=picks_eeg, decim=3)

        # Plot the ICA components as time series
        ica_ts_plot = ica.plot_sources(
            original_EEG, show_scrollbars=False, show=True)
        # saving the timeseries plot
        ica_ts_plot.figsave(output_path + subject + 'ica_timeseries.png')

        # Plot the ICA components as topographies in multiple windows
        log.write("Plotting the ICA components as topographies... \n")
        n_components_actual = ica.n_components_

        # Selecting ICA components automatically using ICLabel
        ic_labels = label_components(original_EEG, ica, method="iclabel")
        component_labels = ic_labels["labels"]  # Extract the labels
        # Extract the probabilities
        component_probabilities = ic_labels["y_pred_proba"]
        for i in range(0, n_components_actual, 62):
            # Plot the components
            fig = ica.plot_components(picks=range(
                i, min(i + 62, n_components_actual)), ch_type='eeg', inst=original_EEG)
            # Set titles for each axis based on the labels and probabilities
            for ax, label, prob in zip(fig.axes, component_labels[i:min(i + 62, n_components_actual)],
                                       component_probabilities[i:min(i + 62, n_components_actual)]):
                # Displaying label and probability rounded to 2 decimal places
                ax.set_title(f"{label} ({prob:.2f})")

            fig.figsave(output_path + subject + 'ica_topo.png')
            # blinks
            fig_overlay = ica.plot_overlay(
                original_EEG, exclude=[0], picks="eeg")
            fig_overlay.savefig(output_path + subject + 'eeg_overlay.png')
        # ICLabel scores

        log.write("Initializing labels file \n")
        label_file = output_path+'labels.txt'
        labelf = open(label_file, "w")

        for idx, label in enumerate(component_labels):
            labelf.write(f"Component {idx}: {label} \n")

        log.write("Completed subject and mode successfully \n")
