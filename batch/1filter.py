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

        log_file = output_path+'log1.txt'
        bad_channel_file = output_path+'bad_channel.txt'
        bad_channel = open(bad_channel_file, "w")
        log = open(log_file, "w")

        # debug line
        log.write("Reading in .vhdr at " + input_path + 'TCOA_' +
                  subject + '_'+mode+'.vhdr' + "\n \n \n \n")
        # loading in VHDR file

        EEG = mne.io.read_raw_brainvision(
            input_path + 'TCOA_' +
            subject + '_'+mode+'.vhdr', preload=True)
        print('Success')
        # except:  # skip if error
        #     log.write("ERROR Reading in .vhdr at " +
        #               input_path + 'TCOA_' +
        #               subject + '_'+mode+'.vhdr' + "\n")
        #     continue

        # drop channels
        print(EEG.ch_names)

        if len(EEG.ch_names) > 64:
            index = EEG.ch_names.index('Ch64')
            channels_to_drop = []
            try:
                channels_to_drop = EEG.ch_names[index+1:]
            except:
                pass
            EEG.drop_channels(channels_to_drop)

        print(EEG.ch_names)
        matplotlib.use('Agg')  # disable plotting
        raw_plot = EEG.plot(n_channels=len(EEG.ch_names),
                            scalings='auto', show=False)
        raw_file = output_path + subject + '_raw.fif'
        EEG.save(raw_file, overwrite=True)

        # write channel info to the log file
        log.write(str(EEG.info))
        log.write("\n \n \n \n")

        # Rename the channels using the new ch_map

        try:
            EEG.rename_channels(ch_map)
        except:
            print('SOMETHING IS GOING WRONG FOR', subject, mode)
            print(EEG.ch_names)
            exit()
        # Now the channels should match the names in the montage
        EEG.set_montage(montage, on_missing='warn')

        # Create a dictionary for channel types
        channel_types = {}

        # Set all channels to 'eeg' by default
        for ch in ch_map.values():
            channel_types[ch] = 'eeg'

        # Update the dictionary with the special channel types
        channel_types['RVEye'] = 'eog'
        channel_types['LHEye'] = 'eog'
        channel_types['RHEye'] = 'eog'
        channel_types['Rneck'] = 'emg'
        channel_types['Lneck'] = 'emg'
        channel_types['Rmastoid'] = 'misc'

        # Retrieve the locations of FP1 and FP2
        fp1_loc = EEG.info['chs'][EEG.ch_names.index('Fp1')]['loc'][:3]
        fp2_loc = EEG.info['chs'][EEG.ch_names.index('Fp2')]['loc'][:3]

        # Compute the average location for FPz
        fpz_loc = (fp1_loc + fp2_loc) / 2

        # Update the location of FPz in the original_EEG object
        EEG.info['chs'][EEG.ch_names.index('GND')]['loc'][:3] = fpz_loc

        # Print the updated location of FPz to verify
        log.write("Updated location of FPz:" + str(fpz_loc) + "\n")

        # Set the channel types in the EEG data
        EEG.set_channel_types(channel_types)

        # Apply a low-pass filter with a cutoff of 50 Hz
        EEG.filter(l_freq=None, h_freq=50)

        # Apply a high-pass filter with a cutoff of 1 Hz
        EEG.filter(l_freq=1, h_freq=None)

        # Add a notch filter from 60 Hz
        # This will create an array [60, 120, 180, 240] to capture the harmonics
        freqs = np.arange(60, 241, 60)
        EEG.notch_filter(freqs)

        # Plot the data to visualize waveforms after filtering
        filtered_plot = EEG.plot(n_channels=len(
            EEG.ch_names), scalings='auto', show=False)

        filtered_plot.set_size_inches(10, 10)
        plt.savefig(output_path+'EEG_filtered.png', dpi=400)

        bad_channel.write('')

        # Plotting EEG signal via PSD to check if the notch filter removed the power line noise
        psd_plot = EEG.plot_psd()
        plt.savefig(output_path+'psd.png')
        # Save the filtered data
        # Replace with your desired output directory
        preprocessed_file = output_path + subject + '_maprenamed&nfiltered.fif'
        EEG.save(preprocessed_file, overwrite=True)
        # commit
