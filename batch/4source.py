# Computing and applying a linear minimum-norm inverse method on evoked/raw/epochs data

# Import necessary libraries and functions
from scipy.stats import zscore
import sklearn as sk
from scipy import signal
import copy  # This is a Python module that allows you to copy objects without changing the original object
from mne.preprocessing import ICA
from mne_icalabel import label_components
from sklearn.decomposition import PCA
from mne.minimum_norm import apply_inverse_epochs
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.datasets import sample, eegbci, fetch_fsaverage
import mne
import matplotlib.pyplot as plt
import os
import os.path as op
import numpy as np
import matplotlib
# matplotlib.use('Qt5Agg')  # Setting the backend BEFORE importing pyplot

# mne.viz.set_3d_backend("pyvista")


#################################################################################
# Adult template MRI (fsaverage)

# Download fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)

# The files live in:
subject = "fsaverage"
trans = "fsaverage"  # MNE has a built-in fsaverage transformation
src = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
bem = op.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")


# Import necessary Python modules
matplotlib.use('Agg')  # disable plotting
mne.viz.set_browser_backend('matplotlib', verbose=None)
mne.set_config('MNE_BROWSER_BACKEND', 'matplotlib')


# defining input and output directory
files_in = '../data/in/subjects/'
files_out = '../data/out/subjects/'


# loading list of subject names from txt file
names = open("./names.txt", "r")
subject_list = names.read().split('\n')

subject_list = subject_list[-15:]

print(subject_list)
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

def process_subject_c(subject, modes=['EC','EO'], files_in='../data/in/subjects/', files_out='../data/out/subjects/', montage, ch_map, trans, src, bem):
    for mode in modes:
        print(subject, mode)
        # defining paths for current subject
        input_path = files_in+subject + '/' + mode + '/'
        output_path = files_out + subject + '/' + mode + '/'

        log_file = output_path+'log4.txt'
        log = open(log_file, "w")

        epochs = mne.read_epochs(output_path + subject+'_epoched.fif')

        # List of channels to drop
        channels_to_drop = ['LHEye', 'RHEye', 'Lneck', 'Rneck', 'RVEye', 'FPz']

        # Drop the channels from the epochs data
        try:
            epochs.drop_channels(channels_to_drop)
        except:
            pass

        # Adjust EEG electrode locations to match the fsaverage template, which are already in fsaverage's
        # # space (MNI space) for standard_1020
        # montage_path = r"../data/in/MFPRL_UPDATED_V2.sfp"
        # montage = mne.channels.read_custom_montage(montage_path)
        epochs.set_montage(montage)
        # needed for inverse modeling
        epochs.set_eeg_reference(projection=True)

        # Compute the forward solution using the fsaverage template
        fwd = mne.make_forward_solution(
            epochs.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=None
        )

        # Adjusting picks to EEG data
        picks = mne.pick_types(epochs.info, meg=False,
                               eeg=True, eog=True, stim=False)

        # Compute regularized noise covariance
        noise_cov = mne.compute_covariance(
            epochs, tmax=0.0, method=["shrunk", "empirical"], rank=None, verbose=True
        )

        fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, epochs.info)

        #################################################################################
        # Visualize the source space on the cortex

        # Read the forward solution
        mne.convert_forward_solution(fwd, surf_ori=True, copy=False)

        # Extract the source space information from the forward solution
        lh = fwd["src"][0]  # Visualize the left hemisphere
        verts = lh["rr"]  # The vertices of the source space
        tris = lh["tris"]  # Groups of three vertices that form triangles
        dip_pos = lh["rr"][lh["vertno"]]  # The position of the dipoles
        dip_ori = lh["nn"][lh["vertno"]]
        dip_len = len(dip_pos)
        dip_times = [0]

        # Create a Dipole instance
        actual_amp = np.ones(dip_len)  # misc amp to create Dipole instance
        actual_gof = np.ones(dip_len)  # misc GOF to create Dipole instance
        dipoles = mne.Dipole(dip_times, dip_pos,
                             actual_amp, dip_ori, actual_gof)
        trans = trans

        mne.write_forward_solution(
            output_path + '_forwardsolution_MRItemplate.fif', fwd, overwrite=True)
        #################################################################################
        # Inverse modeling: eLORETA on evoked data with dipole orientation discarded (pick_ori="None"), only magnitude kept

        # Create a loose-orientation inverse operator, with depth weighting
        inv = make_inverse_operator(
            epochs.info, fwd, noise_cov, fixed=False, loose=0.2, depth=0.8, verbose=True)

        # Compute eLORETA solution for each epoch
        snr = 3.0
        lambda2 = 1.0 / snr**2
        # pick_ori="None" --> Discard dipole orientation, only keep magnitude
        stcs = apply_inverse_epochs(
            epochs, inv, lambda2, "eLORETA", verbose=True, pick_ori=None)

        # Average the source estimates across epochs
        stc_avg = sum(stcs) / len(stcs)

        # Get the time of the peak magnitude
        _, time_max = stc_avg.get_peak(hemi="lh")

        # Visualization parameters
        kwargs = dict(
            hemi="lh",
            subjects_dir=subjects_dir,
            size=(600, 600),
            clim=dict(kind="percent", lims=[90, 95, 99]),
            smoothing_steps=7,
            time_unit="s",
            initial_time=time_max  # Set the initial_time to the time of the peak magnitude
        )

        # Visualizing the averaged source estimate with dipole magnitude
        # brain_magnitude = stc_avg.plot(**kwargs)
        # mne.viz.set_3d_view(figure=brain_magnitude, focalpoint=(0.0, 0.0, 50))

        # Average the data across all source space points
        avg_data = stc_avg.data.mean(axis=(0, 1))

        # Plot the average data as a function of time
        fig, ax = plt.subplots()
        print(stc_avg.times, avg_data)
        # ax.plot(1e3 * stc_avg.times, avg_data)
        # ax.set(xlabel="time (ms)", ylabel="eLORETA value (average)")
        # plt.show()

        print('Saving file')
        # Save the inverse solution data
        output_dir = output_path + 'stc/'  # Replace with your desired output directory
        for idx, stc in enumerate(stcs):
            inverse_solution_file = output_dir + f"{subject}_inversesolution_epoch{idx}.fif"
            print(inverse_solution_file)
            stc.save(inverse_solution_file, overwrite = True)