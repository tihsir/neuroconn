#imports
# Import necessary Python modules
import os, sys
from env import SUBJ_R_DIR, SUBJ_W_DIR, SUBJ_LIST

#import sys args
#run flags can be as follows
# - load
# - filter
# - badchannel
# - ica
# - timecourse
# Each mode can has a suffix -x where x can be r/w, indicating 2 types of operations
# r implies read from existing file and proceed
# w implies overwrite existing files

# example run script
# python3 preprocessing.py load-w, filter-w, badchannel-r
#the flags not provided will be skipped

#read/write/off
lmode, fmode, bmode,imode, tmode = 'off', 'off', 'off', 'off', 'off'



for arg in sys.argv[1:]:
    if 'load' in arg[:-2]:
        lmode = 'read' if arg[-1] == 'r' else 'write'
    elif 'filter'in arg[:-2]:
        fmode = 'read' if arg[-1] == 'r' else 'write'
    elif 'badchannel'in arg[:-2]:
        bmode = 'read' if arg[-1] == 'r' else 'write'
    elif 'ica'in arg[:-2]:
        imode  = 'read' if arg[-1] == 'r' else 'write'
    elif 'timecourse'in arg[:-2]:
        tmode = 'read' if arg[-1] == 'r' else 'write'
    else:
        print('flag error')
        exit()
        
print(lmode, fmode, bmode, imode, tmode)

# defining input and output directory
files_in = SUBJ_R_DIR
files_out = SUBJ_W_DIR


# loading list of subject names from txt file
subjects = [x.strip() for x in SUBJ_LIST.split(',')]
modes = ["EC", "EO"]

for subject in subjects:
    for mode in modes:
        print (subject, mode, 'Beginning Now')
        
        if lmode !='off':
            pass
        if fmode !='off':
            pass
        if bmode != 'off':
            pass
        if imode !='off':
            
            EEG = mne.io.read_raw_fif(
                output_path + subject + '_maprenamed&nfiltered.fif', preload=True)

            # MARKING BAD CHANNELS

            # TODO - get bad channels from .txt files here
            print(EEG.ch_names)
            bad_txt = bad_list[i_s][j_m]  # get list of bad channels

            picks = mne.pick_channels(EEG.ch_names, include=[], exclude=[])
            original_bads = copy.deepcopy(EEG.info["bads"])

            # This can be used to plot the data with the bad channels marked.
            # Uncomment the two lines of code below to see the plot
            # Replace 'regexp=" ."' with the tentative bad channels

            if bad_txt == [''] or len(bad_txt) == 0:
                pass
            else:
                picks = mne.pick_channels(
                    EEG.ch_names, include=[], exclude=bad_txt)
                plot_obj = EEG.plot(order=picks, n_channels=len(picks))
                
            # Change list of bad channels
                original_bads = copy.deepcopy(EEG.info["bads"])
                for bad in bad_txt:
                    EEG.info["bads"].append(bad)  # add a single channel
                    # add a single channel to the original_bads list
                    original_bads.append(bad)
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
            

            # Plot the ICA components as topographies in multiple windows
            log.write("Plotting the ICA components as topographies... \n")
            n_components_actual = ica.n_components_

            # Selecting ICA components automatically using ICLabel
            ic_labels = label_components(
                original_EEG, ica, method='iclabel')
            component_labels = ic_labels["labels"]  # Extract the labels
            # Extract the probabilities
            component_probabilities = ic_labels["y_pred_proba"]


            # ICLabel scores
            EEG_list.append(original_EEG)
            print("Initializing labels file \n")
            label_file = output_path+'labels.txt'
            labelf = open(label_file, "r")
            prob_file = output_path+'probs.txt'
            probf = open(prob_file, "r")

            labels = labelf.read().split('\n')
            

            # Exclude components based on label and probability
            exclude_idx = [idx for idx, (label, prob) in enumerate(zip(labels, component_probabilities))
                if label.strip() not in ["brain", "other"] or prob < 0.70]
            
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
            preprocessed_file = output_path + subject+'_ICA.fif'
            reconst_EEG.save(preprocessed_file, overwrite=True)

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
            preprocessed_file = output_path+ subject + '_interpolated.fif'
            original_EEG_data_interp.save(preprocessed_file, overwrite=True)

            
        if tmode != 'off':
            pass