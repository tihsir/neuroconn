# Source-to-parcel analysis

# Import necessary libraries
from matplotlib.animation import FuncAnimation
# import seaborn as sns  # required for heatmap visualization
import networkx as nx
from scipy.stats import pearsonr
from mne.viz import circular_layout
import pandas as pd
from mne_connectivity.viz import plot_connectivity_circle
import matplotlib.pyplot as plt
import os
import glob
import numpy as npc
import cupy as np #using gpu acceleration
import cupyx.scipy.fft
import mne
from mne.datasets import fetch_fsaverage
from nilearn import datasets
from nilearn.image import get_data
from scipy.signal import hilbert #scipy core modified in env, running custom lib
import scipy
import matplotlib
import os.path as op

matplotlib.use('Agg')  # Setting the backend BEFORE importing pyplot



scipy.fft.set_backend(cupyx.scipy.fft)

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
modes = ['EC', 'EO']
# Read the custom montage
montage_path = r"../data/in/MFPRL_UPDATED_V2.sfp"
montage = mne.channels.read_custom_montage(montage_path)



schaefer_atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100)
fs_dir = '../data/in/fsaverage'
fname = os.path.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
src = mne.read_source_spaces(fname, patch_stats=False, verbose=None)

def process_subject_d(subject, modes=['EC','EO'], files_in='../data/in/subjects/', files_out='../data/out/subjects/'):
    for mode in modes:
        print(subject, mode)
        # defining paths for current subject
        input_path = files_in+subject + '/' + mode + '/'
        output_path = files_out + subject + '/' + mode + '/'

        stc_path  = output_path +'stc/'

        inverse_solution_files_lh = []
        inverse_solution_files_rh = []
        
        
        for path, subdirs, files in os.walk(stc_path):
            for file in files:
                filepath = path + file
                if '-rh.stc' in file:
                    inverse_solution_files_rh.append(filepath)
                elif '-lh.stc' in file:
                    inverse_solution_files_lh.append(filepath)

        # Error here !!!

        # Calculate the total number of inverse solution files for both hemispheres
        total_files_lh = len(inverse_solution_files_lh)
        total_files_rh = len(inverse_solution_files_rh)

        # Calculate the batch size for both hemispheres
        # Change 10 to your desired batch size
        batch_size_lh = total_files_lh // (total_files_lh // 10)

        # Change 10 to your desired batch size
        batch_size_rh = total_files_rh // (total_files_rh // 10)

        # Ensure batch size is a multiple of 10 (or your desired batch size) for both hemispheres
        while total_files_lh % batch_size_lh != 0:
            batch_size_lh -= 1

        while total_files_rh % batch_size_rh != 0:
            batch_size_rh -= 1

        # Initialize lists to store source estimates for both hemispheres
        stcs_lh = []
        stcs_rh = []

        # Load data in batches for both hemispheres
        for i in range(0, total_files_lh, batch_size_lh):
            batch_files_lh = inverse_solution_files_lh[i:i + batch_size_lh]
            batch_files_rh = inverse_solution_files_rh[i:i + batch_size_rh]

            for file_path_lh, file_path_rh in zip(batch_files_lh, batch_files_rh):
                try:
                    
                    stc_epoch_lh = mne.read_source_estimate(file_path_lh)
                    stc_epoch_rh = mne.read_source_estimate(file_path_rh)
                    stcs_lh.append(stc_epoch_lh)
                    stcs_rh.append(stc_epoch_rh)
                except Exception as e:
                    print(f"Error loading files {file_path_lh} or {file_path_rh}: {e}")

        # Load labels from the atlas
        labels = mne.read_labels_from_annot('fsaverage', parc='Schaefer2018_100Parcels_7Networks_order',
                                            subjects_dir=r'../data/in/')

        # Extract label time courses for both hemispheres
        label_time_courses = []  # Initialize a list to store label time courses
        
        
        for idx, (stc_lh, stc_rh) in enumerate(zip(stcs_lh, stcs_rh)):
            try:
                label_tc_lh = stc_lh.extract_label_time_course(
                    labels, src=src, mode='mean_flip')
                label_tc_rh = stc_rh.extract_label_time_course(
                    labels, src=src, mode='mean_flip')
                label_time_courses.extend([label_tc_lh, label_tc_rh])
                
            except Exception as e:
                print(f"Error extracting label time courses for iteration {idx}: {e}")
        else:  # This block will execute if the for loop completes without encountering a break statement
            print("All time courses have been successfully extracted!")

        # Convert label_time_courses to a NumPy array
        label_time_courses_np = np.array(label_time_courses)

        # If you prefer to save as a .csv file
        # Convert to DataFrame and save as .csv
        # label_time_courses_df = pd.DataFrame(label_time_courses_np)
        # label_time_courses_df.to_csv(os.path.join(output_dir, f"{subj}_label_time_courses.csv"), index=False)

        # Save the label time courses as a .npy file
        # Replace with your desired output directory
        label_time_courses_file = output_path + f"{subject}_label_time_courses.npy"
        
        np.save(label_time_courses_file, label_time_courses_np)
        continue
