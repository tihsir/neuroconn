
# Integrating a Hidden Markov Model for brain state indentification

# Step 1: Compute the orthogonalized envelope of the analytic signal

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

from markovian_helpers import downsample_with_filtering, apply_orthogonalization



# defining input and output directory
files_in = '../data/in/subjects/'
files_out = '../data/out/subjects/'


# loading list of subject names from txt file
names = open("./names.txt", "r")
subject_list = names.read().split('\n')
modes = ['EC', 'EO']
for subject in subject_list:
    for mode in modes:

        #defining input and output directories for each subject and mode
        dir_in = files_in + subject + '/' + mode + '/'
        dir_out = files_out + subject +"/" + mode +'/'

        label_time_courses_file = dir_out+ f"{subject}_label_time_courses.npy"
        label_time_courses = np.load(label_time_courses_file)

        # Original and target sampling frequencies
        original_sampling_freq = 513
        target_sampling_freq = 250

        # Calculate the downsampling factor
        downsampling_factor = int(np.floor(original_sampling_freq / target_sampling_freq))
        downsampled_label_time_courses = downsample_with_filtering(label_time_courses, original_sampling_freq, target_sampling_freq)


        # Verify the new shape
        print(downsampled_label_time_courses.shape)  # Expected shape: (num_labels, num_channels, new_num_samples)

        # Load labels from the atlas
        labels = ['7Networks_LH_Cont_Cing_1-lh', '7Networks_LH_Cont_PFCl_1-lh', '7Networks_LH_Cont_Par_1-lh', '7Networks_LH_Cont_pCun_1-lh', '7Networks_LH_Default_PFC_1-lh', '7Networks_LH_Default_PFC_2-lh', '7Networks_LH_Default_PFC_3-lh', '7Networks_LH_Default_PFC_4-lh', '7Networks_LH_Default_PFC_5-lh', '7Networks_LH_Default_PFC_6-lh', '7Networks_LH_Default_PFC_7-lh', '7Networks_LH_Default_Par_1-lh', '7Networks_LH_Default_Par_2-lh', '7Networks_LH_Default_Temp_1-lh', '7Networks_LH_Default_Temp_2-lh', '7Networks_LH_Default_pCunPCC_1-lh', '7Networks_LH_Default_pCunPCC_2-lh', '7Networks_LH_DorsAttn_FEF_1-lh', '7Networks_LH_DorsAttn_Post_1-lh', '7Networks_LH_DorsAttn_Post_2-lh', '7Networks_LH_DorsAttn_Post_3-lh', '7Networks_LH_DorsAttn_Post_4-lh', '7Networks_LH_DorsAttn_Post_5-lh', '7Networks_LH_DorsAttn_Post_6-lh', '7Networks_LH_DorsAttn_PrCv_1-lh', '7Networks_LH_Limbic_OFC_1-lh', '7Networks_LH_Limbic_TempPole_1-lh', '7Networks_LH_Limbic_TempPole_2-lh', '7Networks_LH_SalVentAttn_FrOperIns_1-lh', '7Networks_LH_SalVentAttn_FrOperIns_2-lh', '7Networks_LH_SalVentAttn_Med_1-lh', '7Networks_LH_SalVentAttn_Med_2-lh', '7Networks_LH_SalVentAttn_Med_3-lh', '7Networks_LH_SalVentAttn_PFCl_1-lh', '7Networks_LH_SalVentAttn_ParOper_1-lh', '7Networks_LH_SomMot_1-lh', '7Networks_LH_SomMot_2-lh', '7Networks_LH_SomMot_3-lh', '7Networks_LH_SomMot_4-lh', '7Networks_LH_SomMot_5-lh', '7Networks_LH_SomMot_6-lh', '7Networks_LH_Vis_1-lh', '7Networks_LH_Vis_2-lh', '7Networks_LH_Vis_3-lh', '7Networks_LH_Vis_4-lh', '7Networks_LH_Vis_5-lh', '7Networks_LH_Vis_6-lh', '7Networks_LH_Vis_7-lh', '7Networks_LH_Vis_8-lh', '7Networks_LH_Vis_9-lh', '7Networks_RH_Cont_Cing_1-rh', '7Networks_RH_Cont_PFCl_1-rh', '7Networks_RH_Cont_PFCl_2-rh', '7Networks_RH_Cont_PFCl_3-rh', '7Networks_RH_Cont_PFCl_4-rh', '7Networks_RH_Cont_PFCmp_1-rh', '7Networks_RH_Cont_Par_1-rh', '7Networks_RH_Cont_Par_2-rh', '7Networks_RH_Cont_pCun_1-rh', '7Networks_RH_Default_PFCdPFCm_1-rh', '7Networks_RH_Default_PFCdPFCm_2-rh', '7Networks_RH_Default_PFCdPFCm_3-rh', '7Networks_RH_Default_PFCv_1-rh', '7Networks_RH_Default_PFCv_2-rh', '7Networks_RH_Default_Par_1-rh', '7Networks_RH_Default_Temp_1-rh', '7Networks_RH_Default_Temp_2-rh', '7Networks_RH_Default_Temp_3-rh', '7Networks_RH_Default_pCunPCC_1-rh', '7Networks_RH_Default_pCunPCC_2-rh', '7Networks_RH_DorsAttn_FEF_1-rh', '7Networks_RH_DorsAttn_Post_1-rh', '7Networks_RH_DorsAttn_Post_2-rh', '7Networks_RH_DorsAttn_Post_3-rh', '7Networks_RH_DorsAttn_Post_4-rh', '7Networks_RH_DorsAttn_Post_5-rh', '7Networks_RH_DorsAttn_PrCv_1-rh', '7Networks_RH_Limbic_OFC_1-rh', '7Networks_RH_Limbic_TempPole_1-rh', '7Networks_RH_SalVentAttn_FrOperIns_1-rh', '7Networks_RH_SalVentAttn_Med_1-rh', '7Networks_RH_SalVentAttn_Med_2-rh', '7Networks_RH_SalVentAttn_TempOccPar_1-rh', '7Networks_RH_SalVentAttn_TempOccPar_2-rh', '7Networks_RH_SomMot_1-rh', '7Networks_RH_SomMot_2-rh', '7Networks_RH_SomMot_3-rh', '7Networks_RH_SomMot_4-rh', '7Networks_RH_SomMot_5-rh', '7Networks_RH_SomMot_6-rh', '7Networks_RH_SomMot_7-rh', '7Networks_RH_SomMot_8-rh', '7Networks_RH_Vis_1-rh', '7Networks_RH_Vis_2-rh', '7Networks_RH_Vis_3-rh', '7Networks_RH_Vis_4-rh', '7Networks_RH_Vis_5-rh', '7Networks_RH_Vis_6-rh', '7Networks_RH_Vis_7-rh', '7Networks_RH_Vis_8-rh', 'Background+FreeSurfer_Defined_Medial_Wall-lh', 'Background+FreeSurfer_Defined_Medial_Wall-rh']
        labels = np.array(labels)

        # Orthogonalization
        orthogonalized_data = apply_orthogonalization(downsampled_label_time_courses)
        np.save( dir_out + "orth.npy", orthogonalized_data)


