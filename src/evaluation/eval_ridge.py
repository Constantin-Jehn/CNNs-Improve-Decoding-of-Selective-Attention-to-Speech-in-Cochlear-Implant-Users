
from src.evaluation.eval_vanilla_ridge import RidgeEvaluator
import os
import git
from scipy.signal import hilbert
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
#from mne.filter import filter_data
import h5py
from src.models.ridge import Ridge
import pickle
np.random.seed(0)
from scipy.stats import zscore
import pandas
from src.data.eeg_measurement import EegMeasurement
import matplotlib.pyplot as plt
import mne
import scipy
from tqdm import tqdm

"""
Calculate model performance on different window sizes
Note:
The accuracies are calculated for indivdvidual trials and then averaged over all trials.
For the overall accuracy use the raw scores and calculate the accuracy over all trials, a difference arises as an effect of sequential averaging.
"""


subjects = list(range(102,115))
subjects = subjects + list(range(116,117))
subjects = subjects + list(range(118,126))
subjects = subjects + [127, 128, 130]
subjects = [str(x) for x in subjects]
shifts = [-0.5]
lags = ['-200-800ms']
tmin, tmax = [-0.2, 0.8]
sampling_rate = 125
training_scheme = 'concat'
base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir


model_id_s = [str(x) for x in range(50, 57)]
speech_features = ['env' for x in range(7)]
ica_list = [True for x in range(7)]
dB_values = [0, 5, 7.5, 10, 12.5, 15, 20]

database_filenames = [f'ci_attention_SNR_5ms_{int(dB)}dB.hdf5' for dB in dB_values]

freq_range = '1-32Hz'

test_indices_list = [np.array([x]) for x in range(8,20)]
val_indices = np.roll(np.array(test_indices_list), 1).reshape(-1).tolist()
val_indices_list = [np.array([i]) for i in val_indices]
acc_window_sizes = [60,45,30,20,10,5,2,1]


if __name__ == '__main__':
    for model_id, feature, ica, database_filename in tqdm(zip(model_id_s, speech_features, ica_list, database_filenames), total=len(model_id_s)):
        ridge_evaluator = RidgeEvaluator(-200, 800, database_filename = database_filename, freq_range=freq_range, speech_feature=feature, use_ica_data=ica)
        ridge_evaluator.lagged_matrix_fill = 0
        model_dir = os.path.join(base_dir, 'models', 'ridge', model_id)
        attended_scores_mean, distractor_scores_mean, accuracies = [],[],[]
        attended_scores_raw, distractor_scores_raw = [],[]
        
        #iterating over subjects
        for subject in tqdm(subjects):
            print(f'Evaluating subject {subject}')
            path_subj = os.path.join(model_dir, subject)
            subj_attended_scores_mean, subj_distractor_scores_mean, subj_accuracies = [],[],[]
            subj_attended_scores_raw, subj_distractor_scores_raw = [],[]

            #iterating over folds
            for val_indices, test_indices in zip(val_indices_list, test_indices_list):

                model = os.path.join(path_subj, str(test_indices.item()) + '.pk')
                mdl = pandas.read_pickle(model)
                mdl.lagged_matrix_fill = 0

                attended_scores_fold_mean, distractor_scores_fold_mean, accuracies_fold = [],[],[]
                attended_scores_fold_raw, distractor_scores_fold_raw = [],[]

                #iterating over different window sizes
                for acc_win in acc_window_sizes:
                    attended_score_raw, attended_mean, distractor_score_raw, distractor_mean, accuracy = ridge_evaluator.get_attended_scores_from_trained_mdl(mdl, subject, val_indices, test_indices, accuracy_window_size=acc_win)
                    attended_scores_fold_mean.append(attended_mean)
                    distractor_scores_fold_mean.append(distractor_mean)
                    accuracies_fold.append(accuracy)
                    attended_scores_fold_raw.append(attended_score_raw)
                    distractor_scores_fold_raw.append(distractor_score_raw)
                
                subj_attended_scores_mean.append(attended_scores_fold_mean)
                subj_distractor_scores_mean.append(distractor_scores_fold_mean)
                subj_accuracies.append(accuracies_fold)
                subj_attended_scores_raw.append(attended_scores_fold_raw)
                subj_distractor_scores_raw.append(distractor_scores_fold_raw)
            
            attended_scores_mean.append(subj_attended_scores_mean)
            distractor_scores_mean.append(subj_distractor_scores_mean)
            accuracies.append(subj_accuracies)
            attended_scores_raw.append(subj_attended_scores_raw)
            distractor_scores_raw.append(subj_distractor_scores_raw)
        
        #save calculated metrics
        accuracies = np.array(accuracies)
        attended_scores_mean = np.array(attended_scores_mean)
        distractor_scores_mean = np.array(distractor_scores_mean)

        path = os.path.join(base_dir, 'reports', 'metrics', 'ridge', model_id, 'windowed_accuracies_and_raw_scores_ordered_by_subject')
        os.makedirs(path, exist_ok=True)

        np.save(os.path.join(path, 'windowed_accuracies' + model_id + '.npy'), accuracies)
        attended_scores_mean = np.array(attended_scores_mean)
        np.save(os.path.join(path, 'windowed_attended_scores_averaged' + model_id + '.npy'), attended_scores_mean)
        distractor_scores_mean = np.array(distractor_scores_mean)
        np.save(os.path.join(path, 'windowed_distractor_scores_averaged' + model_id + '.npy'), distractor_scores_mean)

        #save raw scores as list using pickle
        with open(os.path.join(path, 'windowed_attended_scores_raw' + model_id + '.pkl'), 'wb') as f:
            pickle.dump(attended_scores_raw, f)
        
        with open(os.path.join(path, 'windowed_distractor_scores_raw' + model_id + '.pkl'), 'wb') as f:
            pickle.dump(distractor_scores_raw, f)



