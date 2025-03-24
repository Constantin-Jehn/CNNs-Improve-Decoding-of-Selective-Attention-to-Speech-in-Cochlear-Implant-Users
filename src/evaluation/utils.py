from src.data.eeg_measurement import EegMeasurement
from src.data.eeg_attention_pytorch_dataset import EegAttentionDataset
from src.evaluation.eval_vanilla_ridge import RidgeEvaluator
import numpy as np
import git
import os
import pandas
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import scipy
from spyeeg.models.TRF import TRFEstimator

from sklearn import svm
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import cross_validate

import git
import src.data.utils as data_utils
import pickle

from scipy.stats import ttest_rel
import h5py
from tqdm import tqdm
import mne

from scipy.ndimage.morphology import binary_erosion

import torch
from torch.ao.quantization import (
  get_default_qconfig_mapping,
  get_default_qat_qconfig_mapping,
  QConfigMapping,
)
import torch.ao.quantization.quantize_fx as quantize_fx

base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir

def create_dataset(elb_att_raw, elb_distr_raw, pol_att_raw, pol_distr_raw, window_index, n_subjects):
    """
    Creates dataset to use for sklearn cross-validation for svm and thresholding

    Args:
        elb_att_raw (list): reconstruction scres of elbenwald attended, the attended stream
        elb_distr_raw (list): reconstruction scres of elbenwald attended, the distractor stream
        pol_att_raw (list): reconstruction scres of pol attended, the attended stream
        pol_distr_raw (list): reconstruction scres of pol attended, the distractor stream
        window_index (int): desired window lenght, encoded as index from following list: [60,45,30,20,10,5,2,1]
        n_subjects (int): number of subjects

    Returns:
        tuple: X (n_samples x 2 features), y (n_samples x 1 labels), cv (cross-validation splits)
    """
    #elb_distr: elbenwald was in focus: scores of the distractor, meaning pol
    #elb_att: elbenwald was in focus: scores of the attended, meaning elb
    #build the feature vectors r_elb and r_pl structre [subj0, subj1,....,subjn]
    #within each subject [elb_attended, pol_attended]

    #get elbenwald and polarnacht scores independent of focus
    elb_scores_raw = data_utils.flatten_subject_first_over_two_lists(elb_att_raw, pol_distr_raw, window_index)
    pol_scores_raw = data_utils.flatten_subject_first_over_two_lists(elb_distr_raw, pol_att_raw, window_index)

    #number of samples for the two classes
    n_elb = data_utils.flatten_subject_raw_data(elb_att_raw, window_index).shape[0] / n_subjects
    n_pol = data_utils.flatten_subject_raw_data(pol_att_raw, window_index).shape[0] / n_subjects
    assert n_elb % 1 == 0, 'number of elb samples is not an integer'
    assert n_pol % 1 == 0, 'number of pol samples is not an integer'
    #assemble the labels
    elb_ridge_labels = np.tile(np.append(np.ones(int(n_elb)), np.zeros((int(n_pol))), axis=0), n_subjects)
    assert elb_scores_raw.shape[0] == elb_ridge_labels.shape[0], 'number of samples and labels do not match'

    X = np.vstack((elb_scores_raw, pol_scores_raw)).T
    y = elb_ridge_labels
    
    # Create test split - leave one subject out
    cv = []
    #number of samples per subject
    n_ridge = int(n_elb + n_pol)
    for sub in range(0, n_subjects):
        test_ind = np.arange(sub * n_ridge, (sub + 1) * n_ridge)
        train_ind = np.delete(np.arange(0, n_subjects * n_ridge), test_ind)
        cv.append((train_ind, test_ind))

    return X, y, cv

def score_thresholding(X,y, cv):
    """
    X: 2D array of shape (n_samples, (r_elb, r_pol))
    y: labels: 1 if elbenwald was attended, 0 if polarnacht was attended
    cv: list of tuples (train_ind, test_ind)
    
    returns
    tuple: acc, elb_recall_thr, elb_precision_thr, f1_elb, pol_recall_thr, pol_precision_thr, f1_pol
    """

    acc_list, elb_recall_thr_list, elb_precision_thr_list, f1_elb_list, pol_recall_thr_list, pol_precision_thr_list, f1_pol_list = [], [], [], [], [], [], []
    for _, test_ind in cv:
        #one split is one subject
        X_tmp = X[test_ind,:]
        y_tmp = y[test_ind]

        predictions = X_tmp[:,0] > X_tmp[:,1]
        true_pred = np.sum(predictions == y_tmp)
        acc = true_pred / y_tmp.shape[0]
        acc_list.append(acc)

        #for Elbenwald
        true_pos = np.sum(predictions[y_tmp == 1])
        false_pos = np.sum(predictions[y_tmp == 0])
        false_neg = np.sum(np.invert(predictions[y_tmp == 1]))

        elb_recall_thr = true_pos / (true_pos + false_neg)
        elb_recall_thr_list.append(elb_recall_thr)

        elb_precision_thr = true_pos / (true_pos + false_pos)
        elb_precision_thr_list.append(elb_precision_thr)

        f1_elb = 2 * (elb_precision_thr * elb_recall_thr) / (elb_precision_thr + elb_recall_thr)
        f1_elb_list.append(f1_elb)

        #for Polarnacht
        pol_recall_thr =  np.sum(np.invert(predictions[y_tmp == 0])) / np.sum(y_tmp == 0)
        pol_recall_thr_list.append(pol_recall_thr)

        pol_precision_thr = np.sum(np.invert(predictions[y_tmp == 0])) / np.sum(predictions == 0)
        pol_precision_thr_list.append(pol_precision_thr)

        f1_pol = 2 * (pol_precision_thr * pol_recall_thr) / (pol_precision_thr + pol_recall_thr)
        f1_pol_list.append(f1_pol)

    return acc_list, elb_recall_thr_list, elb_precision_thr_list, f1_elb_list, pol_recall_thr_list, pol_precision_thr_list, f1_pol_list


def score_SVM(X, y, cv, verbose=False):
    """
    Scores a SVM with linear kernel on the data X with labels y.
    Args:
        X (np.array): n_samples x (r_elb, r_pol)
        y (np.array): n_samples x 1 (labels: 1 if elbenwald was attended, 0 if polarnacht was attended)
        cv(list): list of tuples of train and test indices
        verbose (bool, optional): Whether to print the scores. Defaults to False.

    Returns:
        mean_test_acc, mean_test_recall_elb, mean_test_precision_elb,  mean_test_f1_elb, mean_test_recall_pol, mean_test_precision_pol,  mean_test_f1_pol
    """
    model = SVC(kernel='linear', C=100, random_state=0)
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1']
    scores_elb = cross_validate(model, X, y, scoring=scoring, cv=cv, return_train_score=True)
    test_acc = scores_elb["test_accuracy"]
    test_precision_elb = scores_elb["test_precision_macro"]
    test_recall_elb = scores_elb["test_recall_macro"]
    test_f1_elb = scores_elb["test_f1"]
    
    if verbose:
        print(f'Mean test accuracy: {test_acc.mean()}\n')
        print('For Detecting Elbenwald:')
        print(f'Mean test precision: {test_precision_elb.mean()}')
        print(f'Mean test recall: {test_recall_elb.mean()}')
        print(f'Mean test f1: {test_f1_elb.mean()} \n')

    y_pol = np.copy(y)
    y_pol[y == 0] = 1
    y_pol[y==1] = 0
    #y_pol[::2] = 0
    model = SVC(kernel='linear', C=100, random_state=0)
    scores_pol = cross_validate(model, X, y_pol, scoring=scoring, cv=cv, return_train_score=True)
    #print(f'Mean test accuracy: {np.mean(scores_pol["test_accuracy"])}')
    test_precision_pol = scores_pol["test_precision_macro"]
    test_recall_pol = scores_pol["test_recall_macro"]
    test_f1_pol = scores_pol["test_f1"]

    if verbose:
        print('For Detecting Polarnacht:')
        print(f'Mean test precision: {test_precision_pol.mean()}')
        print(f'Mean test recall: {test_recall_pol.mean()}')
        print(f'Mean test f1: {test_f1_pol.mean()}')

    return test_acc, test_recall_elb, test_precision_elb,  test_f1_elb, test_recall_pol, test_precision_pol,  test_f1_pol


from scipy.stats import ttest_rel
def check_significances_on_windows(acc_arr_0, acc_arr_1, alternative='two-sided'):
    """
    Checks one-sided t-test for significance of difference between two arrays of accuracies
    acc_arr_1 ist tested to be larger than acc_arr_0

    Args:
        acc_arr_0 (np.array): shape (n_windows, n_subjects)
        acc_arr_1 (np.array): shape (n_windows, n_subjects)
        alternative (str): 'less' or 'greater' or 'two-sided'
    """
    window_sizes = [60, 45, 30, 20, 10, 5, 2,1]
    print('p-values for different window sizes on population level comparing ridge and cnn (pretrained)')
    significances = []
    counter_significant = 0
    for wind_size, i in zip(window_sizes, range(0, len(window_sizes))):
        acc_0_win = acc_arr_0[i,:]
        acc_1_win = acc_arr_1[i,:]
        res = ttest_rel(acc_0_win, acc_1_win, alternative=alternative)
        if res.pvalue < 0.05:
            counter_significant += 1
            significances.append(True)
        else:
            significances.append(False)
        print('window size: {}s, p-value: {}'.format(wind_size, res.pvalue))
    print('number of windows with significanct differeces: {}'.format(counter_significant))
    return significances

def find_start_distractor(distr_arr, thres_window = 10):
    """
    Finds index where the distractor stars, by finding the first {thres_window} consecutive samples above the mean.

    Args:
        distr_arr (np.array): array of distractor envelope
        thres_window (int, optional): number of consecutive samples above the mean to be considered as start of distractor. Defaults to 10.

    Returns:
        int: index where the distractor starts
    """
    m = distr_arr > distr_arr.mean()
    k = np.ones(thres_window,dtype=bool)
    starting_index = binary_erosion(m,k,origin=-(thres_window//2)).argmax()
    return starting_index

def trf_helper(speech_feature, eeg, tmin, tmax, Fs, alpha):
    ""
    #Carries out trf computation and returns coefficients
    ""
    n_times = eeg.shape[-1]
    n_train = int(n_times * 0.8)
    speech_train, eeg_train = speech_feature[:n_train], eeg[:,:n_train]
    speech_test, eeg_test = speech_feature[n_train:], eeg[:, n_train:]

    trf = TRFEstimator(tmin=tmin, tmax=tmax, srate=Fs, alpha=alpha, alpha_feat=False)
    trf.fit(np.expand_dims(speech_train, axis=1), eeg_train.T)

    scores = trf.score(np.expand_dims(speech_test, axis=1), eeg_test.T)

    mean_scores_alpha = np.mean(scores, axis=0)
    best_alpha = np.argmax(mean_scores_alpha)
    coefs = trf.get_coef()[:, 0, :, best_alpha].T
    return coefs
    


def calc_trf_coefs(data_dir, subjects, trials = np.arange(1,21), tmin = -0.1, tmax= 0.5, null_model = False, re_ref_avg = True, cohort = 'ci', alpha=[0.001,0.1,1,10,100]):
    """
    Calculate TRF coeficients equivialent to coeficients of the linear forward model

    Args:
        data_dir (str): path to hdf5 database
        subjects (np.array): list of subjects to include
        trials (np.array, optional): list of trials to include. Defaults to np.arange(1,21).
        null_model (bool, optional): Wheter to calculate a null model for later statistical analyse. Defaults to False.
        re_ref_avg (bool, optional): Wheter to use the re-referenced EEG data (to average). Defaults to True. Watch out to use the correct data_dir.
        cohort (str, optional): Which patient cohort is the data from (ci or control)

    Returns:
        (list0, list1, ....): coefs_eeg, coefs_eeg_distr, coefs_eeg_ica, coefs_eeg_ica_distr, coefs_eeg_onset, coefs_eeg_onset_distr, coefs_eeg_ica_onset, coefs_eeg_ica_onset_distr
    """
    assert cohort in ['ci', 'control'], f'Cohort must be ci or control. But {cohort} passed.'
    #constants for TRF
    Fs = 125

    header_file_dir = os.path.join(base_dir, 'data/raw_input/103/103.vhdr')
    #load info file
    raw = mne.io.read_raw_brainvision(header_file_dir)
    raw.drop_channels(['Aux1', 'Aux2'])
    raw.resample(Fs)
    info = raw.info

    #ica components to exclude for ica on raw data
    if re_ref_avg:
        #ica copmponents to exclude for ica on raw data rereferenced to avg
        ica_exclude = {
        '101': [0,2,3,6,11,13,14,16,18, 19],
        '102': [0,1,2,3,4,7,8,10,13,16,18,20,22,23,29],
        '103': [0,1,2,3,4,5,6,8,10,12,13,14,15,19,21,22,23,24,25],
        '104': [0,4],
        '105': [0,1,2,3,4,6,7,8,10,11,12,13,15,16,17,23,27,28,29],
        '106': [0,1,2,3,4,5,6,7,8,9,10,12,14,15,16,17,18,24,25,29],
        '107': [0,1,2,3,4,6,7,8,9,10,12,15,17,18,21,22,24,25,29],
        '108': [0,1,2,4,5,6,8,11,12,13,21,26],
        '109': [0,1,2,3,4,5,6,7,9,12,13,14,15,17,22,23,24,25,26,28],
        '110': [0,1,2,3,4,5,6,7,8,9,10,14,15,19,20,21,22,25],
        '111': [0,1,3,5,6,8,7,9,11,12,13,14,18,21,26,27,28],
        '112': [0,1,3,2,5,6,8,9,10,11,12,14,17,18,20,24,21,26,29],
        '113': [0,1,2,3,4,5,6,7,8,9,10,12,14,15,22],
        '114': [0,1,2,3,4,6,7,8,10,11,12,13,15,17,18,22,25,27,28,29],
        '116': [0,1,2,3,5,6,7,9,11,12,13,21,24,25],
        '118': [0,1,2,3,4,5,6,7,8,9,13,14,15,16,17,18,19,20,21,22,23,25,26,28,29],
        '119': [0,1,2,3,4,5,20,28],
        '120': [0,1,2,6,7],
        '121': [0,1,2,3,4,6,7,8,10,11,13,18,23,24,26,28,29],
        '122': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,14,22],
        '123': [0,1,2,3,6,8,9,10,11,12,16,18,20,21,22,24],
        '124': [0,1,2,3,4,5,6,9,11,12,13,18,20,23,28],
        '125': [0,1,2,3,8,9,12,16,18,23,29]
        }
    else:
        #ica copmponents to exclude for ica on raw data
        ica_exclude = {
        '101': [0,1,2,3,4,5,6,9,11,13],
        '102': [0,1,2,3,4,5,7,11,15,16],
        '103': [0,1,2,3,4,5,6,8,10,11,13,14,17,19],
        '104': [0,3,6,9,11],
        '105': [0,1,2,3,5,6,7,8,9,10,11,12,14,15,16,17,18,19,22,26,28,29],
        '106': [0,1,2,3,4,5,8,10,14,15,16],
        '107': [0,1,2,3,6,7,9,11,12,13,16,17,26],
        '108': [0,1,2,3,4,5,6,7,8,10,11,12,13, 14,16],
        '109': [0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,18,19,24,27,28,29],
        '110': [0,1,2,3,4,5,6,7,8,9,10,12,13,14,18],
        '111': [0,1,4,5,6,8,9,12,13,14,15,17,18,19,22,24,26,27,29],
        '112': [0,2,3,5,9,10,11,12,14,15,16,19,22,25,26],
        '113': [0,1,2,3,4,5,6,9,11,13,14,15,16,18],
        '114': [0,1,2,3,5,7,8,9,10,17,26],
        '116': [0,1,2,3,5,6,8,9,10,12,14,15,20,25,26,27],
        '118': [0,1,2,3,4,5,6,8,9,10,11,12,14,18,19,20],
        '119': [0,1,2,3,4,5,6,7,8,9,10,11,12,14,16,17,18,19,20,22,26,27],
        '120': [0,1,2,3,4,5,6,9,12,13,17],
        '121': [0,1,2,3,4,5,6,10,11,13,14,17,19,20,21,27,28],
        '122': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,19,20,21,22,23,24,25,27],
        '123': [0,1,2,3,4,5,6,7,8,9,10,11,14,18,19,20,21,22,24,25,26,28,29],
        '124': [0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,18,20,21,22,23,24,25,26,27,28,29],
        '125': [0,1,2,3,4,5,6,8,9,10,11,12,15,19,20,23,24]
        }

    with h5py.File(data_dir, 'r') as f:
        coefs_eeg = []
        coefs_eeg_distr = []
        coefs_eeg_ica = []
        coefs_eeg_ica_distr = []

        coefs_eeg_onset = []
        coefs_eeg_onset_distr = []
        coefs_eeg_ica_onset = []
        coefs_eeg_ica_onset_distr = []

        for subject in tqdm(subjects):
            coefs_eeg_subject = []
            coefs_eeg_distr_subject = []
            coefs_eeg_ica_subject = []
            coefs_eeg_ica_distr_subject = []

            coefs_eeg_onset_subject = []
            coefs_eeg_onset_distr_subject = []
            coefs_eeg_ica_onset_subject = []
            coefs_eeg_ica_onset_distr_subject = []

            if cohort == 'ci':
                # use cleaned data from hdf5
                pass
                # if re_ref_avg:
                #     ica_path = os.path.join(base_dir, f'data/ica/re_ref/{subject}/{subject}ica')
                # else:
                #     ica_path = os.path.join(base_dir, f'data/ica/processed/{subject}/{subject}ica')
                # ica = mne.preprocessing.read_ica(ica_path, verbose=False)
                # ica.exclude = ica_exclude[str(subject)]
            
            elif cohort == 'control':
                #using cleaned data from h5py
                pass


            for trial in trials:
                eeg_path = f'eeg/{subject}/{trial}'
                eeg_ica_path = f'eeg_ica/{subject}/{trial}'

                stim_code = f[eeg_path].attrs['stimulus']

                env_attended_path = f'stimulus_files/{stim_code}/attended_env'
                env_distractor_path = f'stimulus_files/{stim_code}/distractor_env'

                onset_env_attended_path = f'stimulus_files/{stim_code}/attended_onset_env'
                onset_env_distractor_path = f'stimulus_files/{stim_code}/distractor_onset_env'

                env_attended_compl, env_distractor_compl = f[env_attended_path][:], f[env_distractor_path][:]
                onset_env_attended_compl, onset_env_distractor_compl = f[onset_env_attended_path][:], f[onset_env_distractor_path][:]
                
                #limit to space where distractor is present
                start_env_distractor = find_start_distractor(env_distractor_compl)
                env_attended, env_distractor = env_attended_compl[start_env_distractor:], env_distractor_compl[start_env_distractor:]
                onset_env_attended, onset_env_distractor = onset_env_attended_compl[start_env_distractor:], onset_env_distractor_compl[start_env_distractor:]

                #for null-model reverse the speech features
                if null_model:
                    env_attended, env_distractor = env_attended[::-1], env_distractor[::-1]
                    onset_env_attended, onset_env_distractor = onset_env_attended[::-1], onset_env_distractor[::-1]

                eeg_data_compl = f[eeg_path][:]
                eeg_ica_data_compl = f[eeg_ica_path][:]
                
                assert eeg_data_compl.shape[0] == 33, "EEG measurement requires 33 channels. Loaded that is incorrect."
                assert eeg_ica_data_compl.shape[0] == 33, "EEG measurement requires 33 channels. Loaded that is incorrect."

                #drop auxiliary channels and remove part where no distractor is present
                eeg_data = eeg_data_compl[:31,start_env_distractor:]
                
                #eeg_ica_data = eeg_ica_data[:31,:]
                #eeg_data = eeg_data[~mask,:]
                #
                #eeg + attended env
                # trf = TRFEstimator(tmin=tmin, tmax=tmax, srate=Fs, alpha=alpha)
                # trf.fit(np.expand_dims(env_attended, axis=1), eeg_data.T)
                # coef_envelope = trf.get_coef()[:, 0, :, 0].T

                coef_envelope = trf_helper(speech_feature = env_attended, eeg = eeg_data, tmin = tmin, tmax=tmax, Fs=Fs, alpha=alpha)
                coefs_eeg_subject.append(coef_envelope)
                
                if trial > 8:
                    #eeg + distractor env
                    # trf = TRFEstimator(tmin=tmin, tmax=tmax, srate=Fs, alpha=alpha)
                    # trf.fit(np.expand_dims(env_distractor, axis=1), eeg_data.T)
                    # coef_envelope_distr = trf.get_coef()[:, 0, :, 0].T

                    coef_envelope_distr = trf_helper(speech_feature = env_distractor, eeg = eeg_data, tmin = tmin, tmax=tmax, Fs=Fs, alpha=alpha)
                    coefs_eeg_distr_subject.append(coef_envelope_distr)

                #eeg + attended onset env
                # trf = TRFEstimator(tmin=tmin, tmax=tmax, srate=Fs, alpha=alpha)
                # trf.fit(np.expand_dims(onset_env_attended, axis=1), eeg_data.T)
                # coef_onset = trf.get_coef()[:, 0, :, 0].T

                coef_onset = trf_helper(speech_feature = onset_env_attended, eeg = eeg_data, tmin = tmin, tmax=tmax, Fs=Fs, alpha=alpha)
                coefs_eeg_onset_subject.append(coef_onset)

                if trial > 8:
                    #eeg + distractor onset env
                    # trf = TRFEstimator(tmin=tmin, tmax=tmax, srate=Fs, alpha=alpha)
                    # trf.fit(np.expand_dims(onset_env_distractor, axis=1), eeg_data.T)
                    # coef_onset_distr = trf.get_coef()[:, 0, :, 0].T

                    coef_onset_distr = trf_helper(speech_feature = onset_env_distractor, eeg = eeg_data, tmin = tmin, tmax=tmax, Fs=Fs, alpha=alpha)
                    coefs_eeg_onset_distr_subject.append(coef_onset_distr)

                #apply ica
                if cohort == 'ci':
                    # take data from hdf5 dataset
                    eeg_ica_data = eeg_ica_data_compl[:31, start_env_distractor:]
                    calc_trf = True
                    # if len(ica.exclud) > 0 :
                    #     raw_ica = mne.io.RawArray(eeg_data, info, verbose=False)
                    #     raw_ica = ica.apply(raw_ica, exclude=ica.exclude, verbose=False)
                    #     eeg_ica_data = raw_ica.get_data()
                    #     assert eeg_ica_data.shape[0] == 31, "EEG measurement requires 31 channels. Loaded that is incorrect."
                    #     calc_trf = True
                    # else:
                    #     calc_trf = False
                elif cohort == 'control':
                    #take data from hdf5 dataset
                    eeg_ica_data = eeg_ica_data_compl[:31, start_env_distractor:]
                    calc_trf = True
                
                if calc_trf:
                    #eeg ica + attended env
                    # trf = TRFEstimator(tmin=tmin, tmax=tmax, srate=Fs, alpha=alpha)
                    # trf.fit(np.expand_dims(env_attended, axis=1), eeg_ica_data.T)
                    # coef_envelope = trf.get_coef()[:, 0, :, 0].T

                    coef_envelope = trf_helper(speech_feature = env_attended, eeg = eeg_ica_data, tmin = tmin, tmax=tmax, Fs=Fs, alpha=alpha)
                    coefs_eeg_ica_subject.append(coef_envelope)

                    if trial > 8:
                        #eeg ica + distractor env
                        # trf = TRFEstimator(tmin=tmin, tmax=tmax, srate=Fs, alpha=[10])
                        # trf.fit(np.expand_dims(env_distractor, axis=1), eeg_ica_data.T)
                        # coef_envelope_distr = trf.get_coef()[:, 0, :, 0].T

                        coef_envelope_distr = trf_helper(speech_feature = env_distractor, eeg = eeg_ica_data, tmin = tmin, tmax=tmax, Fs=Fs, alpha=alpha)
                        coefs_eeg_ica_distr_subject.append(coef_envelope_distr)

                    #eeg ica + attended onset env
                    # trf = TRFEstimator(tmin=tmin, tmax=tmax, srate=Fs, alpha=alpha)
                    # trf.fit(np.expand_dims(onset_env_attended, axis=1), eeg_ica_data.T)
                    # coef_onset = trf.get_coef()[:, 0, :, 0].T

                    coef_onset = trf_helper(speech_feature = onset_env_attended, eeg = eeg_ica_data, tmin = tmin, tmax=tmax, Fs=Fs, alpha=alpha)
                    coefs_eeg_ica_onset_subject.append(coef_onset)

                    if trial > 8:
                        #eeg ica + distractor onset env
                        # trf = TRFEstimator(tmin=tmin, tmax=tmax, srate=Fs, alpha=alpha)
                        # trf.fit(np.expand_dims(onset_env_distractor, axis=1), eeg_ica_data.T)
                        # coef_onset_distr = trf.get_coef()[:, 0, :, 0].T

                        coef_onset_distr = trf_helper(speech_feature = onset_env_distractor, eeg = eeg_ica_data, tmin = tmin, tmax=tmax, Fs=Fs, alpha=alpha)
                        coefs_eeg_ica_onset_distr_subject.append(coef_onset_distr)
                else:
                    # just use coefs from normal envelope
                    coefs_eeg_ica_subject.append(coef_envelope)
                    coefs_eeg_ica_onset_subject.append(coef_onset)
                    if trial > 8:
                        coefs_eeg_ica_distr_subject.append(coef_envelope_distr)
                        coefs_eeg_ica_onset_distr_subject.append(coef_onset_distr)

            coefs_eeg.append(coefs_eeg_subject)
            coefs_eeg_distr.append(coefs_eeg_distr_subject)

            coefs_eeg_ica.append(coefs_eeg_ica_subject)
            coefs_eeg_ica_distr.append(coefs_eeg_ica_distr_subject)

            coefs_eeg_onset.append(coefs_eeg_onset_subject)
            coefs_eeg_onset_distr.append(coefs_eeg_onset_distr_subject)

            coefs_eeg_ica_onset.append(coefs_eeg_ica_onset_subject)
            coefs_eeg_ica_onset_distr.append(coefs_eeg_ica_onset_distr_subject)
    return coefs_eeg, coefs_eeg_distr, coefs_eeg_ica, coefs_eeg_ica_distr, coefs_eeg_onset, coefs_eeg_onset_distr, coefs_eeg_ica_onset, coefs_eeg_ica_onset_distr

def load_quantized_model(path, model):
    """_summary_

    Args:
        path (int): path to quantized model
        model (torch.module): initialized model with random weights
    """
    qconfig_mapping = get_default_qat_qconfig_mapping('qnnpack')
    model.eval()
    model = quantize_fx.prepare_fx(model, qconfig_mapping, None)
    model = quantize_fx.convert_fx(model)
    state_dic = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dic)
    return model
