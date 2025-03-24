from src.data.eeg_measurement import EegMeasurement
from src.data.eeg_attention_pytorch_dataset import EegAttentionDataset
from src.evaluation.eval_vanilla_ridge import RidgeEvaluator

import numpy as np
import git
import os
from pandas import read_csv
from os.path import join

from sklearn import svm
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay

import git
import src.data.utils as data_utils
import src.evaluation.utils as eval_utils
import pickle

import matplotlib.ticker as mtick
from tqdm import tqdm

import argparse

def get_csv_log_file(base_dir, subject):
    """Return absolute path of psychopy csv log-file for given base directory and subject number
    Args:
        base_dir (str): base dir of repository
        subject (int): subject number e.g. 129
    Returns:
        string: path to csv log file of psychopy experiment
    """
    subject_folder = join(base_dir,"data","raw_input", str(subject))
    header_file = ''
    for file in os.listdir(subject_folder):
        if file.endswith('.csv'):
            header_file = file
    assert header_file.endswith('.csv'), "no file with matching datatype found!"
    assert header_file[:3] == str(subject), f"log file should start with subject number {str(subject)} but filename {header_file} was found."
    return join(subject_folder, header_file)
base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir

parser = argparse.ArgumentParser(description='Train CNN model on EEG attention dataset.')

parser.add_argument('-model', type=int, help='0: ridge, 1: cnn pretrained, 2: cnn subject specific')
parser.add_argument('-debug', type=int, help='1:debug mode, 0: normal mode')
parser.add_argument('-thres_only', type=int, help='evaluate only thresholding')
args = parser.parse_args()

subjects = list(range(102,115))
subjects = subjects + list(range(116,117))
subjects = subjects + list(range(118,126))
subjects = subjects + [127, 128, 130]

#defining some global variables
n_subjects = 25
#subjects 115 and 117 are left out --> randomisations have to be adjusted
randomisations = []
for sub in subjects:
    csv_log_path = get_csv_log_file(base_dir, sub)
    psychopy_log_file = read_csv(csv_log_path, sep=',')
    rand = psychopy_log_file.loc[0, 'randomisation']
    randomisations.append(rand)

#set model ids to evaluate
ridge_id = '020'
cnn_pre_id = '161'
cnn_subj_id = '162'

#define paths to reconstruction scores
ridge_attended_raw_path = os.path.join(base_dir, 'reports/metrics/ridge/' + ridge_id + '/windowed_accuracies/windowed_attended_scores_raw' + ridge_id + '.pkl')
ridge_distracor_path = os.path.join(base_dir, 'reports/metrics/ridge/' + ridge_id + '/windowed_accuracies/windowed_distractor_scores_raw' + ridge_id + '.pkl')
ridge_acc_path = os.path.join(base_dir, 'reports/metrics/ridge/' + ridge_id + '/windowed_accuracies/windowed_accuracies' + ridge_id + '.npy')

cnn_attended_raw_path = os.path.join(base_dir, 'reports/metrics/cnn/' + cnn_pre_id + 'pickled_raw/attended_scores_raw.pkl')
cnn_distractor_raw_path = os.path.join(base_dir, 'reports/metrics/cnn/' + cnn_pre_id + 'pickled_raw/distractor_scores_raw.pkl')

cnn_attended_subj_raw_path = os.path.join(base_dir, 'reports/metrics/cnn/' + cnn_subj_id + 'pickled_raw/attended_scores_raw.pkl')
cnn_distractor_subj_raw_path = os.path.join(base_dir, 'reports/metrics/cnn/' + cnn_subj_id + 'pickled_raw/distractor_scores_raw.pkl')


def can_convert_to_numpy(lst):
    try:
        np.array(lst)
        return True
    except ValueError:
        return False

if __name__ == '__main__':
    if args.model == 0:
        #load reconstruction scores
        ##Ridge
        with open(ridge_attended_raw_path, 'rb') as f:
            ridge_attended_raw = pickle.load(f)
        with open(ridge_distracor_path, 'rb') as f:
            ridge_distractor_raw = pickle.load(f)

        ridge_att_raw = data_utils.correct_trials_list(ridge_attended_raw, randomisations)
        ridge_distr_raw = data_utils.correct_trials_list(ridge_distractor_raw, randomisations)

        elb_att_ridge_raw, pol_att_ridge_raw = data_utils.get_elb_pol_attended(ridge_att_raw)
        elb_distr_ridge_raw, pol_distr_ridge_raw = data_utils.get_elb_pol_attended(ridge_distr_raw)
        
        lin_acc_thres_list = []
        lin_acc_svm_list = []

    elif args.model == 1:
        ##CNN Pretrained
        with open(cnn_attended_raw_path, 'rb') as f:
            cnn_attended_raw = pickle.load(f)
        with open(cnn_distractor_raw_path, 'rb') as f:
            cnn_distractor_raw = pickle.load(f)

        #correct trals for subjects with randomisation 1
        cnn_att_raw = data_utils.correct_trials_list(cnn_attended_raw, randomisations)
        cnn_distr_raw = data_utils.correct_trials_list(cnn_distractor_raw, randomisations)

        elb_att_cnn_raw, pol_att_cnn_raw = data_utils.get_elb_pol_attended(cnn_att_raw)
        elb_distr_cnn_raw, pol_distr_cnn_raw = data_utils.get_elb_pol_attended(cnn_distr_raw)
        
        cnn_acc_thres_list = []
        cnn_acc_svm_list = []
    
    elif args.model == 2:
        ###CNN subject specific
        with open(cnn_attended_subj_raw_path, 'rb') as f:
            cnn_attended_subj_raw = pickle.load(f)
        with open(cnn_distractor_subj_raw_path, 'rb') as f:
            cnn_distractor_subj_raw = pickle.load(f)

        cnn_att_raw_subj = data_utils.correct_trials_list(cnn_attended_subj_raw, randomisations)
        cnn_distr_raw_subj = data_utils.correct_trials_list(cnn_distractor_subj_raw, randomisations)

        elb_att_cnn_raw_subj, pol_att_cnn_raw_subj = data_utils.get_elb_pol_attended(cnn_att_raw_subj)
        elb_distr_cnn_raw_subj, pol_distr_cnn_raw_subj = data_utils.get_elb_pol_attended(cnn_distr_raw_subj)

        cnn_acc_thres_list_subj = []
        cnn_acc_svm_list_subj = []


    if args.debug:
        n_windows = 2
    else:
        n_windows = 8

    for window_index in tqdm(range(0, n_windows)):
        ###Ridge
        if args.model == 0:
            #create dataset for current window length
            X_lin_raw, y_lin_raw, cv_lin = eval_utils.create_dataset(elb_att_ridge_raw, elb_distr_ridge_raw, pol_att_ridge_raw, pol_distr_ridge_raw, window_index, n_subjects=n_subjects)
            #eval thresholding
            lin_acc_thres, lin_elb_recall_thres, lin_elb_prec_thres, lin_elb_f1_thres, lin_pol_recall_thres, lin_pol_prec_thres, lin_pol_f1_thres = eval_utils.score_thresholding(X_lin_raw, y_lin_raw, cv_lin)
            #eval SVM
            if not args.thres_only:
                lin_acc_svm, lin_elb_recall_svm, lin_elb_prec_svm, lin_elb_f1_svm, lin_pol_recall_svm, lin_pol_prec_svm, lin_pol_f1_svm = eval_utils.score_SVM(X_lin_raw,y_lin_raw, cv = cv_lin)
            #keep track of scores
            lin_acc_thres_list.append(lin_acc_thres)
            if not args.thres_only:
                lin_acc_svm_list.append(lin_acc_svm)

        elif args.model == 1:
            ####Pretrained CNN
            X_cnn_raw, y_cnn_raw, cv_cnn = eval_utils.create_dataset(elb_att_cnn_raw, elb_distr_cnn_raw, pol_att_cnn_raw, pol_distr_cnn_raw, window_index, n_subjects=n_subjects)
            cnn_acc_thres, cnn_elb_recall_thres, cnn_elb_prec_thres, cnn_elb_f1_thres, cnn_pol_recall_thres, cnn_pol_prec_thres, cnn_pol_f1_thres = eval_utils.score_thresholding(X_cnn_raw, y_cnn_raw, cv_cnn)
            if not args.thres_only:
                cnn_acc_svm, cnn_elb_recall_svm, cnn_elb_prec_svm, cnn_elb_f1_svm, cnn_pol_recall_svm, cnn_pol_prec_svm, cnn_pol_f1_svm = eval_utils.score_SVM(X_cnn_raw,y_cnn_raw, cv = cv_cnn)
            cnn_acc_thres_list.append(cnn_acc_thres)
            if not args.thres_only:
                cnn_acc_svm_list.append(cnn_acc_svm)

        elif args.model == 2:
            #### Pretrained CNN
            X_cnn_raw_subj, y_cnn_raw_subj, cv_cnn_subj = eval_utils.create_dataset(elb_att_cnn_raw_subj, elb_distr_cnn_raw_subj, pol_att_cnn_raw_subj, pol_distr_cnn_raw_subj, window_index, n_subjects=n_subjects)
            cnn_acc_subj_thres, cnn_elb_recall_subj_thres, cnn_elb_prec_subj_thres, cnn_elb_f1_subj_thres, cnn_pol_recall_subj_thres, cnn_pol_prec_subj_thres, cnn_pol_f1_subj_thres = eval_utils.score_thresholding(X_cnn_raw_subj, y_cnn_raw_subj, cv_cnn_subj)
            if not args.thres_only:
                cnn_acc_svm_subj, cnn_elb_recall_svm_subj, cnn_elb_prec_svm_subj, cnn_elb_f1_svm_subj, cnn_pol_recall_svm_subj, cnn_pol_prec_svm_subj, cnn_pol_f1_svm_subj = eval_utils.score_SVM(X_cnn_raw_subj,y_cnn_raw_subj, cv = cv_cnn_subj)
            cnn_acc_thres_list_subj.append(cnn_acc_subj_thres)
            if not args.thres_only:
                cnn_acc_svm_list_subj.append(cnn_acc_svm_subj)


    if args.model == 0:
        #save the scores
        ridge_path =os.path.join(base_dir, 'reports/metrics/ridge/' + ridge_id + '/windowed_svm_values')
        os.makedirs(ridge_path, exist_ok=True)

        if can_convert_to_numpy(lin_acc_thres_list):
            lin_acc_thres_list = np.array(lin_acc_thres_list)
            np.save(os.path.join(ridge_path, 'lin_acc_thres_list.npy'), lin_acc_thres_list)
        else:
            #save as pickle
            with open(os.path.join(ridge_path, 'lin_acc_thres_list.pkl'), 'wb') as f:
                pickle.dump(lin_acc_thres_list, f)

        if not args.thres_only:
            if can_convert_to_numpy(lin_acc_svm_list):
                lin_acc_svm_list = np.array(lin_acc_svm_list)
                np.save(os.path.join(ridge_path, 'lin_acc_svm_list.npy'), lin_acc_svm_list)
            else:
                #save as pickle
                with open(os.path.join(ridge_path, 'lin_acc_svm_list.pkl'), 'wb') as f:
                    pickle.dump(lin_acc_svm_list, f)

    elif args.model == 1:
        cnn_path = os.path.join(base_dir, 'reports/metrics/cnn/' + cnn_pre_id + '/windowed_svm_valued')
        os.makedirs(cnn_path, exist_ok=True)

        if can_convert_to_numpy(cnn_acc_thres_list):
            cnn_acc_thres_list = np.array(cnn_acc_thres_list)
            np.save(os.path.join(cnn_path, 'cnn_acc_thres_list.npy'), cnn_acc_thres_list)
        else:
            #save as pickle
            with open(os.path.join(cnn_path, 'cnn_acc_thres_list.pkl'), 'wb') as f:
                pickle.dump(cnn_acc_thres_list, f)
        
        if not args.thres_only:
            if can_convert_to_numpy(cnn_acc_svm_list):
                cnn_acc_svm_list = np.array(cnn_acc_svm_list)
                np.save(os.path.join(cnn_path, 'cnn_acc_svm_list.npy'), cnn_acc_svm_list)
            else:
                #save as pickle
                with open(os.path.join(cnn_path, 'cnn_acc_svm_list.pkl'), 'wb') as f:
                    pickle.dump(cnn_acc_svm_list, f)


    elif args.model == 2:
        cnn_path_subj = os.path.join(base_dir, 'reports/metrics/cnn/' + cnn_subj_id + '/windowed_svm_valued')
        os.makedirs(cnn_path_subj, exist_ok=True)

        if can_convert_to_numpy(cnn_acc_thres_list_subj):
            cnn_acc_thres_list_subj = np.array(cnn_acc_thres_list_subj)
            np.save(os.path.join(cnn_path_subj, 'cnn_acc_thres_list.npy'), cnn_acc_thres_list_subj)
        else:
            #save as pickle
            with open(os.path.join(cnn_path_subj, 'cnn_acc_thres_list.pkl'), 'wb') as f:
                pickle.dump(cnn_acc_thres_list_subj, f)

        if not args.thres_only:
            if can_convert_to_numpy(cnn_acc_svm_list_subj):
                cnn_acc_svm_list_subj = np.array(cnn_acc_svm_list_subj)
                np.save(os.path.join(cnn_path_subj, 'cnn_acc_svm_list.npy'), cnn_acc_svm_list_subj)
            else:
                #save as pickle
                with open(os.path.join(cnn_path_subj, 'cnn_acc_svm_list.pkl'), 'wb') as f:
                    pickle.dump(cnn_acc_svm_list_subj, f)
