"""
This script has turned into the place where I put experiments and hyperparemter tuning for the CNN model.
In the long run it is supposed to work with cnn_trainer.py - a class that allows to run the experiments based on a config file.
"""
from src.evaluation.training_functions import train_dnn
import os
import git
import pickle
import torch
import numpy as np
import h5py
from src.data.eeg_attention_pytorch_dataset import EegAttentionDataset
from scipy.stats import pearsonr
import optuna
from src.models.dnn import CNN, CNN_2
import tomllib
import argparse
from torch.optim import NAdam
from collections.abc import Iterable
from torch.utils.data import DataLoader

from torch.optim import NAdam, Adam

from optuna.storages import JournalStorage, JournalFileStorage
import copy

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

parser = argparse.ArgumentParser(description='Train CNN model on EEG attention dataset.')

parser.add_argument('-subj_0', type=int, help='index of first subject to train')
parser.add_argument('-subj_1', type=int, help='index of last subject to train')
parser.add_argument('-model_id', type=str, help='model id')
parser.add_argument('-job_nr', type=int, help='learning rate')
parser.add_argument('-activation_fct', type=str, help='Activation function either ELU, ReLU or LeakyReLU')
parser.add_argument('-conv_bias', type=int, help='Whether to use bias in convolutional layer')
parser.add_argument('-dB', type=int, help='SNR threshold level for CI artifact rejection')
args = parser.parse_args()

base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
data_dir = os.path.join(base_dir, f"data/processed/ci_attention_final_SNR_crit_30dB.hdf5")

def generate_test_and_val_indices():
    random_state = 672
    competing_indices_book_0 = np.arange(8,19,2, dtype=int)
    competing_indices_book_1 = np.arange(9,20,2, dtype=int)
    competing_indcies = np.arange(8,20,1,dtype=int)
    #generate pseudo random test and validation indices - where test indices are balanced between the books
    random_generator = np.random.default_rng(seed=random_state)

    book_0_test = random_generator.choice(competing_indices_book_0, size=3, replace=False)
    book_1_test = random_generator.choice(competing_indices_book_1, size=3, replace=False)
    test_indices = np.concatenate((book_0_test, book_1_test), axis=0)

    possible_val_indices = np.setdiff1d(competing_indcies, test_indices)
    val_indices = random_generator.choice(possible_val_indices, size=6, replace=False)

    #convert to list of arrays - needed this way by evaluator class
    test_indices = [np.array([i]) for i in test_indices.tolist()]
    val_indices = [np.array([i]) for i in val_indices.tolist()]

    return test_indices, val_indices

##To Do: move methods to class 
def tune_lrs(subject, cnn_hyperparameters, cnn_train_parameters, data_file, checkpoint_path, train_trials, val_trials, n_optuna_trials=20):

    del cnn_train_parameters['lr']

    def cnn_objective(trial):
        #lr = trial.suggest_float('lr', 1e-8, 1e-1)
        lr =  trial.suggest_float('lr', 1e-8, 1e-1, log=True)

        print('>', lr)
        correlation, best_state_dict = train_dnn(subject_string=subject, checkpoint_path=checkpoint_path, train_indices=train_trials, val_indices = val_trials, workers=0, lr=lr, **cnn_train_parameters, **cnn_hyperparameters)
        torch.save(best_state_dict, os.path.join(checkpoint_path, 'best_model_op_trial' + str(trial.number) + '.ckpt'))
        return correlation
    
    #tpesampler = optuna.samplers.TPESampler()
    gridsampler = optuna.samplers.GridSampler({"lr": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]})
    cnn_pruner = optuna.pruners.MedianPruner(n_startup_trials=np.infty)

    cnn_study = optuna.create_study(
        direction="maximize",
        sampler=gridsampler,
        pruner=cnn_pruner,
        study_name=f'cnn_lr_search_P{subject}'
    )

    cnn_study.optimize(cnn_objective, n_trials=n_optuna_trials)
    cnn_summary = cnn_study.trials_dataframe()
    cnn_summary.to_csv(os.path.join(checkpoint_path, f'cnn_lr_search_P{subject}.csv'))
    
    cnn_train_parameters['lr'] = cnn_study.best_trial.params['lr']

    pickle.dump(cnn_train_parameters, open(os.path.join(checkpoint_path, f'opt_cnn_train_params_P{subject}.pkl'), 'wb'))
    pickle.dump(cnn_study, open(os.path.join(checkpoint_path, f'optuna_cnn_lr_study_P{subject}.pk'), 'wb'))

def tune_lrs_and_input_lengths(subject, cnn_hyperparameters, cnn_train_parameters, data_file, checkpoint_path, train_trials, val_trials, n_optuna_trials=100):
    del cnn_train_parameters['lr']
    del cnn_hyperparameters['input_length']
    del cnn_train_parameters['window_size']
    del cnn_hyperparameters['F1']
    del cnn_hyperparameters['F2']

    def cnn_objective(trial):
        #lr = trial.suggest_float('lr', 1e-8, 1e-1)
        lr =  trial.suggest_float('lr', 1e-6, 1e-2, log=True)
        input_length = trial.suggest_int('input_length', 50, 130, step = 10)
        F1 = trial.suggest_categorical('F1', [2,4,8])
        F2 =cnn_hyperparameters['D'] * F1
        #input_length = 70
        correlation, best_state_dict = train_dnn(subject_string=subject, checkpoint_path=checkpoint_path, optuna_trial=trial, train_indices=train_trials, val_indices = val_trials, workers=0, lr=lr, input_length = input_length, F1 = F1, F2 = F2, **cnn_train_parameters, **cnn_hyperparameters)
        torch.save(best_state_dict, os.path.join(checkpoint_path, 'best_model_op_trial' + str(trial.number) + '.ckpt'))
        return correlation
    
    tpesampler = optuna.samplers.TPESampler()
    #gridsampler = optuna.samplers.GridSampler({"lr": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]})
    cnn_pruner = optuna.pruners.MedianPruner(n_startup_trials=np.infty)

    cnn_study = optuna.create_study(
        direction="maximize",
        sampler=tpesampler,
        pruner=cnn_pruner,
        study_name=f'cnn_lr_search_P{subject}'
    )

    cnn_study.optimize(cnn_objective, n_trials=n_optuna_trials)
    cnn_summary = cnn_study.trials_dataframe()
    cnn_summary.to_csv(os.path.join(checkpoint_path, f'cnn_lr_search_P{subject}.csv'))
    
    cnn_train_parameters['lr'] = cnn_study.best_trial.params['lr']
    cnn_hyperparameters['input_length'] = cnn_study.best_trial.params['input_length']
    cnn_train_parameters['window_size'] = cnn_study.best_trial.params['input_length']
    cnn_hyperparameters['F1'] = cnn_study.best_trial.params['F1']
    cnn_hyperparameters['F2'] = cnn_hyperparameters['D'] * cnn_hyperparameters['F1']

    pickle.dump(cnn_train_parameters, open(os.path.join(checkpoint_path, f'opt_cnn_train_params_P{subject}.pkl'), 'wb'))
    pickle.dump(cnn_study, open(os.path.join(checkpoint_path, f'optuna_cnn_lr_study_P{subject}.pk'), 'wb'))


def cnn_cross_val_training(subject_string, cnn_hyperparameters, cnn_train_params, model_id):
    """
    Trains for one subject 6 CNN models with different left-out training sets

    Args:
        subject_string (string): subject identifier e.g. "108"
        cnn_hyperparameters (dict): model hyperarameters of CNN. e.g. 
        {"dropout_rate": 0.20,"F1": 8,"D": 8,"F2": 64, "input_length": window_size_training, "num_input_channels": 31}

        cnn_train_param_list n(list): list of training parameters training hyperparameters. e.g.
        {"data_dir": data_dir, "lr": 0.01, "batch_size": 256, "weight_decay": 1e-08, "epochs": 1, "window_size": window_size_training}

        model_id (string): three-digit identfier of the used model parameterization e.g. "013"
    """

    base_dir = base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    data_dir = cnn_train_params['data_dir']
    
    #these are trials their naming starts from 1.... (legacy naming from experiment)
    #trial 9 is the first trial of competing speaker scenario
    test_trials_list = [np.array([i,i+1]) for i in range(9,21,2)]
    val_trials_list = list(np.roll(np.array(test_trials_list),2))
    #list of train_indices exclude train and val indices - weird -1 because index = trial - 1
    train_trials_list = [np.delete(np.linspace(1,20,20, dtype=int), np.hstack((test_trials_list[i] - 1, val_trials_list[i] -1))) for i in range(0,len(test_trials_list))]

    for train_trails, val_trails, test_trials in zip(train_trials_list, val_trials_list, test_trials_list):
        check_point_path = os.path.join(base_dir,'models','cnn', 'checkpoints', model_id, subject_string, str(test_trials[0]) + '_' + str(test_trials[1]))

        if not os.path.exists(check_point_path):
            os.makedirs(check_point_path)
        
        model_param_file = os.path.join(check_point_path, 'model_param_' + model_id + '.pkl')
        train_param_file = os.path.join(check_point_path, 'train_param_' + model_id + '.pkl')

        with open(model_param_file,'wb') as fp:
            pickle.dump(cnn_hyperparameters, fp)
        fp.close()

        with open(train_param_file,'wb') as fp:
            pickle.dump(cnn_train_params, fp)
        fp.close()

        best_correlation, best_state_dict = train_dnn(subject_string=subject_string, checkpoint_path=check_point_path, train_indices=train_trails, val_indices = val_trails, workers=0, **cnn_train_params, **cnn_hyperparameters)
        torch.save(best_state_dict, os.path.join(check_point_path, 'best_model.ckpt'))

        print(f'Subject: {subject_string} test_trials: {test_trials}')

def lr_study(model_id:str):

    base_dir = base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    data_dir = os.path.join(base_dir, "data/processed/ci_l_1,1_h_32,50_out_125_114.hdf5")
    window_size_training = 50

    cnn_hyperparameters = {"dropout_rate": 0.2,"F1": 8,"D": 8,"F2": 64, "input_length": window_size_training, "num_input_channels": 31}
    cnn_train_params = {"data_dir": data_dir, "lr": 0.0001, "batch_size": 256, "weight_decay": 1e-08, "epochs": 20, "window_size": window_size_training}

    test_trials = np.array([13,14])
    val_trials = np.array([11,12])
    train_trials = np.delete(np.linspace(1,20,20, dtype=int), np.hstack((test_trials - 1, val_trials -1)))
    n_optuna_trials = 15

    with h5py.File(data_dir, 'r') as f:
        subject_strings = list(f['eeg'].keys())
    f.close()

    for subject_string in subject_strings:
        check_point_path = os.path.join(base_dir,'models','cnn', 'checkpoints', model_id, subject_string, str(test_trials[0]) + '_' + str(test_trials[1]))
        #check if path exists and create it if not
        if not os.path.exists(check_point_path):
            os.makedirs(check_point_path)
        tune_lrs(subject=subject_string, 
                cnn_hyperparameters=cnn_hyperparameters, 
                cnn_train_parameters=cnn_train_params, 
                data_file=data_dir, 
                checkpoint_path=check_point_path, 
                train_trials=train_trials, 
                val_trials=val_trials, 
                n_optuna_trials= n_optuna_trials)
        print(f"Learning rate tuning for subject {subject_string} finished.")

def lr_input_length_study(model_id:str):
    base_dir = base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    data_dir = os.path.join(base_dir, "data/processed/ci_l_1,1_h_32,50_out_125_114.hdf5")
    window_size_training = 50

    cnn_hyperparameters = {"dropout_rate": 0.2,"F1": 8,"D": 8,"F2": 64, "input_length": window_size_training, "num_input_channels": 31}
    cnn_train_params = {"data_dir": data_dir, "lr": 0.0001, "batch_size": 256, "weight_decay": 1e-08, "epochs": 30, "window_size": window_size_training}

    #using two single speaker and two competing speaker trials for testing each from different audio books
    test_trials = np.array([6,7,18,20])
    val_trials = np.array([10,11])

    train_trials = np.delete(np.linspace(1,20,20, dtype=int), np.hstack((test_trials - 1, val_trials -1)))
    n_optuna_trials = 100

    with h5py.File(data_dir, 'r') as f:
        subject_strings = list(f['eeg'].keys())
    f.close()

    for subject_string in subject_strings:
        check_point_path = os.path.join(base_dir,'models','cnn', 'checkpoints', model_id, subject_string, str(test_trials[0]) + '_' + str(test_trials[1]) + str(test_trials[2]) + '_' + str(test_trials[3]) )
        #check if path exists and create it if not
        if not os.path.exists(check_point_path):
            os.makedirs(check_point_path)
        tune_lrs_and_input_lengths(subject=subject_string, 
                cnn_hyperparameters=cnn_hyperparameters, 
                cnn_train_parameters=cnn_train_params, 
                data_file=data_dir, 
                checkpoint_path=check_point_path, 
                train_trials=train_trials, 
                val_trials=val_trials, 
                n_optuna_trials= n_optuna_trials)
        print(f"Learning rate tuning for subject {subject_string} finished.")


def get_optimal_lr(base_dir, subject, model_id):
    """
    Getting optimal learning rate from optuna study for given subject and model_id.
    Learning rate study must be finished before.

    Args:
        base_dir (string): base directory of the project
        subject (string): subject identifier e.g. "108"
        model_id (string): three-digit identfier of the used model parameterization e.g. "013"

    Returns:
        float: optimal learning rate
    """
    optuna_study_path = os.path.join(base_dir, "models/cnn/checkpoints", model_id, subject, "13_14", "optuna_cnn_lr_study_P" + subject + ".pk")
    optuna_study = pickle.load(open(optuna_study_path, "rb"))
    params = optuna_study.best_params
    return params['lr']

def eval_lr_study(optuna_study_id, output_model_id, data_dir):

    #load from study in future
    cnn_hyperparameters = {"dropout_rate": 0.2,"F1": 8,"D": 8,"F2": 64, "input_length": 50, "num_input_channels": 31}
    
    base_dir = base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    
    with h5py.File(data_dir, 'r') as f:
        subject_strings = list(f['eeg'].keys())
    f.close()

    for subject in subject_strings:
        train_params_path = os.path.join(base_dir, "models/cnn/checkpoints", optuna_study_id, subject, "13_14", "opt_cnn_train_params_P" + subject + ".pkl")
        train_params = pickle.load(open(train_params_path, "rb"))
        cnn_cross_val_training(subject_string=subject, cnn_hyperparameters=cnn_hyperparameters, cnn_train_params=train_params, model_id=output_model_id)


def ridge_reference_cnn():
    """
    Training CNN models as reference for comparison with ridge regression.
    """
    base_dir = base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    data_dir = os.path.join(base_dir, "data/processed/ci_l_1,1_h_32,32_out_125_116_incl_ica_onset.hdf5")

    cnn_hyperparameters = {"dropout_rate": 0.2, "F1": 8,"D": 8,"F2": 64, "input_length": 90, "num_input_channels": 31}
    cnn_train_params = {"data_dir": data_dir, "lr": 0.0001, "batch_size": 256, "weight_decay": 1e-08, "epochs": 1, "early_stopping_patience": 5}

    random_state = 672
    competing_indices_book_0 = np.arange(8,19,2, dtype=int)
    competing_indices_book_1 = np.arange(9,20,2, dtype=int)
    competing_indcies = np.arange(8,20,1,dtype=int)
    #generate pseudo random test and validation indices - where test indices are balanced between the books
    random_generator = np.random.default_rng(seed=random_state)

    book_0_test = random_generator.choice(competing_indices_book_0, size=3, replace=False)
    book_1_test = random_generator.choice(competing_indices_book_1, size=3, replace=False)
    test_indices = np.concatenate((book_0_test, book_1_test), axis=0)

    possible_val_indices = np.setdiff1d(competing_indcies, test_indices)
    val_indices = random_generator.choice(possible_val_indices, size=6, replace=False)

    #convert to list of arrays - needed this way by evaluator class
    test_indices = [np.array([i]) for i in test_indices.tolist()]
    val_indices = [np.array([i]) for i in val_indices.tolist()]

    with h5py.File(data_dir, 'r') as f:
        subject_strings = list(f['eeg'].keys())
    f.close()

    model_id = '009'

    for subject_string in subject_strings:
        for val_ind, test_ind in zip(val_indices, test_indices):
            train_indices = np.delete(np.linspace(0,19,20, dtype=int), np.hstack((test_ind, val_ind)))
            check_point_path = os.path.join(base_dir,'models','cnn', 'checkpoints', model_id, subject_string, 'test_ind_' + str(test_ind.item()) )
            #check if path exists and create it if not
            if not os.path.exists(check_point_path):
                os.makedirs(check_point_path)
            
            best_correlation, best_state_dict = train_dnn(subject_string=subject_string, checkpoint_path= check_point_path, train_indices=train_indices, val_indices = val_ind, workers=0, mdl_checkpointing=True, **cnn_train_params, **cnn_hyperparameters)

    #smaller batchsize
    model_id = '010'
    cnn_hyperparameters = {"dropout_rate": 0.2, "F1": 8,"D": 8,"F2": 64, "input_length": 90, "num_input_channels": 31}
    cnn_train_params = {"data_dir": data_dir, "lr": 0.0001, "batch_size": 128, "weight_decay": 1e-08, "epochs": 30, "early_stopping_patience": 5}

    for subject_string in subject_strings:
        for val_ind, test_ind in zip(val_indices, test_indices):
            train_indices = np.delete(np.linspace(0,19,20, dtype=int), np.hstack((test_ind, val_ind)))
            check_point_path = os.path.join(base_dir,'models','cnn', 'checkpoints', model_id, subject_string, 'test_ind_' + str(test_ind.item()) )
            #check if path exists and create it if not

            if not os.path.exists(check_point_path):
                os.makedirs(check_point_path)
            
            best_correlation, best_state_dict = train_dnn(subject_string=subject_string, checkpoint_path= check_point_path, train_indices=train_indices, val_indices = val_ind, workers=0, mdl_checkpointing=True, **cnn_train_params, **cnn_hyperparameters)

def cnn_diff_corr_loss():
    """
    First try to train CNN with correlation difference loss function.
    """
    base_dir = base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    data_dir = os.path.join(base_dir, "data/processed/ci_l_1,1_h_32,32_out_125_116_incl_ica_onset.hdf5")

    cnn_hyperparameters = {"dropout_rate": 0.2, "F1": 8,"D": 8,"F2": 64, "input_length": 90, "num_input_channels": 31}
    cnn_train_params = {"data_dir": data_dir, "lr": 0.0001, "batch_size": 256, "weight_decay": 1e-08, "epochs": 1, "early_stopping_patience": 15, "loss_fnc": "corr_diff"}

    test_indices, val_indices = generate_test_and_val_indices()

    with h5py.File(data_dir, 'r') as f:
        subject_strings = list(f['eeg'].keys())
    f.close()

    model_id = '011'
    if args.subj_0 is not None and args.subj_1 is not None:
        subject_strings = subject_strings[args.subj_0:args.subj_1]
    
    #print(f'subject strings: {subject_strings}')

    for subject_string in subject_strings:
        for val_ind, test_ind in zip(val_indices, test_indices):
            train_indices = np.delete(np.linspace(0,19,20, dtype=int), np.hstack((test_ind, val_ind)))
            check_point_path = os.path.join(base_dir,'models','cnn', 'checkpoints', model_id, subject_string, 'test_ind_' + str(test_ind.item()) )
            #check if path exists and create it if not
            if not os.path.exists(check_point_path):
                os.makedirs(check_point_path)
            
            best_correlation, best_state_dict = train_dnn(subject_string=subject_string, checkpoint_path= check_point_path, train_indices=train_indices, val_indices = val_ind, workers=0, mdl_checkpointing=True, **cnn_train_params, **cnn_hyperparameters)

def train_basemodels():
    """
    Training baseline models leaving several subjects out for finetuning
    """
    #train baseline model for subjects 
    #for subjects 111,112,113,114

    window_size_training = 100

    cnn_hyperparameters = {"dropout_rate": 0.35,"F1": 8,"D": 8,"F2": 64, "input_length": window_size_training, "num_input_channels": 31}
    cnn_train_params = {"data_dir": data_dir, "lr": 0.0001, "batch_size": 1024, "weight_decay": 1e-08, "epochs": 35, "early_stopping_patience": 15}

    trial_indices = np.arange(0,20,1).tolist()


    if args.model_id == '012':
        print('Entering model 012 training')
        #holding out subjects 112,113,114 for finetuning
        subjects = np.arange(101,112,1)
        #drop three subjects radomly
        subject_strings = [str(subject) for subject in subjects]
        model_id = '012'
        check_point_path = os.path.join(base_dir,'models','cnn', 'checkpoints', model_id, 'basemodel')
        #check if path exists and create it if not
        if not os.path.exists(check_point_path):
            os.makedirs(check_point_path)
        best_correlation, best_state_dict = train_dnn(subject_string_train=subject_strings, checkpoint_path = check_point_path, train_indices = trial_indices, 
                                            subject_string_val = ['112','113'], val_indices = [15,16,17,18], workers=0, mdl_checkpointing=True, **cnn_train_params, **cnn_hyperparameters)

    elif args.model_id == '013':
        #holding out subjects 101,102,103 for finetuning
        subjects = np.arange(104,115,1)
        subject_strings = [str(subject) for subject in subjects]
        model_id = '013'
        check_point_path = os.path.join(base_dir,'models','cnn', 'checkpoints', model_id, 'basemodel')
        #check if path exists and create it if not
        if not os.path.exists(check_point_path):
            os.makedirs(check_point_path)
        best_correlation, best_state_dict = train_dnn(subject_string_train=subject_strings, checkpoint_path = check_point_path, train_indices = trial_indices, 
                                                subject_string_val = ['101','102'], val_indices = [15,16,17,18], workers=0, mdl_checkpointing=True, **cnn_train_params, **cnn_hyperparameters)
    
    elif args.model_id == '020':
        print('Entering model 020 basemodel training')
        #holding out subjects 112,113,114 for finetuning
        #subjects = np.arange(101,112,1)
        #drop three subjects radomly
        subject_strings = ['101', '102', '103', '107', '108', '109', '110', '111', '112', '113', '114']
        model_id = '020'
        check_point_path = os.path.join(base_dir,'models','cnn', 'checkpoints', model_id, 'basemodel')
        #check if path exists and create it if not
        if not os.path.exists(check_point_path):
            os.makedirs(check_point_path)
        best_correlation, best_state_dict = train_dnn(subject_string_train=subject_strings, checkpoint_path = check_point_path, train_indices = trial_indices, 
                                            subject_string_val = ['104','105'], val_indices = [15,16,17,18], workers=0, mdl_checkpointing=True, **cnn_train_params, **cnn_hyperparameters)

    elif args.model_id == '021':
        print('Entering model 021 basemodel training')
        #holding out subjects 101,102,103 for finetuning
        #subjects = np.arange(104,115,1)
        subject_strings = subject_strings = ['101', '102', '103', '104', '105', '106', '111', '112', '113', '114']
        model_id = '021'
        check_point_path = os.path.join(base_dir,'models','cnn', 'checkpoints', model_id, 'basemodel')
        #check if path exists and create it if not
        if not os.path.exists(check_point_path):
            os.makedirs(check_point_path)
        best_correlation, best_state_dict = train_dnn(subject_string_train=subject_strings, checkpoint_path = check_point_path, train_indices = trial_indices, 
                                                subject_string_val = ['107','108'], val_indices = [15,16,17,18], workers=0, mdl_checkpointing=True, **cnn_train_params, **cnn_hyperparameters)


    elif args.model_id == '021':
        print('Entering model 021 basemodel training')
        #holding out subjects 101,102,103 for finetuning
        #subjects = np.arange(104,115,1)
        subject_strings = subject_strings = ['101', '102', '103', '104', '105', '106', '111', '112', '113', '114']
        model_id = '021'
        check_point_path = os.path.join(base_dir,'models','cnn', 'checkpoints', model_id, 'basemodel')
        #check if path exists and create it if not
        if not os.path.exists(check_point_path):
            os.makedirs(check_point_path)
        best_correlation, best_state_dict = train_dnn(subject_string_train=subject_strings, checkpoint_path = check_point_path, train_indices = trial_indices, 
                                                subject_string_val = ['107','108'], val_indices = [15,16,17,18], workers=0, mdl_checkpointing=True, **cnn_train_params, **cnn_hyperparameters)

    elif args.model_id == '028':
        print('Entering model 028 basemodel training')
        #holding out subjects 101,102,103 for finetuning
        #subjects = np.arange(104,115,1)
        subject_strings = subject_strings = ['101', '102', '103', '104', '105', '106', '107', '108', '112', '113', '114']
        model_id = '028'
        check_point_path = os.path.join(base_dir,'models','cnn', 'checkpoints', model_id, 'basemodel')
        #check if path exists and create it if not
        if not os.path.exists(check_point_path):
            os.makedirs(check_point_path)
        best_correlation, best_state_dict = train_dnn(subject_string_train=subject_strings, checkpoint_path = check_point_path, train_indices = trial_indices, 
                                                subject_string_val = ['109','110'], val_indices = [15,16,17,18], workers=0, mdl_checkpointing=True, **cnn_train_params, **cnn_hyperparameters)

def tune_overfit_params(train_subjects, train_indices, val_subjects, val_indices, cnn_hyperparameters, cnn_train_parameters, checkpoint_path, model_id, n_optuna_trials=100):
    del cnn_train_parameters['lr']
    del cnn_train_parameters['batch_size']
    del cnn_hyperparameters['dropout_rate']
    del cnn_train_parameters['weight_decay']

    checkpoint_path = checkpoint_path


    def cnn_objective(trial):
        lr =  trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [512, 1024, 2048, 4096])
        weight_decay = trial.suggest_float('weight_decay', 1e-9, 1e-4, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.7)
        optimizer = trial.suggest_categorical('optimizer_handle', ['NAdam', 'Adam'])
        if optimizer == 'NAdam':
            optimizer_handle = NAdam
        elif optimizer == 'Adam':
            optimizer_handle = Adam
        cnn_hyperparameters['dropout_rate'] = dropout_rate


        #get optuna trial number
        trial_number = trial.number

        print(f'Optuna trial number: {trial_number}')

        nonlocal checkpoint_path
        checkpoint_path_opt = os.path.join(checkpoint_path, 'optuna_n_' + str(trial_number))
        if not os.path.exists(checkpoint_path_opt):
            os.makedirs(checkpoint_path_opt)
        
        correlation, best_state_dict = train_dnn(subject_string_train=train_subjects, train_indices=train_indices,
                                                 subject_string_val=val_subjects, val_indices=val_indices,
                                                  checkpoint_path=checkpoint_path_opt,
                                                 optuna_trial=trial,
                                                 workers=0, lr=lr, weight_decay=weight_decay, batch_size = batch_size,
                                                 optimizer_handle=optimizer_handle,
                                                  **cnn_train_parameters, **cnn_hyperparameters)
        torch.save(best_state_dict, os.path.join(checkpoint_path_opt, 'best_model_op_trial' + str(trial.number) + '.ckpt'))
        return correlation
    
    tpesampler = optuna.samplers.TPESampler()
    #gridsampler = optuna.samplers.GridSampler()
    cnn_pruner = optuna.pruners.MedianPruner(n_startup_trials=np.infty)

    #allows to parallelize optuna studies
    storage = JournalStorage(JournalFileStorage(os.path.join(checkpoint_path, "optuna-journal.log")))

    cnn_study = optuna.create_study(
        direction="maximize",
        sampler=tpesampler,
        pruner=cnn_pruner,
        study_name=f'cnn_lr_search_{model_id}',
        storage=storage,
        load_if_exists=True
    )

    cnn_study.optimize(cnn_objective, n_trials=n_optuna_trials)
    cnn_summary = cnn_study.trials_dataframe()
    cnn_summary.to_csv(os.path.join(checkpoint_path, f'cnn_lr_search_P{model_id}.csv'))
    
    cnn_train_parameters['lr'] = cnn_study.best_trial.params['lr']
    cnn_train_parameters['batch_size'] = cnn_study.best_trial.params['batch_size']
    cnn_train_parameters['weight_decay'] = cnn_study.best_trial.params['weight_decay']
    cnn_hyperparameters['dropout_rate'] = cnn_study.best_trial.params['dropout_rate']
    
    pickle.dump(cnn_train_parameters, open(os.path.join(checkpoint_path, f'opt_cnn_train_params_{model_id}.pkl'), 'wb'))
    pickle.dump(cnn_study, open(os.path.join(checkpoint_path, f'optuna_cnn_lr_study_{model_id}.pk'), 'wb'))

def overfit_param_study():
    base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    data_dir = os.path.join(base_dir, "data/processed/ci_l_1,1_h_32,32_out_125_116_incl_ica_onset.hdf5")

    window_size_training = 100

    cnn_hyperparameters = {"dropout_rate": 0.35,"F1": 8,"D": 8,"F2": 64, "input_length": window_size_training, "num_input_channels": 31}
    cnn_train_params = {"data_dir": data_dir, "lr": 0.0001, "batch_size": 1024, "weight_decay": 1e-08, "epochs": 30, "early_stopping_patience": 8}

    train_indices = np.arange(0,20,1).tolist()

    n_optuna_trials = 5

    if args.model_id == '014':
        print('Entering parameter study 014')
        #holding out subjects 112,113,114 for finetuning
        subjects = np.arange(101,112,1)
        #drop three subjects radomly
        subject_strings = [str(subject) for subject in subjects]
        model_id = '014'
        check_point_path = os.path.join(base_dir,'models','cnn', 'checkpoints', model_id)
        #check if path exists and create it if not
        if not os.path.exists(check_point_path):
            os.makedirs(check_point_path)
        tune_overfit_params(train_subjects=subject_strings, train_indices=train_indices, val_subjects=['112','114'], val_indices=[15,16,17,18], 
                            cnn_hyperparameters=cnn_hyperparameters, cnn_train_parameters=cnn_train_params, checkpoint_path=check_point_path, model_id=model_id, n_optuna_trials=n_optuna_trials)

    elif args.model_id == '015':
        #holding out subjects 101,102,103 for finetuning
        subjects = np.arange(104,115,1)
        subject_strings = [str(subject) for subject in subjects]
        model_id = '015'
        check_point_path = os.path.join(base_dir,'models','cnn', 'checkpoints', model_id)
        #check if path exists and create it if not
        if not os.path.exists(check_point_path):
            os.makedirs(check_point_path)
        tune_overfit_params(train_subjects=subject_strings, train_indices=train_indices, val_subjects=['101','102'], val_indices=[15,16,17,18], 
                            cnn_hyperparameters=cnn_hyperparameters, cnn_train_parameters=cnn_train_params, checkpoint_path=check_point_path, model_id=model_id, n_optuna_trials=n_optuna_trials)

def load_pretrained(model_id):
    base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    model_dir = os.path.join(base_dir, 'models', 'cnn', 'checkpoints', model_id, 'basemodel')
    model_kwargs = pickle.load(open(os.path.join(model_dir, 'model_kwargs.pkl'), 'rb'))

    state_dict = torch.load(os.path.join(model_dir, 'best_model.ckpt'), map_location=device)
    model = CNN(**model_kwargs)
    model.load_state_dict(state_dict)

    return model

def create_train_val_test_indices():
    random_state = 672
    competing_indices_book_0 = np.arange(8,19,2, dtype=int)
    competing_indices_book_1 = np.arange(9,20,2, dtype=int)
    competing_indcies = np.arange(8,20,1,dtype=int)
    #generate pseudo random test and validation indices - where test indices are balanced between the books
    random_generator = np.random.default_rng(seed=random_state)

    book_0_test = random_generator.choice(competing_indices_book_0, size=3, replace=False)
    book_1_test = random_generator.choice(competing_indices_book_1, size=3, replace=False)
    test_indices = np.concatenate((book_0_test, book_1_test), axis=0)

    possible_val_indices = np.setdiff1d(competing_indcies, test_indices)
    val_indices = random_generator.choice(possible_val_indices, size=6, replace=False)

    #convert to list of arrays - needed this way by evaluator class
    test_indices = [np.array([i]) for i in test_indices.tolist()]
    val_indices = [np.array([i]) for i in val_indices.tolist()]
    train_indices = [np.delete(np.arange(0,20), np.hstack((test_indices[i],val_indices[i]))) for i in range(0,len(test_indices))]

    return train_indices, val_indices, test_indices

def fine_tune_models():
    base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    data_dir = os.path.join(base_dir, "data/processed/ci_l_1,1_h_32,32_out_125_116_incl_ica_onset.hdf5")

    cnn_train_params = {"data_dir": data_dir, "lr": 0.0001, "batch_size": 1024, "weight_decay": 1e-08, "epochs": 15, "early_stopping_patience": 7}
    cnn_hyperparameters = {"input_length": 100}

    train_indices_list, val_indices_list, test_indices_list = create_train_val_test_indices()

    if args.model_id == '012':
        pretrained_model = load_pretrained('012')
        #model id for fine tuning
        model_id = '016'
        #subjects left out for finetuning
        for subject in ['112','113','114']:
            for train_indices, val_indices, test_indices in zip(train_indices_list, val_indices_list, test_indices_list):

                checkpoint_path = os.path.join(base_dir,'models','cnn', 'checkpoints', model_id, subject, 'test_ind_' + str(test_indices[0].item()))
                #check if path exists and create it if not
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)
                
                _, _ = train_dnn(subject_string_train = subject, checkpoint_path = checkpoint_path, train_indices = train_indices, 
                                            subject_string_val = [subject], val_indices = val_indices, workers=0, mdl_checkpointing=True, use_pretrained=True, pretrained_model=pretrained_model, **cnn_train_params, **cnn_hyperparameters)
    
    elif args.model_id == '013':
        pretrained_model = load_pretrained('013')
        #model id for fine tuning
        model_id = '017'
        #subjects left out for finetuning
        for subject in ['101','102','103']:
            for train_indices, val_indices, test_indices in zip(train_indices_list, val_indices_list, test_indices_list):
                checkpoint_path = os.path.join(base_dir,'models','cnn', 'checkpoints', model_id, subject, 'test_ind_' + str(test_indices[0].item()))
                #check if path exists and create it if not
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)
                _, _ = train_dnn(subject_string_train = subject, checkpoint_path = checkpoint_path, train_indices = train_indices, 
                                            subject_string_val = [subject], val_indices = val_indices, workers=0, mdl_checkpointing=True, use_pretrained=True, pretrained_model=pretrained_model, **cnn_train_params, **cnn_hyperparameters)

def fine_tune_models_small_batch():
    """
    Redo finetuning with smaller batch size and learning rate, hoping for less overfitting behaviour.
    
    """
    base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    data_dir = os.path.join(base_dir, "data/processed/ci_l_1,1_h_32,32_out_125_116_incl_ica_onset.hdf5")

    cnn_train_params = {"data_dir": data_dir, "lr": 0.00005, "batch_size": 256, "weight_decay": 1e-08, "epochs": 15, "early_stopping_patience": 7}
    cnn_hyperparameters = {"input_length": 100}

    train_indices_list, val_indices_list, test_indices_list = create_train_val_test_indices()

    if args.model_id == '012':
        pretrained_model = load_pretrained('012')
        #model id for fine tuning
        model_id = '018'
        #subjects left out for finetuning
        for subject in ['112','113','114']:
            for train_indices, val_indices, test_indices in zip(train_indices_list, val_indices_list, test_indices_list):

                checkpoint_path = os.path.join(base_dir,'models','cnn', 'checkpoints', model_id, subject, 'test_ind_' + str(test_indices[0].item()))
                #check if path exists and create it if not
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)
                
                _, _ = train_dnn(subject_string_train = subject, checkpoint_path = checkpoint_path, train_indices = train_indices, 
                                            subject_string_val = [subject], val_indices = val_indices, workers=0, mdl_checkpointing=True, use_pretrained=True, pretrained_model=pretrained_model, **cnn_train_params, **cnn_hyperparameters)
    
    elif args.model_id == '013':
        pretrained_model = load_pretrained('013')
        #model id for fine tuning
        model_id = '019'
        #subjects left out for finetuning
        for subject in ['101','102','103']:
            for train_indices, val_indices, test_indices in zip(train_indices_list, val_indices_list, test_indices_list):
                checkpoint_path = os.path.join(base_dir,'models','cnn', 'checkpoints', model_id, subject, 'test_ind_' + str(test_indices[0].item()))
                #check if path exists and create it if not
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)
                _, _ = train_dnn(subject_string_train = subject, checkpoint_path = checkpoint_path, train_indices = train_indices, 
                                            subject_string_val = [subject], val_indices = val_indices, workers=0, mdl_checkpointing=True, use_pretrained=True, pretrained_model=pretrained_model, **cnn_train_params, **cnn_hyperparameters)
    #the model id of the base model
    elif args.model_id == '020':
        pretrained_model = load_pretrained('020')
        #model id for fine tuning
        model_id = '022'
        #subjects left out for finetuning
        for subject in ['104','105','106']:
            for train_indices, val_indices, test_indices in zip(train_indices_list, val_indices_list, test_indices_list):

                checkpoint_path = os.path.join(base_dir,'models','cnn', 'checkpoints', model_id, subject, 'test_ind_' + str(test_indices[0].item()))
                #check if path exists and create it if not
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)
                
                _, _ = train_dnn(subject_string_train = subject, checkpoint_path = checkpoint_path, train_indices = train_indices, 
                                            subject_string_val = [subject], val_indices = val_indices, workers=0, mdl_checkpointing=True, use_pretrained=True, pretrained_model=pretrained_model, **cnn_train_params, **cnn_hyperparameters)
    
    elif args.model_id == '021':
        pretrained_model = load_pretrained('021')
        #model id for fine tuning
        model_id = '023'
        #subjects left out for finetuning
        for subject in ['107','108','109', '110']:
            for train_indices, val_indices, test_indices in zip(train_indices_list, val_indices_list, test_indices_list):
                checkpoint_path = os.path.join(base_dir,'models','cnn', 'checkpoints', model_id, subject, 'test_ind_' + str(test_indices[0].item()))
                #check if path exists and create it if not
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)
                _, _ = train_dnn(subject_string_train = subject, checkpoint_path = checkpoint_path, train_indices = train_indices, 
                                            subject_string_val = [subject], val_indices = val_indices, workers=0, mdl_checkpointing=True, use_pretrained=True, pretrained_model=pretrained_model, **cnn_train_params, **cnn_hyperparameters)
    ####for testing
    elif args.model_id == '022':
        pretrained_model = load_pretrained('021')
        #model id for fine tuning
        model_id = '024'
        #subjects left out for finetuning
        for subject in ['107','108','109', '110']:
            for train_indices, val_indices, test_indices in zip(train_indices_list, val_indices_list, test_indices_list):
                checkpoint_path = os.path.join(base_dir,'models','cnn', 'checkpoints', model_id, subject, 'test_ind_' + str(test_indices[0].item()))
                #check if path exists and create it if not
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)
                cnn_train_params['epochs'] = 1
                _, _ = train_dnn(subject_string_train = subject, checkpoint_path = checkpoint_path, train_indices = np.array([0,1]), 
                                            subject_string_val = [subject], val_indices = val_indices, workers=0, mdl_checkpointing=True, use_pretrained=True, pretrained_model=pretrained_model, **cnn_train_params, **cnn_hyperparameters)

def fine_tune_models_small_batch_complete():
    """
    Redo finetuning with smaller batch size and learning rate, hoping for less overfitting behaviour.
    Testing on all competing trials
    """
    base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    data_dir = os.path.join(base_dir, "data/processed/ci_l_1,1_h_32,32_out_125_116_incl_ica_onset.hdf5")

    cnn_train_params = {"data_dir": data_dir, "lr": 0.00005, "batch_size": 256, "weight_decay": 1e-08, "epochs": 15, "early_stopping_patience": 7}
    cnn_hyperparameters = {"input_length": 100}

    #test on all sets
    test_indices_list = [np.array([i]) for i in range(8,20)]
    val_indices = np.roll(np.array(test_indices_list), 1).reshape(-1).tolist()
    val_indices_list = [np.array([i]) for i in val_indices]

    train_indices_list = [np.delete(np.arange(0,20), np.hstack((test_indices_list[i],val_indices_list[i]))) for i in range(0,len(test_indices_list))]

    if args.model_id == '024':
        pretrained_model = load_pretrained('013')
        #model id for fine tuning
        model_id = '024'
        #subjects left out for finetuning
        for subject in ['101','102','103']:
            for train_indices, val_indices, test_indices in zip(train_indices_list, val_indices_list, test_indices_list):
                checkpoint_path = os.path.join(base_dir,'models','cnn', 'checkpoints', model_id, subject, 'test_ind_' + str(test_indices[0].item()))
                #check if path exists and create it if not
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)
                _, _ = train_dnn(subject_string_train = subject, checkpoint_path = checkpoint_path, train_indices = train_indices, 
                                            subject_string_val = [subject], val_indices = val_indices, workers=0, mdl_checkpointing=True, use_pretrained=True, pretrained_model=pretrained_model, **cnn_train_params, **cnn_hyperparameters)
    #the model id of the base model
    elif args.model_id == '025':
        pretrained_model = load_pretrained('020')
        #model id for fine tuning
        model_id = '025'
        #subjects left out for finetuning
        for subject in ['104','105','106']:
            for train_indices, val_indices, test_indices in zip(train_indices_list, val_indices_list, test_indices_list):

                checkpoint_path = os.path.join(base_dir,'models','cnn', 'checkpoints', model_id, subject, 'test_ind_' + str(test_indices[0].item()))
                #check if path exists and create it if not
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)
                
                _, _ = train_dnn(subject_string_train = subject, checkpoint_path = checkpoint_path, train_indices = train_indices, 
                                            subject_string_val = [subject], val_indices = val_indices, workers=0, mdl_checkpointing=True, use_pretrained=True, pretrained_model=pretrained_model, **cnn_train_params, **cnn_hyperparameters)
    
    elif args.model_id == '026':
        pretrained_model = load_pretrained('021')
        #model id for fine tuning
        model_id = '026'
        #subjects left out for finetuning
        for subject in ['107','108','109', '110']:
            for train_indices, val_indices, test_indices in zip(train_indices_list, val_indices_list, test_indices_list):
                checkpoint_path = os.path.join(base_dir,'models','cnn', 'checkpoints', model_id, subject, 'test_ind_' + str(test_indices[0].item()))
                #check if path exists and create it if not
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)
                _, _ = train_dnn(subject_string_train = subject, checkpoint_path = checkpoint_path, train_indices = train_indices, 
                                            subject_string_val = [subject], val_indices = val_indices, workers=0, mdl_checkpointing=True, use_pretrained=True, pretrained_model=pretrained_model, **cnn_train_params, **cnn_hyperparameters)

    elif args.model_id == '027':
        pretrained_model = load_pretrained('012')
        #model id for fine tuning
        model_id = '026'
        #subjects left out for finetuning
        for subject in ['112','113','114']:
            for train_indices, val_indices, test_indices in zip(train_indices_list, val_indices_list, test_indices_list):

                checkpoint_path = os.path.join(base_dir,'models','cnn', 'checkpoints', model_id, subject, 'test_ind_' + str(test_indices[0].item()))
                #check if path exists and create it if not
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)
                
                _, _ = train_dnn(subject_string_train = subject, checkpoint_path = checkpoint_path, train_indices = train_indices, 
                                            subject_string_val = [subject], val_indices = val_indices, workers=0, mdl_checkpointing=True, use_pretrained=True, pretrained_model=pretrained_model, **cnn_train_params, **cnn_hyperparameters)

    elif args.model_id == '029':
        pretrained_model = load_pretrained('028')
        print(f'Pretrained model loaded')
        #model id for fine tuning
        model_id = '029'
        #subjects left out for finetuning
        for subject in ['111']:
            for train_indices, val_indices, test_indices in zip(train_indices_list, val_indices_list, test_indices_list):

                checkpoint_path = os.path.join(base_dir,'models','cnn', 'checkpoints', model_id, subject, 'test_ind_' + str(test_indices[0].item()))
                #check if path exists and create it if not
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)
                
                _, _ = train_dnn(subject_string_train = subject, checkpoint_path = checkpoint_path, train_indices = train_indices, 
                                            subject_string_val = [subject], val_indices = val_indices, workers=0, mdl_checkpointing=True, use_pretrained=True, pretrained_model=pretrained_model, **cnn_train_params, **cnn_hyperparameters)

def eval_pretrain_effectiveness():
    """
    Experiment to evaluate the effectiveness of pretraining on the competing trials.
    Subject specific models are trained using same models and training parameters as in the fine tuning experiments, just without pretraining.
    """
    window_size_training = 100
    #test on all sets
    test_indices_list = [np.array([i]) for i in range(8,20)]
    val_indices = np.roll(np.array(test_indices_list), 1).reshape(-1).tolist()

    if args.model_id == '030':
        pretrained_id = '013'
        subjects = ['101','102','103']
        compare_pretraining(subjects=subjects, pretrain_id=pretrained_id)

    #the model id of the base model
    elif args.model_id == '032':
        pretrained_id = '020'
        subjects = ['104','105','106']
        compare_pretraining(subjects=subjects, pretrain_id=pretrained_id)
    
    elif args.model_id == '034':
        pretrained_id = '021'
        subjects = ['107','108','109', '110']
        compare_pretraining(subjects=subjects, pretrain_id=pretrained_id)

    elif args.model_id == '036':
        pretrained_id = '012'
        subjects = ['112','113','114']
        compare_pretraining(subjects=subjects, pretrain_id=pretrained_id)

    elif args.model_id == '038':
        pretrained_id = '028'
        subjects = ['111']
        compare_pretraining(subjects=subjects, pretrain_id=pretrained_id)

def compare_pretraining(subjects, pretrain_id, ica=False, feature='env'):
    """
    Compare the performance of pretrained models to models trained without pretraining.
    For each pretrained model, a subject specific model is trained on the same subjects as the pretrained model with model id increased by one.
    args.model_id is used to give id to the finetuned model, subject specific models get id increased by one.
    watch out when using the method to increase model_ids by two

    Args:
        subjects (list): subjects to compare
        pretrain_id (str): model_id of pretrained model to load for the corresponding subjects
        ica (bool, optional): whether to use ica data. Defaults to False.
        feature (str, optional): feature to use. Defaults to 'env'. Must be either 'env' or 'onset_env'.
    """

    assert feature in ['env', 'onset_env'], 'feature must be either env or onset_env'

    window_size_training = 100

    cnn_train_params = {"data_dir": data_dir, "lr": 0.00005, "batch_size": 256, "weight_decay": 1e-08, "epochs": 15, "early_stopping_patience": 10}
    #test on all sets
    test_indices_list = [np.array([i]) for i in range(8,20)]
    val_indices = np.roll(np.array(test_indices_list), 1).reshape(-1).tolist()
    val_indices_list = [np.array([i]) for i in val_indices]

    train_indices_list = [np.delete(np.arange(0,20), np.hstack((test_indices_list[i],val_indices_list[i]))) for i in range(0,len(test_indices_list))]

    #remaining parameters are predifined by pretrained model
    cnn_hyperparameters_pretrained = {"input_length": 100}

    #parameters for subject specific models (identical to pretrained model)
    cnn_hyperparameters = {"dropout_rate": 0.35,"F1": 8,"D": 8,"F2": 64, "input_length": window_size_training, "num_input_channels": 31}

    pretrained_model = load_pretrained(pretrain_id)
    print(f'Pretrained model loaded')


    #model id where to drop the finetuned models
    model_id_pretrained = args.model_id
    #subjects left out for finetuning
    for subject in subjects:
        for train_indices, val_indices, test_indices in zip(train_indices_list, val_indices_list, test_indices_list):

            checkpoint_path = os.path.join(base_dir,'models','cnn', 'checkpoints', model_id_pretrained, subject, 'test_ind_' + str(test_indices[0].item()))
            #check if path exists and create it if not
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            
            _, _ = train_dnn(subject_string_train = subject, checkpoint_path = checkpoint_path, train_indices = train_indices, 
                                        subject_string_val = [subject], val_indices = val_indices, ica=ica, feature=feature, workers=0, mdl_checkpointing=True,use_pretrained=True,
                                        pretrained_model=pretrained_model, **cnn_train_params, **cnn_hyperparameters_pretrained)
    
    #model id for models without pretraining
    model_id_subj = format(int(args.model_id) + 1, '03d')
    #training loop without pretraining
    for subject in subjects:
        for train_indices, val_indices, test_indices in zip(train_indices_list, val_indices_list, test_indices_list):

            checkpoint_path = os.path.join(base_dir,'models','cnn', 'checkpoints', model_id_subj, subject, 'test_ind_' + str(test_indices[0].item()))
            #check if path exists and create it if not
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            
            _, _ = train_dnn(subject_string_train = subject, checkpoint_path = checkpoint_path, train_indices = train_indices, 
                                        subject_string_val = [subject], val_indices = val_indices, ica=ica, feature=feature, workers=0, mdl_checkpointing=True, 
                                        use_pretrained=False, **cnn_train_params, **cnn_hyperparameters)


def train_basemodels_complete_dataset():
    """
    Training baseline models leaving several subjects out for finetuning.
    Now on the complete dataset.
    """

    #define hyperparameters
    window_size_training = 100
    cnn_hyperparameters = {"dropout_rate": 0.35,"F1": 8,"D": 8,"F2": 64, "input_length": window_size_training, "num_input_channels": 31}
    cnn_train_params = {"data_dir": data_dir, "lr": 0.00003, "batch_size": 512, "weight_decay": 1e-08, "epochs": 35, "early_stopping_patience": 10}

    #subjects in dataset
    #leave out 101, 115, 117, 126, 129
    subjects = list(range(102,115))
    subjects = subjects + list(range(116,117))
    subjects = subjects + list(range(118,126))
    subjects = subjects + [127, 128, 130]

    #create sub lists for pretraining
    held_out_subjects = np.array_split(subjects, 6)
    #subjects that can be used for pretraining
    training_subjects_list = [np.setdiff1d(subjects, held_out) for held_out in held_out_subjects]

    #setting index for subject to be held out --> can train in parallel
    if args.model_id == '130':
        subj_index = 0
    elif args.model_id == '131':
        subj_index = 1
    elif args.model_id == '132':
        subj_index = 2
    elif args.model_id == '133':
        subj_index = 3
    elif args.model_id == '134':
        subj_index = 4
    elif args.model_id == '135':
        subj_index = 5
    else:
        raise ValueError('Invalid model id')

    #to make results reproducible
    np.random.seed(672)

    #choose subjects for training for current execution
    training_subjects = training_subjects_list[subj_index]
    training_subject_strings = [str(subject) for subject in training_subjects]
    #choose three random subjects for validation
    validation_subjects = np.random.choice(training_subjects, size=4, replace=False)
    validation_subject_strings = [str(subject) for subject in validation_subjects]

    #choose four indices between 8 and 19 for validation
    validation_indices = np.random.choice(np.arange(8,19,dtype=int), size=4, replace=False)
    validation_indices = validation_indices.tolist()
    
    #create training indices specific to each subject (validation indices are removed from validation subjects)
    train_indices = []
    for subj, ind in zip(training_subjects, range(0,training_subjects.shape[0])):
        subj_train_indices = np.arange(0,20,1,dtype=int)
        #remove validation trials
        if subj in validation_subjects:
            subj_train_indices = np.delete(subj_train_indices, validation_indices)
        train_indices.append(subj_train_indices.tolist())

    print(f'Entering model {args.model_id} training')

    check_point_path = os.path.join(base_dir,'models','cnn', 'checkpoints', args.model_id, 'basemodel')
    #check if path exists and create it if not
    if not os.path.exists(check_point_path):
        os.makedirs(check_point_path)

    best_correlation, best_state_dict = train_dnn(subject_string_train=training_subject_strings, checkpoint_path = check_point_path, train_indices = train_indices, 
                                            subject_string_val = validation_subject_strings, val_indices = validation_indices, workers=0, mdl_checkpointing=True, **cnn_train_params, **cnn_hyperparameters)

def eval_pretrain_effectiveness_complete_dataset():
    """
    Experiment to evaluate the effectiveness of pretraining on the competing trials.
    Subject specific models are trained using same models and training parameters as in the fine tuning experiments, just without pretraining.
    """

    #test on all sets
    test_indices_list = [np.array([i]) for i in range(8,20)]
    val_indices = np.roll(np.array(test_indices_list), 1).reshape(-1).tolist()
    val_indices_list = [np.array([i]) for i in val_indices]


    #subjects in dataset
    #leave out 101, 115, 117, 126, 129
    subjects = list(range(102,115))
    subjects = subjects + list(range(116,117))
    subjects = subjects + list(range(118,126))
    subjects = subjects + [127, 128, 130]

    #create sub lists for pretraining
    evaluation_subjects = np.array_split(subjects, 6)
    evaluation_subjects = [x.tolist() for x in evaluation_subjects]
    #convert all elements to string
    evaluation_subjects = [[str(x) for x in y] for y in evaluation_subjects]

    #setting pretrained id matching to evaluation subjects

    ###!
    # model ids were not ideal, there should have been a difference of two between consecutive models (the subjects specific models get model_id + 1)
    # this was changed afterwards, identification was possible due to non overlapping subject numbers in the same folders
    #####
    if args.model_id == '140':
        pretrained_id = '130'
        subj_index = 0
    elif args.model_id == '142':
        pretrained_id = '131'
        subj_index = 1
    elif args.model_id == '144':
        pretrained_id = '132'
        subj_index = 2
    elif args.model_id == '146':
        pretrained_id = '133'
        subj_index = 3
    elif args.model_id == '148':
        pretrained_id = '134'
        subj_index = 4
    elif args.model_id == '150':
        pretrained_id = '135'
        subj_index = 5
    else:
        raise ValueError('Invalid model id')
    
    subjects = evaluation_subjects[subj_index]

    compare_pretraining(subjects=subjects, pretrain_id=pretrained_id)

def eval_ica_onset_cnn():
    """
    Evaluates the performance of the subject specific cnn models on ica-cleaned eeg data.
    Also considering onset_envelope as feature.
    """
    base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir

    #subjects in dataset
    #leave out 101, 115, 117, 126, 129
    subjects = list(range(102,115))
    subjects = subjects + list(range(116,117))
    subjects = subjects + list(range(118,126))
    subjects = subjects + [127, 128, 130]

    #create sub lists for pretraining
    evaluation_subjects = np.array_split(subjects, 6)
    evaluation_subjects = [x.tolist() for x in evaluation_subjects]
    #convert all elements to string
    evaluation_subjects = [[str(x) for x in y] for y in evaluation_subjects]

    window_size_training = 100

    cnn_train_params = {"data_dir": data_dir, "lr": 0.00005, "batch_size": 256, "weight_decay": 1e-08, "epochs": 15, "early_stopping_patience": 7}
    #parameters for subject specific models (identical to pretrained model)
    cnn_hyperparameters = {"dropout_rate": 0.35,"F1": 8,"D": 8,"F2": 64, "input_length": window_size_training, "num_input_channels": 31}

    #test on all sets
    test_indices_list = [np.array([i]) for i in range(8,20)]

    val_indices = np.roll(np.array(test_indices_list), 1).reshape(-1).tolist()
    val_indices_list = [np.array([i]) for i in val_indices]

    train_indices_list = [np.delete(np.arange(0,20), np.hstack((test_indices_list[i],val_indices_list[i]))) for i in range(0,len(test_indices_list))]

    #model id for fine tuning
    model_id_subj = format(int(args.model_id), '03d')

    if args.model_id in ['170', '171', '172', '173', '174', '175']:
        ica = True
        feature = 'env'
    elif args.model_id in ['180', '181', '182', '183', '184', '185']:
        ica = False
        feature = 'onset_env'
    elif args.model_id in ['190', '191', '192', '193', '194', '195']:
        ica = True
        feature = 'onset_env'
    else:
        raise ValueError('Invalid model id')
    
    if args.model_id in ['170', '180', '190']:
        subj_index = 0
    elif args.model_id in ['171', '181', '191']:
        subj_index = 1
    elif args.model_id in ['172', '182', '192']:
        subj_index = 2
    elif args.model_id in ['173', '183', '193']:
        subj_index = 3
    elif args.model_id in ['174', '184', '194']:
        subj_index = 4
    elif args.model_id in ['175', '185', '195']:
        subj_index = 5

    subjects = evaluation_subjects[subj_index]

    #training loop without pretraining
    for subject in subjects:
        for train_indices, val_indices, test_indices in zip(train_indices_list, val_indices_list, test_indices_list):

            checkpoint_path = os.path.join(base_dir,'models','cnn', 'checkpoints', model_id_subj, subject, 'test_ind_' + str(test_indices[0].item()))
            #check if path exists and create it if not
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            
            _, _ = train_dnn(subject_string_train = subject, checkpoint_path = checkpoint_path, train_indices = train_indices, 
                                        subject_string_val = [subject], val_indices = val_indices, ica=ica, feature=feature, workers=0, mdl_checkpointing=True, 
                                        use_pretrained=False, **cnn_train_params, **cnn_hyperparameters)


def train_ReLU_Model(ica=False, feature='env'):
    """
    Trains ReLu model with "manual" padding for later quantization
    Args:
        subjects (list): subjects to compare
        pretrain_id (str): model_id of pretrained model to load for the corresponding subjects
        ica (bool, optional): whether to use ica data. Defaults to False.
        feature (str, optional): feature to use. Defaults to 'env'. Must be either 'env' or 'onset_env'.
    """

    assert feature in ['env', 'onset_env'], 'feature must be either env or onset_env'

    window_size_training = 100

    cnn_train_params = {"data_dir": data_dir, "lr": 0.00005, "batch_size": 256, "weight_decay": 1e-08, "epochs": 15, "early_stopping_patience": 10}
    #test on all sets
    test_indices_list = [np.array([i]) for i in range(8,20)]
    val_indices = np.roll(np.array(test_indices_list), 1).reshape(-1).tolist()
    val_indices_list = [np.array([i]) for i in val_indices]

    train_indices_list = [np.delete(np.arange(0,20), np.hstack((test_indices_list[i],val_indices_list[i]))) for i in range(0,len(test_indices_list))]

    #remaining parameters are predifined by pretrained model
    cnn_hyperparameters_pretrained = {"input_length": 100}


    #parameters for subject specific models (identical to pretrained model)
    cnn_hyperparameters = {"dropout_rate": 0.35,"F1": 8,"D": 8,"F2": 64, "input_length": window_size_training, "num_input_channels": 31, 
                           "activation": args.activation_fct, "conv_bias": args.conv_bias}

    #subjects in dataset
    #leave out 101, 115, 117, 126, 129
    subjects = list(range(102,115))
    subjects = subjects + list(range(116,117))
    subjects = subjects + list(range(118,126))
    subjects = subjects + [127, 128, 130]


    subjects = np.array_split(subjects, 6)
    subjects = [x.tolist() for x in subjects]
    #convert all elements to string
    subjects = [[str(x) for x in y] for y in subjects]

    model_id_subj = args.model_id
    subj_index = args.subj_0
    assert subj_index in list(range(0,6)), f'Choose subject index from 0-5. {subj_index} was given.'
    subjects = subjects[subj_index]
    
    #training loop without pretraining
    for subject in subjects:
        for train_indices, val_indices, test_indices in zip(train_indices_list, val_indices_list, test_indices_list):

            checkpoint_path = os.path.join(base_dir,'models','cnn', 'checkpoints', model_id_subj, subject, 'test_ind_' + str(test_indices[0].item()))
            #check if path exists and create it if not
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            
            _, _ = train_dnn(model_handle=CNN_2, subject_string_train = subject, checkpoint_path = checkpoint_path, train_indices = train_indices, 
                                        subject_string_val = [subject], val_indices = val_indices, ica=ica, feature=feature, workers=0, mdl_checkpointing=True, 
                                        use_pretrained=False, **cnn_train_params, **cnn_hyperparameters)


def train_ica_snr_model(ica=True, feature='env', dB = 30, window = '5ms'):
    """
    Train model on dataset with ica art rejection based on snr
    Args:
        subjects (list): subjects to compare
        pretrain_id (str): model_id of pretrained model to load for the corresponding subjects
        ica (bool, optional): whether to use ica data. Defaults to True.
        feature (str, optional): feature to use. Defaults to 'env'. Must be either 'env' or 'onset_env'.
        dB (int, optional): SNR threshold
        window (str, optional): window to look for peak, default '5ms
    """
    assert window in ['1s', '5ms'], 'window must be 1s or 5ms'
    if window == '1s':
        assert dB in [10, 15, 20, 25, 27, 30], 'dB must be either 10, 15, 20, 25, 27 or 30'
        data_dir = os.path.join(base_dir, f"data/processed/ci_attention_final_SNR_crit_{str(dB)}dB.hdf5")
    elif window == '5ms':
        assert dB in [0, 5, 7.5, 10, 12.5, 15, 20]
        data_dir = os.path.join(base_dir, f"data/processed/ci_attention_SNR_5ms_{str(int(dB))}dB.hdf5")
    assert feature in ['env', 'onset_env'], 'feature must be either env or onset_env'

    window_size_training = 100

    cnn_train_params = {"data_dir": data_dir, "lr": 0.00005, "batch_size": 256, "weight_decay": 1e-08, "epochs": 30, "early_stopping_patience": 10}
    #test on all sets
    test_indices_list = [np.array([i]) for i in range(8,20)]
    val_indices = np.roll(np.array(test_indices_list), 1).reshape(-1).tolist()
    val_indices_list = [np.array([i]) for i in val_indices]

    train_indices_list = [np.delete(np.arange(0,20), np.hstack((test_indices_list[i],val_indices_list[i]))) for i in range(0,len(test_indices_list))]

    #remaining parameters are predifined by pretrained model
    cnn_hyperparameters_pretrained = {"input_length": 100}


    #parameters for subject specific models (identical to pretrained model)
    cnn_hyperparameters = {"dropout_rate": 0.35,"F1": 8,"D": 8,"F2": 64, "input_length": window_size_training, "num_input_channels": 31}

    #subjects in dataset
    #leave out 101, 115, 117, 126, 129
    subjects = list(range(102,115))
    subjects = subjects + list(range(116,117))
    subjects = subjects + list(range(118,126))
    subjects = subjects + [127, 128, 130]


    subjects = np.array_split(subjects, 6)
    subjects = [x.tolist() for x in subjects]
    #convert all elements to string
    subjects = [[str(x) for x in y] for y in subjects]

    model_id_subj = args.model_id
    subj_index = args.subj_0
    assert subj_index in list(range(0,6)), f'Choose subject index from 0-5. {subj_index} was given.'
    subjects = subjects[subj_index]
    
    #training loop without pretraining
    for subject in subjects:
        for train_indices, val_indices, test_indices in zip(train_indices_list, val_indices_list, test_indices_list):

            checkpoint_path = os.path.join(base_dir,'models','cnn', 'checkpoints', model_id_subj, subject, 'test_ind_' + str(test_indices[0].item()))
            #check if path exists and create it if not
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            
            _, _ = train_dnn(model_handle=CNN, subject_string_train = subject, checkpoint_path = checkpoint_path, train_indices = train_indices, 
                                        subject_string_val = [subject], val_indices = val_indices, ica=ica, feature=feature, workers=0, mdl_checkpointing=True, 
                                        use_pretrained=False, **cnn_train_params, **cnn_hyperparameters)

if __name__ == '__main__':
    train_ica_snr_model(dB=args.dB)
