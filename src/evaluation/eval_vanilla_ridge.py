# # Vanille Ridge Regression Model on CI Attention Data

import os
import git
import matplotlib.pyplot as plt
import numpy as np
#from mne.filter import filter_data
import h5py
from src.models.ridge import Ridge
import pickle
np.random.seed(0)
from scipy.stats import zscore
import pandas

class RidgeEvaluator():
    def __init__(self, start_ms, end_ms, database_filename, training_scheme = 'concat', freq_range = '1-32Hz', training_window_length = 120, use_ica_data = False, speech_feature = 'env', model_id = '001') -> None:
        """
        Initializes Evaluator class for Ridge Regression.
        database_filename is the name of the the hdf5/h5py file to be used for training and testing it must lay under /data/processed/

        Args:
            start_ms (int): offset of starting lag for ridge regression model
            end_ms (_ty): _description_
            database_filename (string): filename of the hdf5/h5py database file
            training_scheme (str): whether to concat all data or train in windows. Defaults to 'concat'.
            frequ_range (string): preprocessing freq range of EEG
            training_window_length (int, optional): if training scheme is set "windowed": how long shoud a single training window be. Defaults to 120.
            use_ica_data(bool,optional): whether to use ica cleaned EEG data for regression training
            speech_feature (str, optional): which speech feature to use for regression. Must be 'env' or 'onset_env'. Defaults to 'env'.
        """
        # **Set parameters and path files**
        self.base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
        self.data_dir_ci =  os.path.join(self.base_dir, "data", "processed", database_filename)
        self.freq_range = freq_range

        self.regularisation_parameters = np.logspace(-7, 7, 15)

        #set end lag according to findings in Nogueira (2020) https://pubmed.ncbi.nlm.nih.gov/30932825/
        self.end_ms = end_ms
        self.end_lag = int((self.end_ms/1000) * 125)

        self.start_ms = start_ms
        self.start_lag = int((self.start_ms/1000) * 125)

        self.lag_string = str(start_ms) + '-' + str(end_ms) + 'ms'

        assert training_scheme in ['concat', 'windowed'], f"Trianing scheme must be either 'concat' or 'windowed' but input is {training_scheme}"
        self.training_scheme = training_scheme
        self.training_window_length = training_window_length

        assert speech_feature in ['env', 'onset_env'], f"Speech feature must be either 'env' or 'onset_env' but input is {speech_feature}"
        self.speech_feature = speech_feature

        assert os.path.isfile(self.data_dir_ci), f"{self.data_dir_ci} is not a file"
        with h5py.File(self.data_dir_ci, 'r') as f:
            self.subjects = list(f['eeg'].keys())
        f.close()
        #sort by int
        self.subjects = [str(x) for x in sorted([int(x) for x in self.subjects])]


        if len(self.subjects) % 2 == 0:
            self.randomisations = np.tile(np.array([0,1]), int(len(self.subjects)/2))
        else:
            self.randomisations = np.tile(np.array([0,1]), int(len(self.subjects)/2))
            self.randomisations = np.append(self.randomisations, 0)

        self.use_ica_data = use_ica_data
        self.model_id = model_id

    def prepare_training_data(self, subject, val_indices, test_indices, data_set = 'complete'):
        """
        Returns data matrix, label etc. for regression model training
        Can be used for Cross-Validation with appropriate validaten and test indices

        X: EEG-data
        y: speech envelope

        Args:
            subject (string): subject identifier
            val_indices (np.array): indices for validation
            test_indices (np.array): indices for the test set
            dataset (string): 'complete' or 'single_speaker' whether to use the complete dataset or only the single speaker trials

        Returns:
            tuple: X_train, X_val, X_test, y_train, y_val, y_test, y_competing
        """
        assert data_set in ['complete', 'single_speaker'], f"dataset must be either 'complete' or 'single_speaker' but is {data_set}"
        if data_set == 'complete':
            train_indices = np.delete(np.linspace(0,19,20, dtype=int), np.hstack((test_indices, val_indices)))
        elif data_set == 'single_speaker':
            assert np.all(test_indices < 8), f"test indices must be smaller than 8 but are {test_indices}"
            assert np.all(val_indices < 8), f"validation indices must be smaller than 8 but are {test_indices}"
            train_indices = np.delete(np.linspace(0,7,8, dtype=int), np.hstack((test_indices, val_indices)))

        if self.use_ica_data:
            with h5py.File(self.data_dir_ci, 'r') as f:
                trials = (list(f['eeg_ica'][subject].keys()))
                if 'taken_out_indices' in trials:
                    trials.remove('taken_out_indices')
                trials = sorted(trials, key = int)
                trials = np.array(trials)
                #print(f'trials: {trials}')
                
                train_parts = trials[train_indices]
                val_parts = trials[val_indices]
                test_parts = trials[test_indices]

                attended_stimuli = np.array([f[f'eeg/{subject}/{j}'].attrs['stimulus'] for j in trials])
                train_stimuli = attended_stimuli[train_indices]
                val_stimuli = attended_stimuli[val_indices]
                test_stimuli = attended_stimuli[test_indices]

                X_train = np.hstack([zscore(f[f'eeg_ica/{subject}/{j}'][:], axis = 1) for j in train_parts])
                X_val = np.hstack([zscore(f[f'eeg_ica/{subject}/{j}'][:], axis = 1) for j in val_parts])
                X_test = np.hstack([zscore(f[f'eeg_ica/{subject}/{j}'][:], axis = 1) for j in test_parts])

                #drop aux channels
                X_train, X_val, X_test = X_train[:31,:], X_val[:31,:], X_test[:31,:]

                if self.speech_feature == 'env':
                    y_train = np.hstack([f[f'stimulus_files/{str(j)}/attended_env'][:] for j in train_stimuli])
                    y_val =  np.hstack([f[f'stimulus_files/{str(j)}/attended_env'][:] for j in val_stimuli])
                    y_attended = np.hstack([f[f'stimulus_files/{str(j)}/attended_env'][:] for j in test_stimuli])
                    y_competing = np.hstack([f[f'stimulus_files/{str(j)}/distractor_env'][:] for j in test_stimuli])
                elif self.speech_feature == 'onset_env':
                    y_train = np.hstack([f[f'stimulus_files/{str(j)}/attended_onset_env'][:] for j in train_stimuli])
                    y_val =  np.hstack([f[f'stimulus_files/{str(j)}/attended_onset_env'][:] for j in val_stimuli])
                    y_attended = np.hstack([f[f'stimulus_files/{str(j)}/attended_onset_env'][:] for j in test_stimuli])
                    y_competing = np.hstack([f[f'stimulus_files/{str(j)}/distractor_onset_env'][:] for j in test_stimuli])
                
            f.close()

        else:
            with h5py.File(self.data_dir_ci, 'r') as f:
                trials = (list(f['eeg'][subject].keys()))
                if 'taken_out_indices' in trials:
                    trials.remove('taken_out_indices')
                trials = sorted(trials, key = int)
                trials = np.array(trials)
                #print(f'trials: {trials}')
                
                train_parts = trials[train_indices]
                val_parts = trials[val_indices]
                test_parts = trials[test_indices]

                attended_stimuli = np.array([f[f'eeg/{subject}/{j}'].attrs['stimulus'] for j in trials])
                train_stimuli = attended_stimuli[train_indices]
                val_stimuli = attended_stimuli[val_indices]
                test_stimuli = attended_stimuli[test_indices]

                X_train = np.hstack([zscore(f[f'eeg/{subject}/{j}'][:], axis = 1) for j in train_parts])
                X_val = np.hstack([zscore(f[f'eeg/{subject}/{j}'][:], axis = 1) for j in val_parts])
                X_test = np.hstack([zscore(f[f'eeg/{subject}/{j}'][:], axis = 1) for j in test_parts])

                #drop aux channels
                X_train, X_val, X_test = X_train[:31,:], X_val[:31,:], X_test[:31,:]

                if self.speech_feature == 'env':
                    y_train = np.hstack([f[f'stimulus_files/{str(j)}/attended_env'][:] for j in train_stimuli])
                    y_val =  np.hstack([f[f'stimulus_files/{str(j)}/attended_env'][:] for j in val_stimuli])
                    y_attended = np.hstack([f[f'stimulus_files/{str(j)}/attended_env'][:] for j in test_stimuli])
                    y_competing = np.hstack([f[f'stimulus_files/{str(j)}/distractor_env'][:] for j in test_stimuli])

                elif self.speech_feature == 'onset_env':
                    y_train = np.hstack([f[f'stimulus_files/{str(j)}/attended_onset_env'][:] for j in train_stimuli])
                    y_val =  np.hstack([f[f'stimulus_files/{str(j)}/attended_onset_env'][:] for j in val_stimuli])
                    y_attended = np.hstack([f[f'stimulus_files/{str(j)}/attended_onset_env'][:] for j in test_stimuli])
                    y_competing = np.hstack([f[f'stimulus_files/{str(j)}/distractor_onset_env'][:] for j in test_stimuli])
            f.close()

        return X_train, X_val, X_test, y_train, y_val, y_attended, y_competing

    def prepare_per_storytraining_data(self, subject, val_indices, test_indices, story = 'elb', randomisation = 0, attended_only = False):
        """
        Return data specific for one story. E.g. envelope is always from the same story, but attention in EEG data is from different stories
        Returns data matrix, label etc. for regression model training
        Can be used for Cross-Validation with appropriate validaten and test indices

        X: EEG-data
        y: speech envelope

        Args:
            subject (string): subject identifier
            val_indices (np.array): indices for validation
            test_indices (np.array): indices for the test set
            story (string): story identifier. 'elb' or 'pol'
            randomisation (int): randomisation identifier from experiment. 0,1
            attended_only (bool): whether to only use attended speech stream for training, otherwise also distractor stream is used
        Returns:
            tuple: X_train, X_val, X_test, y_train, y_val, y_test, y_competing
        """
        assert story in ['elb', 'pol'], f"story must be either 'elb' or 'pol' but is {story}"
        env_order = np.tile(np.array(['attended_env', 'attended_env', 'distractor_env', 'distractor_env']), 5)
        onset_order = np.tile(np.array(['attended_onset_env', 'attended_onset_env', 'distractor_onset_env', 'distractor_onset_env']), 5)


        #choosing the correct indices for training
        if story == 'elb':
            if randomisation == 0:
                #excludes single speaker trials with polarnacht and test and validation indices
                del_indicices = np.hstack((np.array([2,3,6,7]), test_indices, val_indices))
                if attended_only:
                    #excludes those trials in competing scenario where polarnacht is attended
                    del_indicices = np.hstack((del_indicices,np.array([10,11,14,15,18,19])))
                del_indicices = np.unique(del_indicices)
            
            elif randomisation == 1:
                del_indicices = np.hstack((np.array([0,1,4,5]),  test_indices, val_indices))
                if attended_only:
                    del_indicices = np.hstack((del_indicices,np.array([8,9,12,13,16,17])))
                del_indicices = np.unique(del_indicices)
                
                env_order = np.roll(env_order, 2)
                onset_order = np.roll(onset_order, 2)

        elif story == 'pol':
            if randomisation == 0:
                del_indicices = np.hstack((np.array([0,1,4,5]), test_indices, val_indices))
                if attended_only:
                    del_indicices = np.hstack((del_indicices,np.array([8,9,12,13,16,17])))
                del_indicices = np.unique(del_indicices)

                env_order = np.roll(env_order, 2)
                onset_order = np.roll(onset_order, 2)

            elif randomisation == 1:
                del_indicices = np.hstack((np.array([2,3,6,7]), test_indices, val_indices))
                if attended_only:
                    del_indicices = np.hstack((del_indicices,np.array([10,11,14,15,18,19])))
                del_indicices = np.unique(del_indicices)

        train_indices = np.delete(np.linspace(0,19,20, dtype=int),del_indicices)


        with h5py.File(self.data_dir_ci, 'r') as f:
            trials = (list(f['eeg'][subject].keys()))
            if 'taken_out_indices' in trials:
                trials.remove('taken_out_indices')
            trials = sorted(trials, key = int)
            trials = np.array(trials)
            #print(f'trials: {trials}')

            train_parts = trials[train_indices]
            val_parts = trials[val_indices]
            test_parts = trials[test_indices]

            attended_stimuli = np.array([f[f'eeg/{subject}/{j}'].attrs['stimulus'] for j in trials])

            train_stimuli = attended_stimuli[train_indices]
            train_env_order = env_order[train_indices]
            train_onset_env_order = onset_order[train_indices]

            val_stimuli = attended_stimuli[val_indices]
            val_env_order = env_order[val_indices]
            val_onset_env_order = onset_order[val_indices]

            test_stimuli = attended_stimuli[test_indices]
            test_env_order = env_order[test_indices]
            #flips attended and unattended
            test_competing_env_order = np.roll(env_order,2)[test_indices]

            test_onset_env_order = onset_order[test_indices]
            test_competing_onset_env_order = np.roll(onset_order,2)[test_indices]

            if self.use_ica_data:
                X_train = np.hstack([zscore(f[f'eeg_ica/{subject}/{j}'][:], axis = 1) for j in train_parts])
                X_val = np.hstack([zscore(f[f'eeg_ica/{subject}/{j}'][:], axis = 1) for j in val_parts])
                X_test = np.hstack([zscore(f[f'eeg_ica/{subject}/{j}'][:], axis = 1) for j in test_parts])
            else:
                X_train = np.hstack([zscore(f[f'eeg/{subject}/{j}'][:], axis = 1) for j in train_parts])
                X_val = np.hstack([zscore(f[f'eeg/{subject}/{j}'][:], axis = 1) for j in val_parts])
                X_test = np.hstack([zscore(f[f'eeg/{subject}/{j}'][:], axis = 1) for j in test_parts])

            #drop aux channels
            X_train, X_val, X_test = X_train[:31,:], X_val[:31,:], X_test[:31,:]

            if self.speech_feature == 'env':
                y_train = np.hstack([f[f'stimulus_files/{str(j)}/{env}'][:] for j, env in zip(train_stimuli, train_env_order)])
                y_val =  np.hstack([f[f'stimulus_files/{str(j)}/{env}'][:] for j, env in zip(val_stimuli, val_env_order)])
                y_test = np.hstack([f[f'stimulus_files/{str(j)}/{env}'][:] for j, env in zip(test_stimuli, test_env_order)])
                y_test_competing = np.hstack([f[f'stimulus_files/{str(j)}/{env}'][:] for j, env in zip(test_stimuli, test_competing_env_order)])

            elif self.speech_feature == 'onset_env':
                y_train = np.hstack([f[f'stimulus_files/{str(j)}/{env}'][:] for j, env in zip(train_stimuli, train_onset_env_order)])
                y_val =  np.hstack([f[f'stimulus_files/{str(j)}/{env}'][:] for j, env in zip(val_stimuli, val_onset_env_order)])
                y_test = np.hstack([f[f'stimulus_files/{str(j)}/{env}'][:] for j, env in zip(test_stimuli, test_onset_env_order)])
                y_test_competing = np.hstack([f[f'stimulus_files/{str(j)}/{env}'][:] for j, env in zip(test_stimuli, test_competing_onset_env_order)])
            
        f.close()

        return X_train, X_val, X_test, y_train, y_val, y_test, y_test_competing

    def plot_validation(self, regularisation_parameters, val_scores):
        """
        plots parameter selection cureve
        Args:
            regularisation_parameters (np.array): 
            val_scores (np.array): valdidation scores on tested reg. params
        """
        plt.plot(regularisation_parameters, val_scores)
        plt.xscale('log')
        plt.title('Reconstruction accuracies for validation dataset')
        plt.ylabel('Validation score (Pearson r)')
        plt.xlabel('Regularisation parameter') 

    def eval_on_window(self, mdl, X_test, y_attended, y_distractor, window_size, competing = True):
        """Calculates attended, distractor scores and accuracy on test set in windows
        if competing is set to False, only attended scores are calculated

        Args:
            mdl (src.models.ridge.Ridge): ridge object, trained and validated
            X_test (np.array): EEG test data
            y_attended (np.array): attended test speech envelope
            y_distractor (np.array): distractor test speech envelope
            window_size (float): windows size on which to evaluate in s

        Returns:
            tuple: attended scores, mean(attended scores), distractor scores, mean(distractor scores) accuracy
        """
        
        window_len = window_size * 125
        n_windows = int(np.floor(len(y_attended) / window_len))
        n_true, n_false = 0,0
        attended_array, distractor_array = np.array([]), np.array([])

        if competing:
            for window in range(0, n_windows):
                X_test_win = X_test[:, window * window_len : (window + 1) * window_len]
                y_attended_win, y_distractor_win = y_attended[ window * window_len : (window + 1) * window_len], y_distractor[ window * window_len : (window + 1) * window_len]
                attended_score = mdl.score(X_test_win.T, y_attended_win[:, np.newaxis])
                distractor_score = mdl.score(X_test_win.T, y_distractor_win[:, np.newaxis])
                if attended_score > distractor_score:
                    n_true += 1
                elif distractor_score > attended_score:
                    n_false += 1
                
                attended_array = np.append(attended_array, attended_score)
                distractor_array = np.append(distractor_array, distractor_score)
            
            accuracy = n_true / (n_true + n_false)
            distractor_mean = np.mean(distractor_array).item()
            attended_mean = np.mean(attended_array).item()

        else:
            for window in range(0, n_windows):
                X_test_win = X_test[:, window * window_len : (window + 1) * window_len]
                y_attended_win = y_attended[ window * window_len : (window + 1) * window_len]
                attended_score = mdl.score(X_test_win.T, y_attended_win[:, np.newaxis])
                
                attended_array = np.append(attended_array, attended_score)
                distractor_array = np.append(distractor_array, np.nan)
            
            accuracy = np.nan
            attended_mean = np.mean(attended_array)
            distractor_mean = np.nan
        
        return attended_array, attended_mean, distractor_array, distractor_mean, accuracy

    def eval_story_models_on_window(self, mdl_elb, mdl_pol, X_test, y_test, attended, window_sizes, competing = True):
        """_summary_

        Args:
            mdl_elb (_type_): Elbenwald Model
            mdl_pol (_type_): Polarnacht Model 
            X_test (_type_): Test EEG data
            y_elb (_type_): elbenwald test speech envelope
            y_pol (_type_): polarnacht test speech envelope
            attended(str): attended speech stream
            window_sizes (list): window size in s
            competing (bool, optional): Wheter to evaluation competing scenario Defaults to True.

        Returns:
            _type_: _description_
        """
        
        assert attended in ['elb', 'pol'], f"attended must be either 'elb' or 'pol' but is {attended}"
        #assert y_elb.shape == y_pol.shape, f"y_elb and y_pol must have same shape but are {y_elb.shape} and {y_pol.shape}"

        #if window_sizes is not a list, make it one
        if not isinstance(window_sizes, list):
            window_sizes = [window_sizes]
        

        #lists over window sizes
        elb_scores_list, pol_scores_list, accuracy_list = [], [], []
        if competing:
            for window_size in window_sizes:
                elb_scores_array, pol_scores_array = np.array([]), np.array([])
                window_len = window_size * 125
                n_windows = int(np.floor(len(y_test) / window_len))
                n_true, n_false = 0,0
                for window in range(0, n_windows):
                    X_test_win = X_test[:, window * window_len : (window + 1) * window_len]
                    y_test_win = y_test[ window * window_len : (window + 1) * window_len]

                    #get scores for both models
                    elb_score = mdl_elb.score(X_test_win.T, y_test_win[:, np.newaxis])
                    pol_score = mdl_pol.score(X_test_win.T, y_test_win[:, np.newaxis])

                    if attended == 'elb':
                        if elb_score > pol_score:
                            n_true += 1
                        else:
                            n_false += 1
                    elif attended == 'pol':
                        if pol_score > elb_score:
                            n_true += 1
                        else:
                            n_false += 1
                    
                    elb_scores_array = np.append(elb_scores_array, elb_score)
                    pol_scores_array = np.append(pol_scores_array, pol_score)
            
                accuracy = n_true / (n_true + n_false)
                # elb_mean = np.mean(elb_scores_array).item()
                # pol_mean = np.mean(pol_scores_array).item()

                elb_scores_list.append(elb_scores_array)
                pol_scores_list.append(pol_scores_array)
                accuracy_list.append(accuracy)

        else:
            pass
        
        return elb_scores_list, pol_scores_list, accuracy_list

    def run_model_eval(self, val_indices, test_indices, accuracy_window_size,):
        """Evaluate Ridge regression model on one test set.
        Return pearson coefficient on test set (scores) and accuracies for all subjects

        Args:
            val_indices (np.array): validation indices
            test_indices (np.array): test indices
            accuracy_window_size (float): windows size on which to evaluate in s
            training_scheme(string): 'concat' or 'windowed' whether to concatenate all training data, or train in windows and average the resulting models
            training_window_size (int): seconds to take for one window of training

        Returns:
            tuple: attended_scores, distractor_scores, accuracies
        """
        attended_scores = []
        distractor_scores = []
        accuracies = []

        for subject in self.subjects:
            #init model
            mdl = Ridge(start_lag = self.start_lag, end_lag = self.end_lag, alpha = self.regularisation_parameters)

            #fit model and select hyperparam
            X_train, X_val, X_test, y_train, y_val, y_test, y_competing = self.prepare_training_data(subject, val_indices, test_indices)
            
            if self.training_scheme == 'concat':
                mdl.fit(X_train.T, y_train[:,np.newaxis])
                
            elif self.training_scheme == 'windowed':
                window_len = 125 * self.training_window_length
                n_windows = int(X_train.shape[1] / window_len)

                coefs = []

                for window in range(0,n_windows):
                    X_train_win = X_train[:, window*window_len: (window + 1) *window_len]
                    y_train_win = y_train[window*window_len: (window + 1) *window_len]
                    mdl.fit(X_train_win.T, y_train_win[:, np.newaxis])
                    coefs.append(mdl.coef_)

                #average coefs and add to model
                mean_coefs = np.mean(np.array(coefs), axis = 0)
                mdl.coef_ = mean_coefs

            _ = mdl.model_selection(X_val.T, y_val[:,np.newaxis])
            #get model scores
            
            if np.any(test_indices < 8):
                #single speaker in test set, hence no distractor file, score the model afterwards (it's available)
                
                _, attended_score, _, _, _ = self.eval_on_window(mdl, X_test, y_attended=y_test, y_distractor=y_competing, window_size = accuracy_window_size, competing=False)

            else:
                _, attended_score, _, distractor_score, acc = self.eval_on_window(mdl, X_test, y_attended=y_test, y_distractor=y_competing, window_size = accuracy_window_size, competing=True)
            
            distractor_scores.append(distractor_score)
            accuracies.append(acc)
            attended_scores.append(attended_score)
            
            print(f'subject {subject} done')

            #save model
            model_path = os.path.join(self.base_dir, "models", "ridge", self.model_id, subject)
            if os.path.exists(model_path):
                pickle.dump(mdl, open(os.path.join(model_path, str(test_indices[0])  +".pk"), "wb"))
            else:
                os.makedirs(model_path)
                pickle.dump(mdl, open(os.path.join(model_path, str(test_indices[0]) + ".pk"), "wb"))
        return attended_scores, distractor_scores, accuracies

    def run_story_model_eval(self, val_indices, test_indices, accuracy_window_size, debug = False, attended_only = False):
        """Evaluate Ridge regression model on one test set.
        Return pearson coefficient on test set (scores) and accuracies for all subjects

        Args:
            val_indices (np.array): validation indices
            test_indices (np.array): test indices
            accuracy_window_size (float): windows size on which to evaluate in s
            training_scheme(string): 'concat' or 'windowed' whether to concatenate all training data, or train in windows and average the resulting models
            training_window_size (int): seconds to take for one window of training
            attended_only (bool): whether to only use attended speech stream for training, otherwise also distractor stream is used

        Returns:
            tuple: attended_scores, distractor_scores, label, accuracies
        """
        elb_scores = []
        pol_scores = []
        accuracies = []

        labels = []

        if debug:
            subjects = self.subjects[:2]
            randomisations = self.randomisations[:2]
        else:
            subjects = self.subjects
            randomisations = self.randomisations

        for subject, rand in zip(subjects, randomisations):
            #init model
            mdl_elb = Ridge(start_lag = self.start_lag, end_lag = self.end_lag, alpha = self.regularisation_parameters)
            mdl_pol = Ridge(start_lag = self.start_lag, end_lag = self.end_lag, alpha = self.regularisation_parameters)

            #fit model and select hyperparam
            X_train_elb, X_val_elb, _, y_train_elb, y_val_elb, y_test_elb, y_competing_elb = self.prepare_per_storytraining_data(subject, val_indices, test_indices, story = 'elb', randomisation = rand, attended_only=attended_only)
            X_train_pol, X_val_pol, _, y_train_pol, y_val_pol, y_test_pol, y_competing_pol = self.prepare_per_storytraining_data(subject, val_indices, test_indices, story = 'pol', randomisation = rand, attended_only=attended_only)
            #for the same test index we should get y_test_elb == y_competing_pol

            #for debugging
            if debug:
                X_train_elb, y_train_elb = X_train_elb[:,:1000], y_train_elb[:1000]
                X_train_pol, y_train_pol = X_train_pol[:,:1000], y_train_pol[:1000]

                X_val_elb, y_val_elb = X_val_elb[:,:1000], y_val_elb[:1000]
                X_val_pol, y_val_pol = X_val_pol[:,:1000], y_val_pol[:1000]

            #for reference - what the actual attended and distractor stimuli are
            _, _, X_test, _, _, y_test, _ = self.prepare_training_data(subject, val_indices, test_indices)

            #check if y_test is elb or pol
            if np.all(y_test == y_test_elb):
                attended = 'elb'
            elif np.all(y_test == y_test_pol):
                attended = 'pol'
            else:
                raise ValueError('y_test is neither elb nor pol')

            if self.training_scheme == 'concat':
                #model tries to reconstruct elbenwald env
                mdl_elb.fit(X_train_elb.T, y_train_elb[:,np.newaxis])
                #model tries to reconstruct pol env
                mdl_pol.fit(X_train_pol.T, y_train_pol[:,np.newaxis])

            elif self.training_scheme == 'windowed':
                window_len = 125 * self.training_window_length
                n_windows = int(X_train_elb.shape[1] / window_len)

                coefs_elb = []
                coefs_pol = []

                for window in range(0,n_windows):
                    X_train_win_elb = X_train_elb[:, window*window_len: (window + 1) *window_len]
                    y_train_win_elb = y_train_elb[window*window_len: (window + 1) *window_len]
                    mdl_elb.fit(X_train_win_elb.T, y_train_win_elb[:, np.newaxis])
                    coefs_elb.append(mdl_elb.coef_)
                    
                    #repeat for pol
                    X_train_win_pol = X_train_pol[:, window*window_len: (window + 1) *window_len]
                    y_train_win_pol = y_train_pol[window*window_len: (window + 1) *window_len]
                    mdl_pol.fit(X_train_win_pol.T, y_train_win_pol[:, np.newaxis])
                    coefs_pol.append(mdl_pol.coef_)


                #average coefs and add to model
                mean_coefs_elb = np.mean(np.array(coefs_elb), axis = 0)
                mdl_elb.coef_ = mean_coefs_elb

                mean_coefs_pol = np.mean(np.array(coefs_pol), axis = 0)
                mdl_pol.coef_ = mean_coefs_pol

            _ = mdl_elb.model_selection(X_val_elb.T, y_val_elb[:,np.newaxis])
            _ = mdl_pol.model_selection(X_val_pol.T, y_val_pol[:,np.newaxis])
            #get model scores
            
            #let both models score on same data and see which reconstructs with higher pearson r
            
            if np.any(test_indices < 8):
                #single speaker in test set, hence no distractor file, score the model afterwards (it's available)
                raise ValueError('Single speaker in test set not possible for story model eval')

            else:
                elb_score, pol_score, acc = self.eval_story_models_on_window(mdl_elb, mdl_pol, X_test, y_test, attended, window_sizes = accuracy_window_size, competing=True)
            
            elb_scores.append(elb_score)
            pol_scores.append(pol_score)
            accuracies.append(acc)

            print(f'subject {subject} done')

            #save model
            model_path = os.path.join(self.base_dir, "models", "ridge", self.model_id, subject)
            metrics_path = os.path.join(self.base_dir, "reports", "metrics", "ridge", self.model_id, subject)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            
            pickle.dump(mdl_elb, open(os.path.join(model_path, 'elb' + str(test_indices[0])  +".pk"), "wb"))
            pickle.dump(mdl_pol, open(os.path.join(model_path, 'pol' + str(test_indices[0])  +".pk"), "wb"))
            #dump the metrics
            if not os.path.exists(metrics_path):
                os.makedirs(metrics_path)
            pickle.dump(elb_scores, open(os.path.join(metrics_path, 'elb' + str(test_indices[0])  +".pk"), "wb"))
            pickle.dump(pol_scores, open(os.path.join(metrics_path, 'pol' + str(test_indices[0])  +".pk"), "wb"))
            pickle.dump(accuracies, open(os.path.join(metrics_path, 'acc' + str(test_indices[0])  +".pk"), "wb"))
            #dump attended
            pickle.dump(attended, open(os.path.join(metrics_path, 'attended' + str(test_indices[0])  +".pk"), "wb"))
            labels.append(attended)

        return elb_scores, pol_scores, labels, accuracies

    def get_attended_scores_from_trained_mdl(self, mdl:Ridge, subject, val_indices, test_indices, accuracy_window_size = 60):
        """
        Getting attended, distractor score and decoding accuracy to evaluate fitted model (e.g. if training was run on workstation)

        Args:
            mdl (Ridge): Already fitted Ridge Model to evaluate
            subject (int): subject identifier
            val_indices (np.array): validation indices
            test_indices (np.array): test indices

        Returns:
            (tuple): attended_array, attended_mean, distractor_array, distractor_mean, accuracy
        """
        _, _, X_test, _, _, y_test, y_competing = self.prepare_training_data(subject, val_indices, test_indices)
        attended_array, attended_mean, distractor_array, distractor_mean, accuracy = self.eval_on_window(mdl, X_test, y_attended=y_test, y_distractor=y_competing, window_size = accuracy_window_size, competing=True)
        
        return attended_array, attended_mean, distractor_array, distractor_mean, accuracy

    def plot_attention_scores(self, attended_scores, distractor_scores, test_indices):
        """Plots attention scores to get a comparison between attended and distractor speaker

        Args:
            attended_scores (np.array): subject-wise average pearson scores on attended test data 
            distractor_scores (np.array): subject-wise average pearson scores on distractor test data 
            test_indices (np.array): Indices used for testing between 7 and 19 
        """

        attended_scores, distractor_scores = np.array(attended_scores), np.array(distractor_scores)

        mean_attended, var_attended = np.mean(attended_scores), np.var(attended_scores)
        mean_distractor, var_distractor = np.mean(distractor_scores), np.var(distractor_scores)

        subject_axis = [int(subject) for subject in self.subjects]

        fig, ax = plt.subplots(1,1)
        ax.plot(subject_axis, attended_scores, '-x', label = 'attended', c='#1f77b4')
        ax.plot(subject_axis, np.repeat(mean_attended, len(subject_axis)), linestyle = 'dashed', c='#1f77b4', label = 'mean attended')
        ax.fill_between(subject_axis, mean_attended - var_attended, mean_attended + var_attended, alpha = 0.2, color ='#1f77b4', label = 'variance attended')

        ax.plot(subject_axis, distractor_scores, '-o', label = 'distractor', c = '#ff7f0e')
        ax.plot(subject_axis, np.repeat(mean_distractor, len(subject_axis)), linestyle = 'dashed', c='#ff7f0e', label = 'mean distractor')
        ax.fill_between(subject_axis, mean_distractor - var_distractor, mean_distractor + var_distractor, alpha = 0.2, color = '#ff7f0e', label = 'variance distractor')

        ax.legend()
        ax.grid()
        ax.set_xticks(subject_axis)
        ax.set_xlabel('subject')
        ax.set_ylabel('average pearson r')
        ax.set_title(f'Competing Speaker Scenario, test trials {str(test_indices[0])}')
        
        figure_path = os.path.join(self.base_dir,"reports","figures", "ridge", self.training_scheme, self.lag_string)

        if os.path.exists(figure_path):
            fig.savefig(os.path.join(figure_path, f"Comp_sp_pearson_{str(test_indices[0])}"))
        else:
            os.makedirs(figure_path)
            fig.savefig(os.path.join(figure_path, f"Comp_sp_pearson_{str(test_indices[0])}"))

    def plot_accuracies(self, accuracies, test_indices, accuracy_window_size = 60):
        """Plots accuracies
        Args:
            accuracies (np.array): subject-wise average decoding accuracies
            test_indices (np.array): test indices the on which the accuracies were gathered
        """
        
        mean_accuracies, var_accuracies = np.mean(accuracies), np.var(accuracies)
        subject_axis = [int(subject) for subject in self.subjects]

        fig, ax = plt.subplots(1)
        ax.plot(subject_axis, accuracies, '-o', label = 'subject accuracies', c='#1f77b4')
        ax.plot(subject_axis, np.repeat(mean_accuracies, len(subject_axis)), linestyle = 'dashed', c='#1f77b4', label = 'mean')
        ax.fill_between(subject_axis, mean_accuracies - var_accuracies, mean_accuracies + var_accuracies, alpha = 0.2, color ='#1f77b4', label = 'variance')
        ax.grid()
        ax.set_ylabel('average decoding acc')
        ax.set_xticks(subject_axis)
        ax.set_xlabel('subject')
        ax.set_title(f'Average Decoding accuracies {str(accuracy_window_size)}s test trials {str(test_indices[0])}')
        ax.legend()
        
        figure_path = os.path.join(self.base_dir,"reports","figures", "ridge", self.training_scheme, self.lag_string)

        if os.path.exists(figure_path):
            fig.savefig(os.path.join(figure_path, f"Comp_sp_acc_{str(test_indices[0])}"))
        else:
            os.makedirs(figure_path)
            fig.savefig(os.path.join(figure_path, f"Comp_sp_ac_{str(test_indices[0])}"))

    def eval_single_speaker(self, test_indices_list, val_indices_list):
        """
        Compares Reconstruction scores to null score
        """
        test_scores_single = []
        accuracies = []
        for subject in self.subjects:
            mdl = Ridge(start_lag=self.start_lag, end_lag = self.end_lag, alpha = self.regularisation_parameters)
            X_train, X_val, X_test, y_train, y_val, y_attended, y_null_score = self.prepare_training_data(subject, val_indices = val_indices_list, test_indices = test_indices_list, data_set = 'single_speaker')
            mdl.fit(X_train.T, y_train[:,np.newaxis])
            val_scores = mdl.model_selection(X_val.T, y_val[:,np.newaxis])
            test_score = mdl.score(X_test.T, y_attended[:,np.newaxis])
            test_scores_single.append(test_score)

        return test_scores_single

    def eval_competing_speaker_cross_val(self, test_indices_list, val_indices_list, accuracy_window_size = 60, training_scheme = 'concat'):
        """Running cross-validaton on competing speaker scenario: trains and validates.

        Args:
            training_scheme (str, optional):how to arange training data. Defaults to 'concat'.
            test_indices_list (list): list of test indices as np. arrays (e.g. [np.array([8,9]), np.array([10,11])])
            val_indices_list (list): list of validation indices as np. arrays (e.g. [np.array([6,7]), np.array([8,9])])
            accuracy_window_size (int, optional): window size for decoding accuracy. Defaults to 60.
        """

        """
        # **Initialize indices and lists to runn 6-fold Cross-Val**
        assert book_test_set in ['same', 'different'], f"Parameter book_test set mus be either same of different but {book_test_set} was passed."
        
        if book_test_set == 'same':
            test_indices_list = [np.array([i,i+1]) for i in range(8,20,2)]
            val_indices_list = list(np.roll(np.array(test_indices_list),2))

        elif book_test_set == 'different':
            test_indices_list = [np.array([i,i+2]) for i in [8,9,12,13,16,17]]
            val_indices_list = list(np.roll(np.array(test_indices_list),2))
        """
        cross_val_att_scores, cross_val_distr_scores, cross_val_acc = [], [], []

        for test_indices, val_indices, fold in zip(test_indices_list, val_indices_list, range(1,len(test_indices_list)+1)):
            
            attended_scores, distractor_scores, accuracies = self.run_model_eval(val_indices=val_indices, test_indices=test_indices, accuracy_window_size=accuracy_window_size)
            self.plot_attention_scores(attended_scores, distractor_scores, test_indices)
            self.plot_accuracies(accuracies, test_indices, accuracy_window_size=accuracy_window_size)
            
            cross_val_att_scores.append(attended_scores)
            cross_val_distr_scores.append(distractor_scores)
            cross_val_acc.append(accuracies)

            print(f'Crossvalidation fold {fold} \n')

        subject_means, subject_var = np.mean(np.array(cross_val_acc), axis=0), np.var(np.array(cross_val_acc), axis=0)

        path_acc = os.path.join(self.base_dir, "reports", "metrics", "ridge", self.model_id)
        path_attended = os.path.join(self.base_dir, "reports", "metrics", "ridge", self.model_id)
        path_distr = os.path.join(self.base_dir, "reports", "metrics", "ridge", self.model_id)
        #check if paths exist and create otherwise
        if not os.path.exists(path_acc):
            os.makedirs(path_acc)
        if not os.path.exists(path_attended):
            os.makedirs(path_attended)
        if not os.path.exists(path_distr):
            os.makedirs(path_distr)

        np.save(os.path.join(path_acc, "cross_val_acc.npy"), np.array(cross_val_acc))
        np.save(os.path.join(path_attended, "cross_val_att_scores.npy"), np.array(cross_val_att_scores))
        np.save(os.path.join(path_distr, "cross_val_distr_scores.npy"), np.array(cross_val_distr_scores))

        subject_axis = [int(subject) for subject in self.subjects]

        fig, ax = plt.subplots(1)
        ax.plot(subject_axis, subject_means, '-o', label = 'mean')
        ax.fill_between(subject_axis, subject_means - subject_var, subject_means + subject_var, alpha = 0.2, label = 'variance')
        ax.grid()
        ax.set_xticks(subject_axis)
        ax.set_xlabel('subject')
        ax.set_ylabel('average accuracy')
        ax.legend()
        ax.set_title('Cross-Validated per subject decoding accuracies')
        
        fig_path = os.path.join(self.base_dir,"reports","figures", "ridge", self.training_scheme, self.model_id)
        
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)

        fig.savefig(os.path.join(fig_path, "Cross_Val_subject_acc.png"))
        fig.savefig(os.path.join(fig_path, "Cross_Val_subject_acc.pdf"))

        trial_means, trial_var = np.mean(np.array(cross_val_acc), axis=1), np.var(np.array(cross_val_acc), axis=1)

        fig, ax = plt.subplots(1)
        trial_axis = [i[0] for i in test_indices_list]
        ax.plot(trial_axis, trial_means, '-o', label = 'means')
        ax.fill_between(trial_axis, trial_means - trial_var, trial_means + trial_var, alpha = 0.2, label = 'variance')
        ax.set_xticks(trial_axis)
        ax.grid()
        ax.set_xlabel('first test trial')
        ax.set_ylabel('average accuracy')
        ax.set_title('Cross-Validated per test trial decoding accuracies')
        ax.legend()

        fig.savefig(os.path.join(fig_path, "Cross_Val_trial_acc.png"))
        fig.savefig(os.path.join(fig_path, "Cross_Val_trial_acc.pdf"))

    def test_env(self, training_scheme = 'concat'):
        """
        Runs Cross-Validation on competing speaker.
        Just to check if code works
        """

        # **Initialize indices and lists to runn 6-fold Cross-Val**
        test_indices_list = [np.array([i,i+1]) for i in range(8,10,2)]
        val_indices_list = list(np.roll(np.array(test_indices_list),2))
        accuracy_window_size = 60
        cross_val_att_scores, cross_val_distr_scores, cross_val_acc = [], [], []

        for test_indices, val_indices, fold in zip(test_indices_list, val_indices_list, range(1,len(test_indices_list)+1)):
            
            attended_scores, distractor_scores, accuracies = self.run_model_eval(val_indices=val_indices, test_indices=test_indices, accuracy_window_size=accuracy_window_size)
            self.plot_attention_scores(attended_scores, distractor_scores, test_indices)
            self.plot_accuracies(accuracies, test_indices)
            
            cross_val_att_scores.append(attended_scores)
            cross_val_distr_scores.append(distractor_scores)
            cross_val_acc.append(accuracies)

            print(f'Crossvalidation fold {fold} \n')

        subject_means, subject_var = np.mean(np.array(cross_val_acc), axis=0), np.var(np.array(cross_val_acc), axis=0)
        subject_axis = [int(subject) for subject in self.subjects]

        fig, ax = plt.subplots(1)
        ax.plot(subject_axis, subject_means, '-o', label = 'mean')
        ax.fill_between(subject_axis, subject_means - subject_var, subject_means + subject_var, alpha = 0.2, label = 'variance')
        ax.grid()
        ax.set_xticks(subject_axis)
        ax.set_xlabel('subject')
        ax.set_ylabel('average accuracy')
        ax.legend()
        ax.set_title('Cross-Validated per subject decoding accuracies')
        
        fig_path = os.path.join(self.base_dir,"reports","figures", "ridge", self.training_scheme, self.lag_string)
        
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)

        fig.savefig(os.path.join(fig_path, "Cross_Val_subject_acc.png"))
        fig.savefig(os.path.join(fig_path, "Cross_Val_subject_acc.pdf"))

        trial_means, trial_var = np.mean(np.array(cross_val_acc), axis=1), np.var(np.array(cross_val_acc), axis=1)

    def eval_on_different_windows(self, model_id, val_indices_list, test_indices_list, window_sizes):
        """
        Evaluates Ridge model on different window sizes and save metrics as np.arrays.
        That models must have been trained before.

        Args:
            model_id (str): model identifier
            val_indices_list (list): validation indices (list of np.arrays)
            test_indices_list (list): test indices (list of np.arrays)
            window_sizes (list): list of window sizes in seconds
        """
        #check inputs
        assert isinstance(model_id, str), f"model_id must be string but is {type(model_id)}"
        assert isinstance(val_indices_list, list), f"val_indices must be list but is {type(val_indices_list)}"
        assert isinstance(test_indices_list, list), f"test_indices must be list but is {type(test_indices_list)}"
        assert all(isinstance(item, np.ndarray) for item in val_indices_list), f"val_indices must all be np.arrays"
        assert all(isinstance(item, np.ndarray) for item in test_indices_list), f"test_indices must all be np.arrays"
        assert isinstance(window_sizes, list), f"window_sizes must be list but is {type(window_sizes)}"
        assert all(isinstance(item, int) for item in window_sizes), f"window_sizes must all be integers"

        model_dir = os.path.join(self.base_dir, "models", "ridge", model_id)
        #check if model exists
        assert os.path.exists(model_dir), f"Model {model_id} does not exist"

        attended_scores, distractor_scores, accuracies = [],[],[]

        #calculate the metrics
        for acc_win in window_sizes:
            print(f'Entering window size {acc_win} s')
            attended_scores_win, distractor_scores_win, accuracies_win = [],[],[]
            for subject in self.subjects:
                print(f'Evaluating subject {subject}')
                path_subj = os.path.join(model_dir, subject)
                subj_attended_scores, subj_distractor_scores, subj_accuracies = [],[],[]

                for val_indices, test_indices in zip(val_indices_list, test_indices_list):
                    model = os.path.join(path_subj, str(test_indices.item()) + '.pk')
                    #check if model exists
                    assert os.path.exists(model), f"Model {model} does not exist"

                    #load model
                    mdl = pandas.read_pickle(model)
                    mdl.lagged_matrix_fill = 0
                    attended_score, distractor_score, accuracy = self.get_attended_scores_from_trained_mdl(mdl, subject, val_indices, test_indices, accuracy_window_size=acc_win)
                    
                    subj_attended_scores.append(attended_score)
                    subj_distractor_scores.append(distractor_score)
                    subj_accuracies.append(accuracy)
                
                attended_scores_win.append(subj_attended_scores)
                distractor_scores_win.append(subj_distractor_scores)
                accuracies_win.append(subj_accuracies)
            
            attended_scores.append(attended_scores_win)
            distractor_scores.append(distractor_scores_win)
            accuracies.append(accuracies_win)

        #save calculated metrics
        metrics_path = os.path.join(self.base_dir, 'reports', 'metrics', 'ridge', model_id)
        if not os.path.exists(metrics_path):
            os.makedirs(metrics_path)
        
        accuracies = np.array(accuracies)
        np.save(os.path.join(metrics_path, 'windowed_accuracies' + model_id + '.npy'), accuracies)

        attended_scores = np.array(attended_scores)
        np.save(os.path.join(metrics_path, 'windowed_attended_scores' + model_id + '.npy'), attended_scores)

        distractor_scores = np.array(distractor_scores)
        np.save(os.path.join(metrics_path, 'windowed_distractor_scores' + model_id + '.npy'), distractor_scores)


def eval_500_500():
    data_file = "ci_l_1_h_16_out_125.hdf5"
    eval_500_500 = RidgeEvaluator(-500, 500, data_file, training_scheme='concat', training_window_length=120)
    eval_500_500.eval_competing_speaker_cross_val()

def eval_500_500_32Hz():

    data_file = "ci_l_1,1_h_32,50_out125.hdf5"
    freq_range = '1-32Hz'

    eval_500_500_32 = RidgeEvaluator(-500, 500, database_filename= data_file, training_scheme='concat', freq_range= freq_range)
    eval_500_500_32.eval_competing_speaker_cross_val(book_test_set='different')

    eval_200_500_32 = RidgeEvaluator(-200, 500, database_filename=data_file, training_scheme='concat', freq_range= freq_range)
    eval_200_500_32.eval_competing_speaker_cross_val()

def eval_500_500_32Hz_diff_books():

    data_file = "ci_l_1,1_h_32,50_out_125_114.hdf5"
    freq_range = '1-32Hz'

    eval_500_500_32 = RidgeEvaluator(-500, 500, database_filename= data_file, training_scheme='concat', freq_range= freq_range)
    eval_500_500_32.eval_competing_speaker_cross_val(book_test_set='different')
    eval_500_500_32.eval_competing_speaker_cross_val(book_test_set='same')

def eval_ica_preprocessing():
    data_file = "ci_l_1,1_h_32,50_out_125_116_incl_ica.hdf5"
    freq_range = '1-32Hz'
    eval_500_500_32_ica = RidgeEvaluator(-500, 500, database_filename= data_file, training_scheme='concat', freq_range= freq_range, use_ica_data=True, model_id='001')
    eval_500_500_32_ica.eval_competing_speaker_cross_val(book_test_set='same')

def compare_to_cnn_hyperparamstudy():
    data_file = "ci_l_1,1_h_32,50_out_125_114.hdf5"
    freq_range = '1-32Hz'
    accuracy_window_size = 60

    eval_cnn_comp = RidgeEvaluator(-500, 500, database_filename= data_file, training_scheme='concat', freq_range= freq_range, model_id='002')

    test_trials = np.array([6,7,18,20])
    val_trials = np.array([10,11])

    test_indices, val_indices = test_trials - 1, val_trials - 1

    attended_scores, distractor_scores, accuracies = eval_cnn_comp.run_model_eval(val_indices=val_indices, test_indices=test_indices, accuracy_window_size=accuracy_window_size)
    eval_cnn_comp.plot_attention_scores(attended_scores, distractor_scores, test_indices)
    eval_cnn_comp.plot_accuracies(accuracies, test_indices)

def debug_story_models():
    data_file = "ci_l_1,1_h_32,50_out_125_116_incl_ica.hdf5"
    freq_range = '1-32Hz'
    accuracy_window_size = 60

    eval_cnn_comp = RidgeEvaluator(-500, 500, database_filename= data_file, training_scheme='concat', freq_range= freq_range, model_id='999')

    test_trials = np.array([12])
    val_trials = np.array([13])

    test_indices, val_indices = test_trials - 1, val_trials - 1

    elb_scores, pol_scores, labels, accuracies = eval_cnn_comp.run_story_model_eval(val_indices=val_indices, test_indices=test_indices, accuracy_window_size=accuracy_window_size, debug=True, attended_only=True)


if __name__ == '__main__':
    debug_story_models()