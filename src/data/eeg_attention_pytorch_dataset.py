import torch as t
import os
from torch.utils.data import Dataset
import h5py
import numpy as np
import git
from scipy.stats import zscore

#Note on indexing: subjects in the h5py file have identifiers [101,102,...] trials are numbers [1,2,3,...]
#Hence subj_index and trial_index are introduced with standard indexing [0,1,2,...]

class EegAttentionDataset(Dataset):

    def __init__(self, dir_h5, subj_string, trials_indices, window_size_training, sampling_rate = 125, zscore_eeg = True, training_window_offset = 0, ica=False, feature='env') -> None:
        """
        Intialize Dataset.
        Data will be stored in lists self.eeg, self.env_distractor, self.env_attended
        Those lists are of the structure [subject_101, subject_102,...],
        where subjects are lists of trials [trial_1, trial_2,...]
        each trial are torch tensors

        Args:
            dir_h5 (string): path to h5py dataset
            subj_string(string or list): subject identifier for subject: e.g. '108' or ['108','109']
            trials_indices(list): list of indices to take into the dataset for each subject. One row belongs to the corresponding subject. possible values in [0,1,2,3....,19]. e.g. [[0,1,2,3,4,5,6],[0,1,2,3,4,5,6,7],..], index 0 belongs to trial 1
            if a 1d list is provided, the same trials are used for all subjects
            window_size_training (int): number of data points used predict a single envelope data point. Is not the same as decision window_size used for classification
            sampling_rate (int, optional): Sampling rate in dataset in Hz. Defaults to 125.
            zscore_eeg (bool, optional): Whether to zscore eeg data. Defaults to False.
            training_window_offset (int, optional): Offset of training window in data points. Defaults to 0. Negative values are possible. 
                                                    For negative values the window is shifted to the left. Hence, taking datapoints before the stimulus into account.
            ica (bool, optional): Whether to use ica data. Defaults to False.
            feature (str, optional): Which feature to use. Defaults to 'env'. Currently 'env' and 'onset_env' are supported.
        """

        #handle single subject input
        if not isinstance(subj_string, list):
            subj_string = [subj_string]

        trials_are_lists = [isinstance(trials, list) for trials in trials_indices]
        assert (sum(trials_are_lists) == 0 or sum(trials_are_lists) == len(subj_string)), f"Trials indices must be lists or integers and cannot be mixed. Their number must match the number of subjects or be 1. Input: {trials_indices}"

        assert isinstance(training_window_offset, int), f"Training window offset must be integer. Input: {training_window_offset}"
        self.training_window_offset = training_window_offset

        #handle single trial input repeated for all subjects
        if sum(trials_are_lists) == 0:
            trials_indices = [trials_indices for _ in range(len(subj_string))]
        
        assert len(subj_string) == len(trials_indices), f"Number of subjects {len(subj_string)} does not match number of trial indices {len(trials_indices)}"

        #map indices to trials for 5hpy reading
        trials_int_list = [[elem + 1 for elem in row] for row in trials_indices]

        assert os.path.isfile(dir_h5),  f"{dir_h5} is not a file. Please provide valid directory."

        assert feature in ['env', 'onset_env'], f"Feature {feature} is not supported. Please choose from ['env', 'onset_env']"

        self.dir_h5 = dir_h5
        self.window_size_training = window_size_training
        self.sampling_rate = sampling_rate
        
        #mapping from index to (subject, trial, n_window)
        #https://discuss.pytorch.org/t/custom-dataset-with-getitem-method-that-requires-two-indices-as-input/62083
        self.index_map = {}
        
        #list of eeg data one entry is the matrix of one trial
        self.eeg = []
        
        #list of envelopes of stimulus data
        self.feat_distractor = []
        self.feat_attended = []
        
        #taken out indices were interpolated during preprocessing
        with h5py.File(dir_h5,'r') as f:
            #index for the overall dataset

            #it refers to the number of windows in the overall dataset
            dataset_index = 0
            subject_index = 0

            for subj, trials_int in zip(subj_string, trials_int_list):
                eeg_subject = []
                feat_attended_subject = []
                feat_distractor_subject = []
                #get trials, taken_out_indices are not relevant, sort them by the integer value
                assert all(trial_inputs in list(range(1,21)) for trial_inputs in trials_int), f"Some trials are outside the range [1,2,..20]. Input: {trials_int}"
                
                trial_strings = [str(trial_int) for trial_int in trials_int]

                #this index works only internally for the torch dataset is separate from experiment trial.
                trial_index = 0

                for trial_string in trial_strings:
                    if ica:
                        eeg_path = f'eeg_ica/{subj}/{trial_string}'
                    else:
                        eeg_path = f'eeg/{subj}/{trial_string}'
                    
                    eeg_trial = f[eeg_path][:]
                    assert eeg_trial.shape[0] == 33, "EEG measurement requires 33 channels. Loaded data is incorrect."

                    #drop auxiliary channels
                    eeg_trial = eeg_trial[:31,:]

                    if zscore_eeg:
                        eeg_trial = zscore(eeg_trial, axis=1)
                    
                    eeg_trial = t.from_numpy(eeg_trial)

                    #handle stimulus data
                    #data is stored as tensors of complete trials, accessing the individual windows is taken care of in __getitem__
                    stim_code = f[eeg_path].attrs['stimulus']
                    if feature == 'env':
                        feat_attended_path = f'stimulus_files/{stim_code}/attended_env'
                        feat_distractor_path = f'stimulus_files/{stim_code}/distractor_env'
                    elif feature == 'onset_env':
                        feat_attended_path = f'stimulus_files/{stim_code}/attended_onset_env'
                        feat_distractor_path = f'stimulus_files/{stim_code}/distractor_onset_env'
                    feat_attended, feat_distractor = t.from_numpy(f[feat_attended_path][:]), t.from_numpy(f[feat_distractor_path][:])

                    assert eeg_trial.shape[1] == feat_distractor.shape[0] == feat_attended.shape[0], f"Loaded data is not of matching shape in time. \n Got EEG: {eeg_trial.shape[1]}, Distractor envelope: {feat_distractor.shape[0]}, Attended envelope: {feat_attended.shape[0]}"

                    #add data to subject list
                    eeg_subject.append(eeg_trial)
                    feat_attended_subject.append(feat_attended)
                    feat_distractor_subject.append(feat_distractor)

                    #n_windows: number of complete windows of desired window size in current trial
                    n_data_points = eeg_trial.shape[1]
                    
                    assert n_data_points >= self.window_size_training, f"Window size {self.window_size_training} is larger than number of data points {n_data_points} in trial {trial_string} of subject {subj}"
                    assert training_window_offset < self.window_size_training, f"Training window offset {training_window_offset} is larger than window size {self.window_size_training}"

                    if training_window_offset <= 0:
                        n_windows = n_data_points - self.window_size_training
                        # stim index is the index of the stimulus data point that is predicted by the eeg data
                        # for negative lags the first n data points are not used
                        for stim_index in range(-training_window_offset, n_windows - training_window_offset):
                            self.index_map[dataset_index] = (subject_index,trial_index, stim_index)
                            dataset_index +=1
                    else:
                        
                        n_windows = n_data_points - self.window_size_training
                        #positive offset: don't use last n data points
                        for stim_index in range(0, n_windows - training_window_offset):
                            self.index_map[dataset_index] = (subject_index,trial_index, stim_index)
                            dataset_index +=1 

                    #fill dictionary that maps from index to subject, trial and window
                    #use the reverse mapping in __getitem__
                    # for window in range(0, n_windows):
                    #     self.index_map[dataset_index] = (subject_index,trial_index, window)
                    #     dataset_index +=1
                    
                    trial_index +=1
                
                #add data to dataset list
                self.eeg.append(eeg_subject)
                self.feat_distractor.append(feat_distractor_subject)
                self.feat_attended.append(feat_attended_subject)

                subject_index +=1

            #there is a +=1 after the last assignment, therefore dataset_index is one greater than last entry in self.index_map
            self.len = dataset_index
        f.close()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        """
        Gets one item from dataset: one window of eeg, and stimulus data
        Args:
            index (int): 

        Returns:
            tuple: eeg_data, attended envelope, distractor envelope
        """
        #handle negative indexes
        if abs(index) >= self.len:
            raise IndexError
        elif index < 0 :
            index += self.len
        
        if t.is_tensor(index):
            index = index.item()
        
        subject, trial, stim_index = self.index_map[index]

        start_index_eeg = stim_index + self.training_window_offset
        end_index_eeg = start_index_eeg + self.window_size_training


        eeg = self.eeg[subject][trial][:, start_index_eeg : end_index_eeg]
        feat_attended = self.feat_attended[subject][trial][stim_index]
        feat_distractor = self.feat_distractor[subject][trial][stim_index]

        return eeg, feat_attended, feat_distractor

