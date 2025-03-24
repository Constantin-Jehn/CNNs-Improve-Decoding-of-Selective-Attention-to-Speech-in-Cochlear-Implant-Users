import mne
from mne.preprocessing import ICA
from src.data.utils import *
import os
from os.path import join, dirname
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.io.wavfile import read as read_wav
import git
import h5py
from mne_icalabel import label_components
import json
import pickle

#Specify newly measured subjects
subjects = list(range(101,102))
aux_channels_correct = np.ones(1,dtype=int).tolist()
#subject 103 has switched aux channels
#aux_channels_correct[2] = 0


#hand picked ica components to exclude
ica_exclude = {'101': [0, 1, 2, 3, 4, 6],
'102': [0, 1, 2, 4, 5, 6, 7],
'103': [0, 1, 2, 3, 4, 13],
'104': [0, 1, 2, 3, 4, 5, 7, 9, 10],
'105': [0,1,2,3,4,5, 11, 12, 14],
'106': [0, 1, 2, 3, 4, 5, 6, 8, 10],
'107': [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 13, 17, 18],
'108': [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16],
'109': [0, 1, 2, 3, 4, 5, 7, 9],
'110': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 19],
'111': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14],
'112': [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13],
'113': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
'114': [0, 1, 2, 3, 4, 7, 8, 9, 10, 16],
}

ica_exclude_avg_ref = {
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
    '125': [0,1,2,3,8,9,12,16,18,23,29],
    '127': [0, 1, 2, 3, 4, 5, 6, 7, 9, 12, 13, 18],
    '128': [0, 1, 2, 4, 5, 11],
    '130': [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 19, 20]
    }

#using signal-cross-validation between ICs and audio signal
ica_exclude_avg_ref = {
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
    '125': [0,1,2,3,8,9,12,16,18,23,29],
    '127': [0, 1, 2, 3, 4, 5, 6, 7, 9, 12, 13, 18],
    '128': [0, 1, 2, 4, 5, 11],
    '130': [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 19, 20]
    }

base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
# path to the folder of ICA on previously artifact rejected data
ICA_ARTIFACT_PATH = os.path.join(base_dir, "data", "ica", "corr_criterium_eye")

# path to the folder on raw Data for only CI artifact rejection
ICA_CI_PATH = os.path.join(base_dir, "data", "ica", "corr_criterium")
del base_dir

class EegMeasurement():
    def __init__(self, subject, base_dir, aux_channels_correct = True, eeg_reference = 'avg', cohort='ci', align = True) -> None:
        """
        Abstraction of an Attention EEG Measurement, with useful functionalities as analysing drifts, alignmet and writing data to hdf5 dataset

        Args:
            subject (int): subject identifier
            base_dir (string): repository base directory
            aux_channels_correct (bool, optional): StimTrak connected to correct auxiliary channel. Defaults to True.
            eeg_reference (str, optional): EEG reference. Defaults to 'average'. Must be None or 'avg'
            cohort (str, optional): Cohort of subject. Defaults to 'ci'. Must be 'ci' (Cochlear Implant), 'control' (control group) or 'hi' (hearing impaired)
            align (bool, optional): Whether to align EEG and audio data. Defaults to True.
        """
        assert eeg_reference in [None, 'avg'], "eeg_reference must be None or 'avg'"
        self.eeg_reference = eeg_reference


        assert cohort in ['ci', 'control', 'hi'], "cohort must be 'ci', 'control' or 'hi'"
        self.cohort = cohort

        self.subject = subject
        self.base_dir = base_dir
        self.eeg_header_file = get_eeg_header_file(base_dir, subject)

        montage_file = join(base_dir, "data", "CACS-32_NO_REF.bvef")
        self.montage = mne.channels.read_custom_montage(montage_file)

        self.csv_log_file = get_csv_log_file(base_dir, subject)
        self.psychopy_log_file = read_csv(self.csv_log_file, sep=',')
        stimuli_base_folder = join(base_dir, "data", "stimuli")

        if 'randomisation' in self.psychopy_log_file.columns.tolist():
            self.randomisation = self.psychopy_log_file.loc[0,'randomisation']
        else:
            self.randomisation = int(input('Specifiy randomisation (0 or 1), because it is missing in log file'))

        self.stimuli_list = generate_stimuli_list(stimuli_base_folder, self.randomisation)

        self.eeg_recording = get_brainvision_from_header(self.eeg_header_file, self.montage)
        #rereference eeg_recording to average reference in place
        if self.eeg_reference == 'avg':
            mne.set_eeg_reference(self.eeg_recording, ref_channels='average', projection=False, copy = False)
        elif self.eeg_reference == None:
            #no rereferencing
            pass
        
        if align:
            triggers = get_triggers(self.eeg_recording, stimuli_list=self.stimuli_list)
            trigger_onsets= np.array([trigger['onset'] for trigger in triggers])

            #indices to align eeg and audio according to trigger signal
            self.trigger_indices = np.array(trigger_onsets * 1000, dtype = int)

            #just swap left and right channels if set-up was wrong
            if aux_channels_correct:
                self.stim_eeg_data_l, _ = self.eeg_recording['Aux1']
                self.stim_eeg_data_r, _ = self.eeg_recording['Aux2']
            else:
                self.stim_eeg_data_l, _ = self.eeg_recording['Aux2']
                self.stim_eeg_data_r, _ = self.eeg_recording['Aux1']


            #indices to align eeg and audio according to cross-correlation using the StimTrak signal
            self.corr_indices = get_lag_indicies_all_trials(self.stimuli_list, self.stim_eeg_data_l, self.stim_eeg_data_r)
            if any(self.corr_indices < 0):
                print(f"Correlation alignment failed for {self.subject}. Using trigger alignment instead.")
                self.corr_indices = self.trigger_indices
                self.using_trigger = True
            else:
                self.using_trigger = False

    def extract_experiment_info(self):
        """
        extracts important lag indices according to correlation, trigger, stimuli list, and stimulus data recorded through StimTrak
        saves subject summaries as .csv to data/processed/subject/
        """
        target_dir = join(self.base_dir,"data","processed", str(self.subject))
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        df_csv_log = self.psychopy_log_file.copy()
        #give more descriptive name
        df_csv_log.rename(columns={'attendende_story':'attended_story','key_exp_question_1.keys':'subject_answer_1','key_exp_question_1.corr':'answer_1_score','key_exp_question_2.keys':'subject_answer_2','key_exp_question_2.corr':'answer_2_score'},inplace=True)
        
        #unfortunately some answers in the template were wrong
        #correct the dataframe to get correct answers
        df_csv_log = correct_dataframe(df_csv_log)
        #df_csv_log = correct_dataframe(df_csv_log)

        # Get rows of trials, by selecting non-empty rows in colums attended story
        rows_trials = np.where(np.invert(np.array(pd.isna(df_csv_log.loc[:,'attended_story']))))[0]
        columns_trials = ['attended_story','attended_direction','answer_1','subject_answer_1','answer_1_score','answer_2','subject_answer_2','answer_2_score']
        trials = df_csv_log.loc[rows_trials,columns_trials].copy(deep=True)

        # Create dataframe for trials
        n_single, n_competing = 8 * 2, 12 * 2
        trials = trials.astype({'answer_1_score':'int16','answer_2_score':'int16'})
        trials.insert(loc=0,column="Trial", value= np.linspace(1,20,20,dtype=int))
        trials = trials.set_index('Trial')
        sum_trials_single, sum_trials_competing = trials.loc[0:8,['answer_1_score','answer_2_score']].sum(numeric_only=True).sum(), trials.loc[9:,['answer_1_score','answer_2_score']].sum(numeric_only=True).sum()

        # Extract additional info
        listening_effort = get_listening_effort(df_csv_log)
        comparisons = get_comparisons(df_csv_log)
        volume = get_volume(df_csv_log)

        # Create info dataframe
        info_df = pd.DataFrame(data = {'Volume':volume,
                            'Hoeranstrengung_1': listening_effort[0], 
                            'Hoeranstrengung_2': listening_effort[1],
                            'Hoeranstrengung_3': listening_effort[2],
                            'Hoeranstrengung_4': listening_effort[3],
                            'Geschichtenvergleich_Prädiktor': comparisons[0],
                            'Sprechervergleich' : comparisons[1],
                            'Geschichtenvergleich': comparisons[2],
                            'score_single_abs': sum_trials_single,
                            'score_single_rel': sum_trials_single / n_single,
                            'score_competing_abs': sum_trials_competing,
                            'score_competing_rel': sum_trials_competing / n_competing,
                            'score_overall_abs': sum_trials_single + sum_trials_competing,
                            'score_overall_rel': (sum_trials_single + sum_trials_competing) / (n_single + n_competing)}, index = [0])
        info_df = info_df.reset_index(drop=True)

        # save data as .csv in corresponding folder
        path_info, path_trials, path_corrrected_log = join(target_dir, str(self.subject) + '_info.csv'), join(target_dir, str(self.subject) + '_trials.csv'), join(target_dir, str(self.subject) + '_corrected_log.csv')
        #save corrected .csv
        df_csv_log.to_csv(path_corrrected_log)
        trials.to_csv(path_trials)
        info_df.to_csv(path_info)

    def analyse_alignment(self):
        figure_dir = join("reports","figures","alignment",str(self.subject))
        target_dir = join(self.base_dir,figure_dir)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        #Calculate the delays between trigger signals sent by PsychoPy and audio signal received by the EEG amp transmitted by the StimTrak. Unit: Milliseconds
        delays, delays_mean = self.corr_indices - self.trigger_indices, np.mean(self.corr_indices - self.trigger_indices)

        fig, (ax1) = plt.subplots(1,1, figsize = (8,5))
        fig.tight_layout(pad=3.0)
        ax1.plot(np.linspace(1,20,20),delays)
        ax1.plot(np.linspace(1,20,20),delays_mean.repeat(20), label = 'mean')
        ax1.set_xticks( np.linspace(1,20,20))
        ax1.set_xlabel('Trial', fontsize = 14)
        ax1.set_ylabel('ms', fontsize = 14)
        ax1.annotate(str(delays_mean), (0,0))
        ax1.set_title('Delay StrimTrak against Triggers - Subject: ' + str(self.subject), fontsize = 16)
        ax1.grid()
        ax1.legend(fontsize = 14)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.savefig(join(target_dir,"delay_stimtrak.pdf"))

        # **Compare start of audio on ms second level to very calculated delays***
        #Plot the very beginning of the stimuli
        for index in range(0,20):
            stimulus = self.stimuli_list[index]
            _, stim_wav_0 = read_wav(stimulus['path'])
            stim_wav_0_l, stim_wav_0_r = downsample_wav(stim_wav_0[:,0], stim_wav_0[:,1])

            if stimulus['side'] == 'r':
                stim_wav = stim_wav_0_r
                stim_eeg = self.stim_eeg_data_r[0,:]
            elif stimulus['side'] == 'l':
                stim_wav = stim_wav_0_l
                stim_eeg = self.stim_eeg_data_l[0,:]

            len_diff = len(stim_eeg) - len(stim_wav)
            stim_wav_aligned_trigger = np.pad(stim_wav,(self.trigger_indices[index],len_diff - self.trigger_indices[index]), mode = 'constant', constant_values= (0,0))
            stim_wav_aligned_corr = np.pad(stim_wav,(self.corr_indices[index],len_diff - self.corr_indices[index]), mode = 'constant', constant_values= (0,0))

            audio = np.where(stim_wav > 0.6e8)[0]
            start_audio = audio[0]
            
            #define window in ms to plot
            plot_window_bef, plot_window_after = 50, 200

            fig, (ax1,ax2) = plt.subplots(2,1, figsize = (10,8))
            ax1.plot(stim_wav_aligned_trigger[self.trigger_indices[index] + start_audio - plot_window_bef : self.trigger_indices[index] + start_audio + plot_window_after], label = 'aligned wav audio')
            ax1.plot(np.multiply(stim_eeg[self.trigger_indices[index] + start_audio - plot_window_bef : self.trigger_indices[index] + start_audio + plot_window_after], 4e8), label = 'eeg audio')
            ax1.legend(fontsize = 14)
            ax1.set_title("Alignment according to Trigger Trial: " + str(index +1), fontsize = 14)
            ax1.grid()
            plt.xticks(fontsize = 12)
            plt.yticks(fontsize = 12)

            ax2.plot(stim_wav_aligned_corr[self.trigger_indices[index] + start_audio - plot_window_bef:self.trigger_indices[index] + start_audio + plot_window_after], label = 'aligned wav audio')
            ax2.plot(np.multiply(stim_eeg[self.trigger_indices[index] + start_audio - plot_window_bef : self.trigger_indices[index] + start_audio + plot_window_after],4e8), label = 'eeg audio')
            ax2.set_title("Alignment according to Cross-Correlation Trial: " + str(index +1) + " (Start)", fontsize = 14)
            ax2.grid()
            ax2.legend(fontsize = 14)
            ax2.set_xlabel('ms', fontsize = 14)
            plt.xticks(fontsize = 12)
            plt.yticks(fontsize = 12)
            plt.savefig(join(target_dir,str(index + 1)+ "_start.pdf"))

        for index in range(0,20):
            stimulus = self.stimuli_list[index]
            _, stim_wav_0 = read_wav(stimulus['path'])
            stim_wav_0_l, stim_wav_0_r = downsample_wav(stim_wav_0[:,0], stim_wav_0[:,1])

            if stimulus['side'] == 'r':
                stim_wav = stim_wav_0_r
                stim_eeg = self.stim_eeg_data_r[0,:]
            elif stimulus['side'] == 'l':
                stim_wav = stim_wav_0_l
                stim_eeg = self.stim_eeg_data_l[0,:]

            len_diff = len(stim_eeg) - len(stim_wav)
            stim_wav_aligned_trigger = np.pad(stim_wav,(self.trigger_indices[index],len_diff - self.trigger_indices[index]), mode = 'constant', constant_values= (0,0))
            stim_wav_aligned_corr = np.pad(stim_wav,(self.corr_indices[index],len_diff - self.corr_indices[index]), mode = 'constant', constant_values= (0,0))

            audio = np.where(stim_wav > 0.6e8)[0]
            end_audio = audio[0]
            
            #define window in ms to plot
            plot_window_bef, plot_window_after = 200, 50

            fig, (ax1,ax2) = plt.subplots(2,1, figsize = (10,8))
            ax1.plot(stim_wav_aligned_trigger[self.trigger_indices[index] + end_audio - plot_window_bef : self.trigger_indices[index] + end_audio + plot_window_after], label = 'aligned wav audio')
            ax1.plot(np.multiply(stim_eeg[self.trigger_indices[index] + end_audio - plot_window_bef : self.trigger_indices[index] + end_audio + plot_window_after], 4e8), label = 'eeg audio')
            ax1.legend(fontsize = 14)
            ax1.set_title("Alignment according to Trigger Trial: " + str(index +1), fontsize = 14)
            ax1.grid()
            plt.xticks(fontsize = 12)
            plt.yticks(fontsize = 12)

            ax2.plot(stim_wav_aligned_corr[self.trigger_indices[index] + end_audio - plot_window_bef:self.trigger_indices[index] + end_audio + plot_window_after], label = 'aligned wav audio')
            ax2.plot(np.multiply(stim_eeg[self.trigger_indices[index] + end_audio - plot_window_bef : self.trigger_indices[index] + end_audio + plot_window_after],4e8), label = 'eeg audio')
            ax2.set_title("Alignment according to Cross-Correlation Trial: " + str(index +1) + " (End)", fontsize = 14)
            ax2.grid()
            ax2.legend(fontsize = 14)
            ax2.set_xlabel('ms', fontsize = 14)
            plt.xticks(fontsize = 12)
            plt.yticks(fontsize = 12)
            plt.savefig(join(target_dir,str(index + 1)+ "_end.pdf"))

    def analyse_drift(self):
        """Perform drift analysis on EEG measurement. Plot results and save them to data_pipeline/figures/drift"""
        drifts_subject_corr, drifts_subject_trigger = calc_drift_analysis(self.corr_indices, self.trigger_indices, self.stimuli_list, self.stim_eeg_data_l, self.stim_eeg_data_r)
        plot_drift_analysis(drifts_subject_corr, drifts_subject_trigger, self.subject, self.base_dir)

    def write_stimulus_data(self, data_filename, l_freq_env = 1.0, h_freq_env = 16.0, output_freq = 125):
        """
        Writes preprocessed stimulus data to file data_filename

        Args:
            data_filename (string): filename of h5py file where dataset is generated
            l_freq_env (float): Lower passband for High-Pass filter. Defaults to 1.0.
            h_freq_env (float): Upper passband for low pass filter. Defaults to 16.0.
            output_freq(int): Sampling frequency of envelope.
        """

        self.attended_envs = []
        self.distractor_envs = []

        data_file = join(self.base_dir, "data", "processed", data_filename)

        with h5py.File(data_file, 'a') as f:
            for stimulus in self.stimuli_list:
                wav_freq, stim_wav = read_wav(stimulus['path'])

                stim_wav_l, stim_wav_r = stim_wav[:,0], stim_wav[:,1]

                env_l, env_r = self.__estimate_envelope(stim_wav_l, stim_wav_r, wav_freq, l_freq_env = l_freq_env, h_freq_env = h_freq_env, output_freq = output_freq)
                onset_env_l, onset_env_r = self.__estimate_onset_envelope(env_l, env_r)

                #get locus side and trigger code for stimulus
                side, trigger_code  = stimulus['side'], stimulus['trigger_code']

                #save stimulus data
                path_attended_stim = f'stimulus_files/{trigger_code}/attended_wav'
                path_attended_env = f'stimulus_files/{trigger_code}/attended_env'
                path_attended_onset_env = f'stimulus_files/{trigger_code}/attended_onset_env'

                path_distractor_stim = f'stimulus_files/{trigger_code}/distractor_wav'
                path_distractor_env = f'stimulus_files/{trigger_code}/distractor_env'
                path_distractor_onset_env = f'stimulus_files/{trigger_code}/distractor_onset_env'
                
                #delete old data if it exists
                if path_attended_stim in f:
                    del f[path_attended_stim]
                if path_distractor_stim in f:
                    del f[path_distractor_stim]
                if path_distractor_onset_env in f:
                    del f[path_distractor_onset_env]
                
                if path_distractor_env in f:
                    del f[path_distractor_env]
                if path_attended_env in f:
                    del f[path_attended_env]
                if path_attended_onset_env in f:
                    del f[path_attended_onset_env]
                
                if side == 'l':
                    f.create_dataset(path_attended_stim, data = stim_wav_l)
                    f.create_dataset(path_attended_env, data = env_l)
                    f.create_dataset(path_attended_onset_env, data = onset_env_l)
                    self.attended_envs.append(env_l)

                    f.create_dataset(path_distractor_stim, data = stim_wav_r)
                    f.create_dataset(path_distractor_env, data= env_r)
                    f.create_dataset(path_distractor_onset_env, data = onset_env_r)
                    self.distractor_envs.append(env_r)

                elif side == 'r':
                    f.create_dataset(path_distractor_stim, data = stim_wav_l)
                    f.create_dataset(path_distractor_env, data = env_l)
                    f.create_dataset(path_distractor_onset_env, data = onset_env_l)
                    self.distractor_envs.append(env_l)

                    f.create_dataset(path_attended_stim, data = stim_wav_r)
                    f.create_dataset(path_attended_env, data = env_r)
                    f.create_dataset(path_attended_onset_env, data = onset_env_r)
                    self.attended_envs.append(env_r)
            f.close()

    def write_subjects_eeg_data(self, data_filename, l_freq_eeg = 1.0, h_freq_eeg = 16.0, output_freq = 125, use_ica=False, snr_thres=20, rm_bad_channels = False, rm_ci = False, rm_eye_muscle = False, log_ica_comp = False):
        """
        Adds eegs data of the eeg measurement to the h5py dataset under data/processed
        The snippets are aligned with the audio/envelope files

        Args:
            data_filename (string): filename of h5py file where dataset is generated
            l_freq_eeg (float, optional): Lower passband for High-Pass filter. Defaults to 1.0.
            h_freq_eeg (float, optional): Upper passband for low pass filter. Defaults to 16.0.
            output_freq(float, optional): Sampling frequency of EEG data in dataset
            use_ica (bool, optional): Whether to apply ICA to EEG data. Adds additional dataset with ica-cleaned EEG. Defaults to False.
            snr_thres(int, optional): ICA is used for CI participants, it defines the SNR peak threshold in ICA-components is required to remove that component
            rm_bad_channels (bool, optinal): whether to rm bad channels for the control group
            rm_ci (bool, optional): if ICA is used for CI participants, it defines whether CI artifacts should be removed
            m_eye_muscle  (bool, optional): if ICA is used for CI participants, it defines whether eye and muscle artifacts should be removed
        """

        data_file = join(self.base_dir, "data", "processed", data_filename)
        
        if log_ica_comp:
            ica_log = {}
        
        #check if stimulus was already added to dataset
        with h5py.File(data_file, 'a') as f:
            if not(f.__contains__('stimulus_files')):
                f.close()
                self.write_stimulus_data(data_filename)
            else:
                f.close()

        with h5py.File(data_file, 'a') as f:

            eeg_recording = self.eeg_recording.copy()

            if self.cohort == 'ci':
                #Typically one or two electrodes were taken out during data acquisition.
                taken_out_channels,_,taken_out_indices = get_taken_out_electrodes(eeg_recording)
                #Mark these channels as "bad" and interpolate the missing data
                taken_outs_txt_path = write_taken_out_channels_txt(taken_out_channels)
                eeg_recording.load_bad_channels(taken_outs_txt_path)
                eeg_recording.interpolate_bads(reset_bads=False, mode = 'accurate')
                os.remove(taken_outs_txt_path)
                #write taken out indices to hdf5 file
                path = f'eeg/{str(self.subject)}/taken_out_indices'
                if path in f:
                    del f[path]
                f.create_dataset(path, data = taken_out_indices)
                if use_ica:
                    path_ica = f'eeg_ica/{str(self.subject)}/taken_out_indices'
                    if path_ica in f:
                        del f[path_ica]
                    f.create_dataset(path_ica, data = taken_out_indices)
                
                if use_ica:
                    # #load ica if it exists
                    # ica = self.__load_ica()
                    # #apply ica
                    eeg_recording_ica = eeg_recording.copy()
                    #perform trial level ica further down

                    # if ica is None:
                    #     filt_raw = eeg_recording_ica.copy().filter(l_freq=1.0, h_freq=100.0)
                    #     ica = ICA(n_components=30, max_iter='auto', random_state=1337, method='fastica')
                    #     filt_raw.drop_channels(['Aux1','Aux2'])
                    #     filt_raw.info['bads'] = []
                    #     ica.fit(filt_raw)
                    # else:
                    #     #nothing to do here ica is already loaded
                    #     pass
                    # #exclude predifined components
                    # #depends on chosen reference
                    # if self.eeg_reference == 'avg':
                    #     ica.exclude = ica_exclude_avg_ref[str(self.subject)]
                    # elif self.eeg_reference is None:
                    #     ica.exclude = ica_exclude[str(self.subject)]
                    
                    # eeg_recording_ica.info['bads'] = []
                    # ica.apply(eeg_recording_ica)
            
            elif self.cohort == 'control':
                #identify bad channels using lof algorithm
                if rm_bad_channels:
                    filt_raw_lof = eeg_recording.copy().filter(l_freq=1.0, h_freq=32.0)
                    filt_raw_lof.drop_channels(['Aux1','Aux2'])
                    bad_channels_control, _ = mne.preprocessing.find_bad_channels_lof(filt_raw_lof, n_neighbors = 10, threshold=1.6, return_scores= True)
                    eeg_recording.info['bads'] = bad_channels_control
                    #eeg_recording.drop_channels(bad_channels_control)
                if use_ica:
                    #load ica if it exists
                    ica = self.__load_ica()
                    eeg_recording_ica = eeg_recording.copy()
                    eeg_recording_ica.info['bads'] = bad_channels_control
                    # eeg_recording_ica.info['bads'] = bad_channels_control
                    
                    # drop bad_channels (they were also rejected beforing calculating ica) on temporary file
                    if rm_bad_channels:
                        eeg_recording_ica_tmp = eeg_recording_ica.copy().drop_channels(bad_channels_control)
                    else:
                        eeg_recording_ica_tmp = eeg_recording_ica.copy()
                    eeg_recording_ica_tmp.filter(l_freq=1.0, h_freq = 100)
                    
                    if ica is None:
                        ica = ICA(n_components=20, max_iter='auto', random_state=1337, method="infomax")
                        ica.fit(eeg_recording_ica_tmp)
                    
                    # label bad channels using ic-label
                    ic_labelling = label_components(eeg_recording_ica_tmp, ica, method='iclabel')
                    labels, probabilities = ic_labelling['labels'], ic_labelling['y_pred_proba']

                    # exclude components that are not 'brain' or 'other' and have a probability of more than 0.7
                    exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"] and probabilities[idx] > 0.7]
                    ica.apply(eeg_recording_ica, exclude = exclude_idx)

                #after ica interpolating bad channels
                eeg_recording.interpolate_bads(reset_bads=False, mode = 'accurate', method = 'spline')
                eeg_recording_ica.interpolate_bads(reset_bads=False, mode = 'accurate', method = 'spline')
            

            for stimulus, lag_index, trial in zip(self.stimuli_list, self.corr_indices, range(1,21)):
                if log_ica_comp:
                    ica_log[str(trial)] = {}
                #get length of section (at 125 Hz) from envelope
                trigger_code = stimulus['trigger_code']
                path_attended_env = f'stimulus_files/{trigger_code}/attended_env'
                stimulus_env = f[path_attended_env][:]
                len_env = len(stimulus_env)
                
                #get length of section (at 1000Hz) 
                _, stim_wav = read_wav(stimulus['path'])
                #downsamples to 1kHz by default
                stim_wav_l, _ = downsample_wav(stim_wav[:,0], stim_wav[:,1])
                len_snippet = len(stim_wav_l)

                #crop eeg according to alignment, preprocess it and extract data array
                #crop before preprocessing to get more preciese alignment (preprocessing involves downsampling)
                eeg_snippet = eeg_recording.copy().crop(tmin = lag_index / 1000.0, tmax = (lag_index + len_snippet) / 1000.0)

                if use_ica and self.cohort == 'ci':
                    eeg_snippet_ica = eeg_recording_ica.copy().crop(tmin = lag_index / 1000.0, tmax = (lag_index + len_snippet) / 1000.0)
                    # load trial ica
                    
                    if rm_eye_muscle:
                        # remove eye artifacts
                        ica = self.__load_trial_ica(trial, eye_muscle_removed = False)

                        eeg_ica_tmp = eeg_snippet_ica.copy()
                        # filter for adequate usage of ic-label
                        eeg_ica_tmp.filter(l_freq=1.0, h_freq = 100)
                        ic_labelling = label_components(eeg_ica_tmp, ica, method='iclabel')
                        labels, probabilities = ic_labelling['labels'], ic_labelling['y_pred_proba']
                        if log_ica_comp:
                            ica_log[str(trial)]['labels'] = labels
                            ica_log[str(trial)]['probabilities'] = list(probabilities)
                        exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"] and probabilities[idx] > 0.40]
                        bads = eeg_snippet_ica.info['bads']
                        eeg_snippet_ica.info['bads'] = []
                        ica.apply(eeg_snippet_ica, exclude = exclude_idx)
                        eeg_snippet_ica.info['bads'] = bads

                    if rm_ci:
                        ica = self.__load_trial_ica(trial, eye_muscle_removed = rm_eye_muscle)
                        # remove CI artifacts
                        # load peaking components from json
                        peaking_channels = self.__load_ica_exclude(trial, snr_thres = snr_thres)
                        ica.exclude = peaking_channels
                        # apply ica to eeg snippet
                        bads = eeg_snippet_ica.info['bads']
                        eeg_snippet_ica.info['bads'] = []
                        ica.apply(eeg_snippet_ica)
                        eeg_snippet_ica.info['bads'] = bads
                
                #preprocess bandpass filter and downsample
                eeg_snippet = self.__preprocess_eeg(eeg_snippet, l_freq_eeg = l_freq_eeg, h_freq_eeg = h_freq_eeg, output_freq = output_freq)
                eeg_data = eeg_snippet.get_data()

                if use_ica:
                    eeg_snippet_ica = self.__preprocess_eeg(eeg_snippet_ica, l_freq_eeg = l_freq_eeg, h_freq_eeg = h_freq_eeg, output_freq = output_freq)
                    eeg_data_ica = eeg_snippet_ica.get_data()

                    assert eeg_data.shape == eeg_data_ica.shape, f"Shape of eeg data ({eeg_data.shape}) and eeg data after ica ({eeg_data_ica.shape}) not matching"

                #up and downsamping can lead to very small deviation in signal length
                len_eeg_data = eeg_data.shape[1]
                difflen_env_eeg = len_eeg_data - len_env

                assert abs(difflen_env_eeg) < 3, f"Processed EEG data ({eeg_data.shape}) not matching length of envelope ({len_env})"

                #for smalle deviations (less than 4ms) in length adjust eeg data's shape to match envelope
                if difflen_env_eeg > 0:
                    eeg_data = eeg_data[:,:len_env]
                    if use_ica:
                        eeg_data_ica = eeg_data_ica[:,:len_env]
                elif difflen_env_eeg < 0:
                    eeg_data = np.pad(eeg_data, (0,-difflen_env_eeg), 'constant', constant_values = (0,0))
                    if use_ica:
                        eeg_data_ica = np.pad(eeg_data_ica, (0,-difflen_env_eeg), 'constant', constant_values = (0,0))


                #write eeg data to file
                path = f'eeg/{str(self.subject)}/{str(trial)}'
                #overwrite data if it already exists
                if path in f:
                    del f[path]
                f.create_dataset(path, data = eeg_data)
                f[path].attrs['stimulus'] = int(stimulus['trigger_code'])
                #mne info file is useful for later analysis especially topo plots
                #info_path = f'eeg/{str(self.subject)}/info'
                #f[path].attrs['mne_info'] = eeg_recording.info

                if use_ica:
                    #write ica eeg data to file
                    path_ica = f'eeg_ica/{str(self.subject)}/{str(trial)}'
                    if path_ica in f:
                        del f[path_ica]
                    f.create_dataset(path_ica, data = eeg_data_ica)
                    f[path_ica].attrs['stimulus'] = int(stimulus['trigger_code'])
                    # f[path_ica].attrs['stimulus'] = int(stimulus['trigger_code'])
                    # f[path_ica].attrs['mne_info'] = eeg_recording_ica.info
            if log_ica_comp:
                path_log = join(self.base_dir, "data", "processed", f'ica_log_{str(self.subject)}.pkl')

                #save using pickle
                with open(path_log, 'wb') as f_log:
                    pickle.dump(ica_log, f_log)



            print(f'Subject {str(self.subject)} added to dataset')
            f.close()

    def __preprocess_eeg(self, mne_raw:mne.io.Raw, l_freq_eeg, h_freq_eeg, output_freq):
        """
        Low-Pass - Resample - High-Pass (Avoiding drifts)
        The mne_raw object is modified in place!
        Provide a copy of the original data.

        Args:
            mne_raw (mne.io.Raw): object containting EEG data
            l_freq_eeg (float, optional): Lower passband for High-Pass filter.
            h_freq_eeg (float, optional): Upper passband for low pass filter
            output_freq (int, optional): Sampling frequency of output.

        Returns:
            mne.io.Raw: Preprocesse EEG-object
        """

        mne_raw.filter(None, h_freq_eeg, verbose = False)
        mne_raw.resample(output_freq, verbose = False)
        mne_raw.filter(l_freq_eeg, None, verbose = False)

        return mne_raw

    def __estimate_envelope(self, stim_wav_l, stim_wav_r, wav_freq, l_freq_env, h_freq_env, output_freq):
        """
        Estimates the speech envelope of given stimulus.
        Args:
            stim_wav_l (np.array): left channel stimulus
            stim_wav_r (np.array): right channel stimulus
            wav_freq (_type_): sampling frequency of audio
            l_freq_env (float, optional): Lower passband for High-Pass filter.
            h_freq_env (float, optional): Upper passband for low pass filter.
            output_freq (int, optional): sampling frequency of output envelope

        Returns:
            (np.array, np.array): (envelope_left, envelope_right)
        """

        #envelope computation
        env_l, env_r = np.abs(hilbert(stim_wav_l)), np.abs(hilbert(stim_wav_r))
        env_l, env_r = mne.filter.filter_data(env_l, wav_freq, l_freq_env, h_freq_env, verbose = False), mne.filter.filter_data(env_r, wav_freq, None, 50, verbose= False)
        
        #Resample and normalize
        env_l, env_r = mne.filter.resample(env_l, output_freq, wav_freq, verbose = False), mne.filter.resample(env_r, output_freq, wav_freq, verbose = False)
        env_l, env_r = zscore(env_l, nan_policy='raise'), zscore(env_r, nan_policy = 'raise')

        return env_l, env_r
    
    def __estimate_onset_envelope(self, env_l, env_r):
        """
        Calculates the onset envelope of the given envelope signals
        Args:
            env_l (np.array): left envelope
            env_r (np.array): right envelope

        Returns:
            onset_env_l, onset_env_r: onset envelope left, onset envelope right
        """
        #use append flag to get outputs of same shape
        onset_env_l, onset_env_r = np.diff(env_l, append=env_l[-1]), np.diff(env_r, append=env_r[-1])
        onset_env_l[onset_env_l<0] = 0
        onset_env_r[onset_env_r<0] = 0
        return onset_env_l, onset_env_r
    
    def __load_ica(self):
        """
        Loads ICA object from file if it exists, otherwise returns None

        Returns:
            ica: mne preprocessing ICA object or None
        """
        if self.cohort=='ci' and self.eeg_reference == 'avg':
            ica_path = os.path.join(self.base_dir, "data", "ica", "re_ref", str(self.subject), str(self.subject) + 'ica')
        elif self.cohort=='ci' and self.eeg_reference == None:
            ica_path = os.path.join(self.base_dir, "data", "ica", "no_re_ref", str(self.subject), str(self.subject) + 'ica')
        elif self.cohort=='control':
            ica_path = os.path.join(self.base_dir, "data", "ica", "preprocessed", str(self.subject), 'ica')
        #check if ica already exists
        if os.path.exists(ica_path):
            ica = mne.preprocessing.read_ica(ica_path)
            return ica
        else:
            return None

    def __load_trial_ica(self, trial, eye_muscle_removed = False):
        """
        Load ICA object for a specific trial if it exists, otherwise returns None

        Args:
            trial (int): trial number
            eye_removed (bool): Whether ica from data where eye artifacts were already removed should be loaded
        """
        assert trial in range(1,21), "Trial must be in range 1-20"
        ica_file_name = f'ica{str(self.subject)}_{str(trial)}.fif'
        if eye_muscle_removed:
            ica_path = os.path.join(ICA_ARTIFACT_PATH, str(self.subject), str(trial), ica_file_name)
        else:
            ica_path = os.path.join(ICA_CI_PATH, str(self.subject), str(trial), ica_file_name)

        if os.path.exists(ica_path):
            ica = mne.preprocessing.read_ica(ica_path)
            return ica
        else:
            return None

    def __load_ica_exclude(self, trial, snr_thres=20, eye_muscle_removed = False):
        """
        Load ICA components to exclude according to SNR > 20dB criterion.
        The criteron is based on SNR of ICA time component to audio mixture of presented stimuli
        """
        assert trial in range(1,21), "Trial must be in range 1-20"
        exclude_file_name = f'subj{str(self.subject)}_SNR_analysis.json'
        if eye_muscle_removed:
            exclude_path = os.path.join(ICA_ARTIFACT_PATH, str(self.subject), exclude_file_name)
        else:
            exclude_path = os.path.join(ICA_CI_PATH, str(self.subject), exclude_file_name)
        #open json file
        if os.path.exists(exclude_path):
            with open(exclude_path) as f:
                exclude_dict = json.load(f)
                snrs = exclude_dict[f'trial_{str(trial)}']['snr']
                snrs = np.array(snrs)
                peaking_channels = np.where(snrs > snr_thres)[0]
                peaking_channels = list(peaking_channels)
                f.close()
            return peaking_channels
        else:
            return None



if __name__ == '__main__':
    base_dir = base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    test = EegMeasurement(103, base_dir, aux_channels_correct = True, eeg_reference = None)
    #test.write_stimulus_data(data_filename='test_ci.hdf5')
    test.write_subjects_eeg_data(data_filename='test_ci.hdf5', use_ica=True, snr_thres=15)
    pass