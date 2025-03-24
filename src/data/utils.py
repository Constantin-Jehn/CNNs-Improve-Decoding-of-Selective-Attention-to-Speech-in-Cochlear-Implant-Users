"""
    File containing utils for aligning audio and extacting experimental data. 
    Specifically for the auditory attention experiment conducted at the SCIC Dresden.
    Written by Constantin Jehn (Chair for Neurosensory Engineering, FAU)
"""

import mne
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from scipy.io.wavfile import read as read_wav
from scipy.signal import decimate, correlate, correlation_lags, hilbert
from scipy.stats import sem, linregress, zscore
from pandas import read_csv
import pandas as pd
from os.path import join
import os
import math
import collections

def get_brainvision_from_header(header_file, montage):
    """return mne brainvision file for given header file of eeg measurement
    Args:
        header_file (string): .vhdr header file
        montag: mne montage
    Returns:
        _type_: brainvision object
    """
    brainvion_object = mne.io.read_raw_brainvision(header_file, misc=['Aux1','Aux2'],preload=True)
    brainvion_object.set_channel_types({'Aux1':'stim'})
    brainvion_object.set_channel_types({'Aux2':'stim'})
    brainvion_object.set_montage(montage)
    return brainvion_object

def downsample_wav(stim_wav_l, stim_wav_r):
    """Downsample .wav file from 48kHz to 1kHz
    Args:
        stim_wav_l (_type_): left channel of audio file
        stim_wav_r (_type_): right channel of audio file
    Returns:
        _type_: downsampled left channel, downsamples right channel
    """
    stim_wav_l_dec, stim_wav_r_dec = decimate(stim_wav_l,8), decimate(stim_wav_r,8)
    return decimate(stim_wav_l_dec,6), decimate(stim_wav_r_dec,6)

def align_stim_wav(stim_eeg, stim_wav, plot = False, fname = 'audio_alignment.pdf'):
    """correlates .wav stimulus with eeg stim channel and return index of offset and aligned .wav stimulus
    EEG stimulus must be at least as long as .wav stimulus
    Args:
        stim_eeg (np.array): stim channel of eeg data
        stim_wav (np.array): high-quality .wav audio stimulus
    Returns:
        tuple: (index of lag, aligned .wav stimulus)
    """
    assert len(stim_eeg) >= len(stim_wav), "EEG stimulus must be at least as long as .wav stimulus"

    len_diff = len(stim_eeg) - len(stim_wav)
    stim_wav_padded = np.pad(stim_wav,((0,len_diff)), mode = 'constant', constant_values = (0,0))
    corr = correlate(stim_eeg, stim_wav_padded, mode = 'full')
    lags = correlation_lags(stim_eeg.size, stim_wav_padded.size, mode='full')
    lag_index = int(lags[np.argmax(np.abs(corr))])
    #lag_index = int(lags[np.argmax(corr)])

    #flag if the correlation peak is sufficient
    if np.max(np.abs(corr)) >  6 * 10 ** 11:
        confident = True
    else:
        confident = False
    if lag_index < 0:
        stim_wav_aligned = None
        confident = False
    else:
        stim_wav_aligned = np.pad(stim_wav,(lag_index,len_diff - lag_index), mode = 'constant', constant_values= (0,0))
    if plot:
        plot_alignmet(stim_eeg, stim_wav, stim_wav_aligned, corr, lags, fname)
        plot_alignment_precision(corr,lags,'detail_' + fname)
    return lag_index, stim_wav_aligned, confident

def get_eeg_offset(df_csv_log, stim_eeg, first_stim_wav, end_of_window, eeg_freq = 1000):
    """Calculates time difference between start of PsychoPy and EEG measurement using the csv log file of psychopy
    Note: the difference is only approximatly. Better use triggers.
    Args:
        df_csv_log (_type_): .csv log file as pandas data frame
        stim_eeg (_type_): stim channel of EEG data
        first_stim_wav (_type_): First Audio stimulus
        end_of_window (_type_): End of window in stim channel to look for correlation with first audio stimulus in seconds
        eeg_freq (int): Sampling frequency of the EEG recorder in Hz

    Returns:
        _type_: _description_
    """
    #hard-coded: position in .csv file always identical
    try:
        psychopy_first_stim = df_csv_log.loc[6,'sound_Intro_Elb_1_2.started']
    except:
        raise Warning("Starting time of first stimulus is missing in log file. Setting to 0")
        psychopy_first_stim = 0

    #calc timelag between start of EEG measurement and first stimulus
    stim_eeg_snippet = stim_eeg[0,:eeg_freq * end_of_window]
    lag_index, _ = align_stim_wav(stim_eeg_snippet, first_stim_wav)
    eeg_first_stim = lag_index / eeg_freq

    #return offset between start of PsychoPy and EEG measurement
    return psychopy_first_stim - eeg_first_stim

def plot_alignmet(stim_eeg, stim_wav, stim_wav_aligned, corr, lags, fname):
    seconds = np.linspace(0, stim_eeg.size/1000, num = stim_eeg.size)
    lag_seconds = np.linspace(-stim_eeg.size/1000, stim_eeg.size/1000, lags.size)
    len_diff = len(stim_eeg) - len(stim_wav)
    stim_wav = np.pad(stim_wav,((0,len_diff)), mode = 'constant', constant_values = (0,0))
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, figsize = (20,15))
    fig.tight_layout(pad=5.0)
    ax1.plot(seconds, stim_eeg)
    ax1.set_title('EEG audio')
    ax1.grid()

    ax2.plot(seconds, stim_wav_aligned)
    ax2.set_title('wav audio aligned')
    ax2.set_xlabel('time [s]')
    ax2.grid()

    ax3.plot(seconds, stim_wav)
    ax3.set_title('wav audio')
    ax3.set_xlabel('time [s]')
    ax3.grid()

    ax4.plot(lag_seconds, corr)
    ax4.set_title('Correlation')
    ax4.set_xlabel('time lags [s]')
    ax4.set_ylabel('Pearsons r')
    ax4.grid()

    #plt.show()
    plt.savefig(join('figures',fname))
    #plt.show(block = True)

def plot_alignment_precision(corr, lags,fname, window = 15, eeg_freq = 1000):
    seconds = np.linspace(0, corr.size/eeg_freq, num = corr.size)
    lag_seconds = np.linspace(-np.ceil(corr.size/2)/eeg_freq, np.ceil(corr.size/2)/eeg_freq, lags.size)
    max_index = np.argmax(np.abs(corr))

    fig, (ax1, ax2) = plt.subplots(2,1, figsize = (12,8))
    fig.tight_layout(pad=5.0)
    ax1.plot(lag_seconds, np.abs(corr))
    ax1.set_title('Absolute Correlation')
    ax1.set_xlabel('time lags [s]')
    ax1.set_ylabel(r"$\|$Pearsons r$\|$")
    ax1.grid()

    ax2.plot(lag_seconds[max_index-window : max_index + window], np.abs(corr[max_index - window : max_index + window]))
    ax2.set_title('Correlation Peak')
    ax2.set_xlabel('time lags [s]')
    ax2.set_ylabel(r"$\|$Pearsons r$\|$")
    ax2.vlines(lag_seconds[max_index],0, np.abs(corr[max_index]), color = 'black', label = 'max_index')
    ax2.set_xticks(lag_seconds[max_index- window : max_index + window : 5])
    ax2.legend()
    ax2.grid()

    #plt.show()
    plt.savefig(join('figures',fname))

def get_triggers(brainvision_object, stimuli_list):
    """
    Extracts the sent triggers from PsychoPy to the EEG amplifier from raw brainvision object
    Args:
        brainvision_object (_type_): brainvision object of EEG recordingƒ
    Returns:
        list: list of ordered Dicts containing triggers
    """
    annotations = brainvision_object.annotations
    triggers = [annotation for annotation in annotations if annotation['description'][:8] == 'Stimulus']
    #if same annotation appear twice, take the first one, edge case 124 trigger 225, 226 appear twice
    if len(triggers) > 20:
        Warning('More than 20 triggers identified. Probably some triggers were repeated. Trying to remove duplicates.')
        descriptions = [trigger['description'] for trigger in triggers]
        duplicates = [item for item, count in collections.Counter(descriptions).items() if count == 2]
        duplicate_indices = [index for index, value in enumerate(descriptions) if value in duplicates]
        deletion_indices = duplicate_indices[int(len(duplicate_indices)/2):]
        #take first occurence of duplicate
        triggers = [trigger for index, trigger in enumerate(triggers) if index not in deletion_indices]
    elif len(triggers) < 20:
        Warning('Less than 20 triggers identified. Probably some triggers were missing.')
        # valid_trigger_codes = ['111', '112', '113', '114', '121', '122', '123', '124', '215', '216', '217', '218', '219', '210', '225', '226', '227', '228', '229', '220']
        # valid_trigger_despcriptions = ['Stimulus/S' + code for code in valid_trigger_codes]
        # # check for missing triggers
        # missing_triggers = [trigger for trigger in valid_trigger_despcriptions if trigger not in [trigger['description'] for trigger in triggers]]
        # # indices of missing triggers
        # missing_indices = [valid_trigger_despcriptions.index(trigger) for trigger in missing_triggers]
        
        # trigger_list_experiment = [stim['stimuli_trigger_codes'] for stim in stimuli_list]
    
    if len(triggers) > 20:
        #if still more than 20 triggers, check for invalid triggers
        Warning('More than 20 triggers identified. Checking for invalid triggers.')
        valid_trigger_codes = ['111', '112', '113', '114', '121', '122', '123', '124', '215', '216', '217', '218', '219', '210', '225', '226', '227', '228', '229', '220']
        valid_trigger_despcriptions = ['Stimulus/S' + code for code in valid_trigger_codes]
        triggers = [trigger for trigger in triggers if trigger['description'] in valid_trigger_despcriptions]

    assert len(triggers) == 20, 'too many triggers identified'
    return triggers

def get_lag_indicies_all_trials(stimuli_list, stim_eeg_data_l, stim_eeg_data_r):
    """Returns list of indices where stimuli in 'stimuli_list' start in the EEG data. 
    Sampling Rate is 1kHz --> multiply by 1000 to get time in seconds
    Args:
        stimuli_list (_type_): list of stimuli as ordered dicts
        stim_eeg_data_l (_type_): left channel of EEG data
        stim_eeg_data_r (_type_): rigt channel of EEG data

    Returns:
        np.array: list of indices where stimuli start
    """
    lag_indicies = []
    n = 0
    for stimulus in stimuli_list:
        _, stim_wav_0 = read_wav(stimulus['path'])
        #take part of wav file according to dominant side
        if stimulus['side'] == 'r':
            stim_wav_l, stim_wav_r = downsample_wav(stim_wav_0[:,0], stim_wav_0[:,1])
            lag_index, _, confident = align_stim_wav(stim_eeg_data_r[0,:] ,stim_wav_r, plot = False)
            #for low correlation coefs in competing speaker (n>7), take other side of stereo signal
            if not confident and n > 7:
                lag_index, _, confident = align_stim_wav(stim_eeg_data_l[0,:] ,stim_wav_l, plot = False)
            
        elif stimulus['side'] == 'l':
            stim_wav_l, stim_wav_r = downsample_wav(stim_wav_0[:,0], stim_wav_0[:,1])
            lag_index, _, confident = align_stim_wav(stim_eeg_data_l[0,:] ,stim_wav_l, plot = False)
            if not confident and n > 7:
                lag_index, _, confident = align_stim_wav(stim_eeg_data_r[0,:] ,stim_wav_r, plot = False)
        if not confident:
            lag_index = -1
            #print(f'No confident alignment found for stimulus {stimulus["path"]}. Falling back to trigger alignment.')
        n += 1
        lag_indicies.append(lag_index)
    return np.array(lag_indicies)

def generate_stimuli_list(stimuli_base_folder, randomisation):
    """Reads in Stimuli Files from our Attention experiment. 
    Creates a list of ordered dicts of the stimuli of the form. Length = 20
    

    Args:
        stimuli_base_folder (String): Base Folder where stimuli are organised, with subfolder Elbenwald_Competing, Elbenwald_Single_Speaker etc.
        randomisation (int): Randomisation of the experiment that determines the order of the presented stimuli 0 or 1

    Returns:
        list: of organized dicts of the stimuli:
        [{'path': path to the audio file, 'trigger_code': Code of that stimulus sent as trigger to the EEG-amp, 'side': single-speaker: presentation side competing speaker: focus side}]
    """

    stimuli_folders = [os.path.join(stimuli_base_folder, folder) for folder in os.listdir(stimuli_base_folder) if os.path.isdir(os.path.join(os.path.join(stimuli_base_folder, folder)))]
    stimuli_paths = []

    for stimuli_folder in stimuli_folders:
        for stimuli_name in os.listdir(stimuli_folder):
            if not stimuli_name.startswith('.'):
                stimuli_paths.append(os.path.join(stimuli_folder, stimuli_name))

    #get only the file names
    stimuli_files = np.array([stimuli_path.split('/') for stimuli_path in stimuli_paths])[:,-1]

    #experimental file orders for the two randomisations
    if randomisation == 1:
        stimuli_files_ordered_rand= ['Polarnacht_Focus_FR_1.wav', 'Polarnacht_Focus_FL_2.wav','Elbenwald_FR_1.wav','Elbenwald_FL_2.wav', 
                                'Polarnacht_Focus_FR_3.wav', 'Polarnacht_Focus_FL_4.wav', 'Elbenwald_FR_3.wav','Elbenwald_FL_4.wav', 
                                'Polarnacht_FR_5.wav','Polarnacht_FL_6.wav', 'Elbenwald_FR_5.wav', 'Elbenwald_FL_6.wav', 
                                'Polarnacht_FR_7.wav', 'Polarnacht_FL_8.wav', 'Elbenwald_FR_7.wav', 'Elbenwald_FL_8.wav',
                                'Polarnacht_FR_9.wav', 'Polarnacht_FL_10.wav','Elbenwald_FR_9.wav', 'Elbenwald_FL_10.wav']
    elif randomisation == 0:
        stimuli_files_ordered_rand = ['Elbenwald_FR_1.wav','Elbenwald_FL_2.wav','Polarnacht_Focus_FR_1.wav', 'Polarnacht_Focus_FL_2.wav','Elbenwald_FR_3.wav','Elbenwald_FL_4.wav', 
                                    'Polarnacht_Focus_FR_3.wav', 'Polarnacht_Focus_FL_4.wav', 'Elbenwald_FR_5.wav', 'Elbenwald_FL_6.wav','Polarnacht_FR_5.wav','Polarnacht_FL_6.wav',
                                    'Elbenwald_FR_7.wav', 'Elbenwald_FL_8.wav', 'Polarnacht_FR_7.wav', 'Polarnacht_FL_8.wav', 'Elbenwald_FR_9.wav', 'Elbenwald_FL_10.wav',
                                    'Polarnacht_FR_9.wav', 'Polarnacht_FL_10.wav']

    #reorder stimuli_pahts according to experimal order
    indices = [stimuli_files.tolist().index(stim_file) for stim_file in stimuli_files_ordered_rand]
    stimuli_paths = [stimuli_paths[i] for i in indices]

    #corespondance of filenames to trigger codes
    stimuli_trigger_codes = {'Elbenwald_FR_1.wav' : '111',
                        'Elbenwald_FL_2.wav' : '112',
                        'Elbenwald_FR_3.wav' : '113',
                        'Elbenwald_FL_4.wav' : '114',
                        'Elbenwald_FR_5.wav' : '215',
                        'Elbenwald_FL_6.wav' : '216',
                        'Elbenwald_FR_7.wav' : '217',
                        'Elbenwald_FL_8.wav' : '218',
                        'Elbenwald_FR_9.wav' : '219',
                        'Elbenwald_FL_10.wav' : '210',
                        'Polarnacht_Focus_FR_1.wav' : '121',
                        'Polarnacht_Focus_FL_2.wav' : '122',
                        'Polarnacht_Focus_FR_3.wav' : '123',
                        'Polarnacht_Focus_FL_4.wav' : '124',
                        'Polarnacht_FR_5.wav' : '225',
                        'Polarnacht_FL_6.wav' : '226',
                        'Polarnacht_FR_7.wav' : '227',
                        'Polarnacht_FL_8.wav' : '228',
                        'Polarnacht_FR_9.wav' : '229',
                        'Polarnacht_FL_10.wav' : '220'}

    stimuli_list = [{'path': stimuli_path, 'trigger_code': stimuli_trigger_codes[stimuli_path.split('/')[-1]], 'side': side} for (stimuli_path, side) in zip(stimuli_paths, ['r','l'] * 10)]

    return stimuli_list

def get_eeg_header_file(base_dir, subject):
    """Return absolute path of eeg header file for given base directory and subject number
    Args:
        base_dir (str): base dir of repository
        subject (int): subject number e.g. 129
    Returns:
        string: path to eeg header file of given subject number
    """
    subject_folder = join(base_dir,"data","raw_input", str(subject))
    header_file = ''
    for file in os.listdir(subject_folder):
        if file.endswith('.vhdr'):
            header_file = file
    assert header_file.endswith('.vhdr'), "no file with matching ending found!"
    return join(subject_folder, header_file)

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

def get_listening_effort(df_csv_log):
    """extract listening effort from pandas data frame of the psychopy .csv log file
    order of output depends on randomisation
    
    Args:
        df_csv_log (str): path to log file

    Returns:
        np.array: listening efforts during ci attention experiment. [single speaker, single speaker, competing speaker, competing speaker]
    """
    rows_listening_effort = np.where(np.invert(pd.isna(df_csv_log.loc[:,'button_13.numClicks'])))[0]
    list_of_buttons = ['button_1.numClicks', 'button_02.numClicks', 'button_3.numClicks', 'button_4.numClicks', 'button_5.numClicks','button_6.numClicks', 'button_7.numClicks', 'button_8.numClicks', 'button_9.numClicks', 'button_10.numClicks', 'button_11.numClicks', 'button_12.numClicks', 'button_13.numClicks']
    listening_efforts = np.where((df_csv_log.loc[rows_listening_effort,list_of_buttons] > 0).to_numpy())[1] +1 
    return listening_efforts

def get_comparisons(df_csv_log):
    """extracts comparison of preference for the content of the stories, the understandbility of of the speaker and the story

    Args:
        df_csv_log (str): path to csv log file

    Returns:
        list: comparisons in order of experiment
    """
    #extracts comparison of preference for the content of the stories, the understandbility of of the speaker and the story
    columns = ['key_resp_Sprecher_Vergleich_3.keys', 'key_resp_Sprecher_Vergleich.keys', 'key_resp_Geschichten_Vergleich.keys']
    vals = []
    for column in columns:
        #check for key exists (in the first experiment 'key_resp_Sprecher_Vergleich_3.keys' was not implemented)
        if column in df_csv_log:
            row = np.where(np.invert(pd.isna(df_csv_log.loc[:,column])))[0]
            val = df_csv_log.loc[row, column].iloc[0]
            vals.append(val)
        else:
             vals.append(float("NAN"))
    return vals

def get_volume(df_csv_log):
    """extracts volume setting in psychopy during experiment

    Args:
        df_csv_log (str): path to csv log file

    Returns:
        float: volume setting, 1.0 is default setting
    """
    #extracts volume during experiment
    column = 'Volume'
    if column in df_csv_log:
        row = np.where(np.invert(pd.isna(df_csv_log.loc[:,column])))[0]
        val = df_csv_log.loc[row, column].iloc[0]
    else:
        val = float("NAN")
    return val

def get_taken_out_electrodes(eeg_raw_brainvision):
    """
    Determines taken out electrodes during a measurement based on impedances.
    Nan impedances are taken is hint for taken out electrode.
    Return names of channels, the channel number and the indices (channel_number - 1)

    Args:
        eeg_raw_brainvision (_type_): brainvision object of EEG-measurment

    Returns:
        _type_: _description_
    """
    impedances_dict = eeg_raw_brainvision.__dict__['impedances']
    channels = list(impedances_dict.keys())
    taken_out_channels = []
    taken_out_channel_numbers = []
    for channnel, channel_number in zip(channels,range(1, len(channels) + 1)):
        if math.isnan(impedances_dict[channnel]['imp']):
            taken_out_channels.append(channnel)
            taken_out_channel_numbers.append(channel_number)
    taken_out_channel_indices = np.array(taken_out_channel_numbers) - 1

    assert len(taken_out_channels) < 5, "More than 4 taken out electrodes identified. Probabaly corrupted header file."

    return taken_out_channels, taken_out_channel_numbers, taken_out_channel_indices

def get_ten_second_windows(stimuli_list):
    """_summary_

    Args:
        stimuli_list (list):

    Returns:
        list: number of ten second windows that fit in the stimulu provided in the input list
    """
    ten_second_windows = []
    for stimulus in stimuli_list:
        _, stim_wav_0 = read_wav(stimulus['path'])
        stim_wav_0_l, stim_wav_0_r = downsample_wav(stim_wav_0[:,0], stim_wav_0[:,1])

        if stimulus['side'] == 'r':
            stim_wav = stim_wav_0_r
        elif stimulus['side'] == 'l':
            stim_wav = stim_wav_0_l
        ten_second_windows.append(int(((len(stim_wav) / 1000) - ((len(stim_wav) / 1000) % 10)) / 10))
    return ten_second_windows

def calc_drift_analysis(corr_indices, trigger_indices, stimuli_list, stim_eeg_data_l_study, stim_eeg_data_r_study):
    """Performs analysis of drift within EEG measurement

    Args:
        corr_indices (list):        alignment indices according to cross-correlation
        trigger_indices (list):     alignment indices according to trigger
        stimuli_list (list):        list of audio stimuli
        stim_eeg_data_l_study ():   left channel stimulus data recorded through SimTrak and EEG amp
        stim_eeg_data_r_study ():   right channel stimulus data recorded through SimTrak and EEG amp

    Returns:
        (drifts_subject_corr:   np.array, drifts_subject_trigger:np.array): calculated drifts on ten second windows relative to the global alignment,
                                a row represents one trial
    """
    ten_second_windows = get_ten_second_windows(stimuli_list)
    drifts_subject_corr = np.empty((20,max(ten_second_windows)))
    drifts_subject_corr[:] = np.nan

    drifts_subject_trigger = drifts_subject_corr.copy()

    for corr_index, trigger_index, stimulus, i in zip(corr_indices, trigger_indices, stimuli_list, range(0,20)):
        _, stim_wav_0 = read_wav(stimulus['path'])
        stim_wav_0_l, stim_wav_0_r = downsample_wav(stim_wav_0[:,0], stim_wav_0[:,1])

        if stimulus['side'] == 'r':
            stim_wav = stim_wav_0_r
            stim_eeg = stim_eeg_data_r_study[0,:]
        elif stimulus['side'] == 'l':
            stim_wav = stim_wav_0_l
            stim_eeg = stim_eeg_data_l_study[0,:]

        stim_eeg_snippet_corr = stim_eeg[corr_index: corr_index + len(stim_wav)]
        stim_eeg_snippet_trigger = stim_eeg[trigger_index: trigger_index + len(stim_wav)]

        for j in range(0,ten_second_windows[i]):
            start_second, end_second = j * 10, j * 10 + 10

            #correlation
            corr = correlate(stim_eeg_snippet_corr[start_second * 1000 : end_second * 1000], stim_wav[start_second * 1000 : end_second * 1000], mode = 'full')
            lags = correlation_lags(stim_eeg_snippet_corr[start_second * 1000: end_second * 1000].size, stim_wav[start_second * 1000: end_second * 1000].size, mode='full')
            lag_index = int(lags[np.argmax(np.abs(corr))])
            drifts_subject_corr[i,j] = lag_index

            #trigger
            corr = correlate(stim_eeg_snippet_trigger[start_second * 1000 : end_second * 1000], stim_wav[start_second * 1000 : end_second * 1000], mode = 'full')
            lags = correlation_lags(stim_eeg_snippet_trigger[start_second * 1000: end_second * 1000].size, stim_wav[start_second * 1000: end_second * 1000].size, mode='full')
            lag_index = int(lags[np.argmax(np.abs(corr))])
            drifts_subject_trigger[i,j] = lag_index
            
    return drifts_subject_corr, drifts_subject_trigger

def plot_drift_analysis(drifts_subject_corr, drifts_subject_trigger, subject, base_dir):
    """
    Plots the results of analysing the drifts within an EEG measurement

    Args:
        drifts_subject_corr (np.array):     drifts on ten second windows based on correlation analysis
        drifts_subject_trigger (np.array):  drifts on ten second windows based on triggers
        subject (int):                      subject identifier
        base_dir (string):                  directory where git repository is located
    """

    drifts_mean_corr, drifts_variance_corr = np.mean(drifts_subject_corr, axis=0, where= np.isfinite(drifts_subject_corr)), np.var(drifts_subject_corr, axis = 0, where = np.isfinite(drifts_subject_corr))
    drifts_mean_trigger, drifts_variance_trigger = np.mean(drifts_subject_trigger, axis=0, where= np.isfinite(drifts_subject_trigger)), np.var(drifts_subject_trigger, axis = 0, where = np.isfinite(drifts_subject_trigger))
    x = np.linspace(0, len(drifts_mean_corr) - 1, len(drifts_mean_corr))
    fig, ax = plt.subplots(2,2,figsize = (14,8))
    fig.tight_layout(pad=3.0)
    ax[0,0].plot(x, drifts_mean_corr, label = 'mean', linewidth = 3)
    ax[0,0].fill_between(x, drifts_mean_corr - drifts_variance_corr, drifts_mean_corr + drifts_variance_corr, alpha = .2, label = 'variance')
    ax[0,0].set_ylabel('ms')
    ax[0,0].set_xticks(x)
    ax[0,0].grid()
    ax[0,0].legend()
    ax[0,0].set_title(f'Drift Subject {str(subject)} mean over all trials (StimTrak)')

    ax[0,1].plot(x, drifts_mean_trigger, label = 'mean', linewidth = 3)
    ax[0,1].fill_between(x, drifts_mean_trigger - drifts_variance_trigger, drifts_mean_trigger + drifts_variance_trigger, alpha = .2, label = 'variance')
    ax[0,1].set_ylabel('ms')
    ax[0,1].set_xticks(x)
    ax[0,1].grid()
    ax[0,1].legend()
    ax[0,1].set_title(f'Drift Subject {str(subject)} mean over all trials (Trigger)')

    colors = cm.rainbow(np.linspace(0, 1, 20))

    trial_corr = drifts_subject_corr[0,:]
    trial_corr = trial_corr[np.isfinite(trial_corr)]
    x_reg = np.linspace(0, len(trial_corr) - 1, len(trial_corr))
    res = linregress(x_reg, trial_corr)
    ax[1,0].plot(x_reg, res.intercept + res.slope*x_reg, linewidth = 1.5, color = colors[0], label = 'linear regression')
    ax[1,0].plot(x_reg, trial_corr, 'o', color = colors[0], markersize = 3, label = 'data points')

    for trial_corr, trial_trigger, c, trial in zip(drifts_subject_corr, drifts_subject_trigger, colors, range(1,21)):
        #StimTrak
        trial_corr = trial_corr[np.isfinite(trial_corr)]
        x_reg = np.linspace(0, len(trial_corr) - 1, len(trial_corr))
        res = linregress(x_reg, trial_corr)
        ax[1,0].plot(x_reg, res.intercept + res.slope*x_reg, linewidth = 1.5, color = c)
        ax[1,0].plot(x_reg, trial_corr, 'o', color = c, markersize = 3)
        #Trigger
        trial_trigger = trial_trigger[np.isfinite(trial_trigger)]
        x_reg = np.linspace(0, len(trial_corr) - 1, len(trial_corr))
        res = linregress(x_reg, trial_trigger)
        ax[1,1].plot(x_reg, res.intercept + res.slope*x_reg, linewidth = 1.5, color = c, label = str(trial))
        ax[1,1].plot(x_reg, trial_trigger, 'o', color = c, markersize = 3)

    ax[1,0].set_ylabel('ms')
    ax[1,0].set_xticks(x)
    ax[1,0].set_xlabel('ten second window')
    ax[1,0].grid()
    ax[1,0].set_title(f'Drift Subject {str(subject)} individual trials (StimTrak)')
    ax[1,0].legend()

    ax[1,1].set_ylabel('ms')
    ax[1,1].set_xticks(x)
    ax[1,1].set_xlabel('ten second window')
    ax[1,1].grid()
    ax[1,1].set_title(f'Drift Subject {str(subject)} individual trials (Trigger)')
    ax[1,1].legend(loc='upper center', bbox_to_anchor=(0.0, -0.13),
          fancybox=True, shadow=True, ncol=10, title = "Trial")

    plt.savefig(join(base_dir, "reports", "figures", "drift" ,str(subject) + ".pdf"), bbox_inches='tight')

def estimate_envelope(stim_wav_l, stim_wav_r, wav_freq, output_freq = 125):
    """
    Estimates the speech envelope of given stimulus.
    And resamples it to 1kHz.
    Args:
        stim_wav_l (np.array): left channel stimulus
        stim_wav_r (np.array): right channel stimulus
        wav_freq (_type_): sampling frequency of audio
        output_freq(int): sampling frequency of output envelope

    Returns:
        (np.array, np.array): (envelope_left, envelope_right)
    """
    #envelope computation
    env_l, env_r = np.abs(hilbert(stim_wav_l)), np.abs(hilbert(stim_wav_r))
    env_l, env_r = mne.filter.filter_data(env_l, wav_freq, None, 50, verbose = False), mne.filter.filter_data(env_r, wav_freq, None, 50, verbose= False)
    
    #Resample and normalize
    env_l, env_r = mne.filter.resample(env_l, output_freq, wav_freq, verbose = False), mne.filter.resample(env_r, output_freq, wav_freq, verbose = False)
    env_l, env_r = zscore(env_l, nan_policy='raise'), zscore(env_r, nan_policy = 'raise')

    return env_l, env_r

def write_taken_out_channels_txt(taken_out_channels):
    """Writes taken out channels to a .txt file and returns its path.
    The text file can be used as an input to mne.raw.load_bad_channels
    
        Args:
        taken_out_channels (list): List of taken out channels e.g. ['CP5', 'CP6']

    Returns:
        string: _description_
    """
    f = open("bad_channels_tmp.txt", "w")
    for channel in taken_out_channels:
        f.write(channel + "\n")
    path = f.name
    f.close()
    return f.name

####
#these functions are used for post-processing of raw reconstruction results
####

def moveaxis(ridge_list_raw):
    """bring ridge data into shape (n_subjects, n_trials, n_windows)
    Args:
        ridge_list_raw (list): in order of (n_windows, n_subjects, n_trials)

    Returns:
        list: in order of (n_subjects, n_trials, n_windows)
    """
    
    n_windows = len(ridge_list_raw)
    n_subjects = len(ridge_list_raw[0])
    n_trials = len(ridge_list_raw[0][0])

    ridge_attended_reordered = []
    for subj in range(n_subjects):
        subj_list = []
        for trial in range(n_trials):
            trial_list = []
            for win in range(n_windows):
                trial_list.append(ridge_list_raw[win][subj][trial])
            subj_list.append(trial_list)
        ridge_attended_reordered.append(subj_list)
    return ridge_attended_reordered

def correct_trials_list(raw_list, randomisations):
    """
    Takes in the list of raw cnn scores.
    The problem ist that the order of trials is different for odd and even subjects du to randomisation.

    Args:
        raw_list (list): output of raw scroes from evaluation
        randomisations (list): list of randomisations for each subject

    Returns:
        list: list of reconstruction scores with consistent trial order for all subjects
    """
    assert isinstance(raw_list, list), "raw_list must be a list"
    assert len(raw_list) == len(randomisations), "number of randomisations must match number of subjects"

    correct_trials = []
    for subject_index, rand in zip(range(0, len(raw_list)), randomisations):
        if rand == 0:
            correct_trials.append(raw_list[subject_index])
        #reorder trials for odd subjects
        elif rand == 1:
            corrected_tmp = []
            for trial in range(0, len(raw_list[subject_index])):
                if trial % 4 < 2:
                    corrected_tmp.append(raw_list[subject_index][trial + 2])
                else:
                    corrected_tmp.append(raw_list[subject_index][trial - 2])
            correct_trials.append(corrected_tmp)
    return correct_trials


def correct_trials(input_array, randomisations):
    """
    The problem ist that the order of trials is different for odd and even subjects du to randomisation.
    Outputs an array with the same order of trials for all subjects.

    Args:
        input array (np.array): for uneven subjects, the order of trials is changed
        randomisations (list): list of randomisations for each subject

    Returns:
        np.array: array with same order of trials for all subjects
    """
    #due to randomisation of trials, the order of trials is not the same for all subjects (it's alternating)
    #second dimension in test must be divisible by 4
    assert input_array.shape[1] % 4 == 0, "second dimension in test must be divisible by 4"
    #chck if randomisations is a list
    assert isinstance(randomisations, list), "randomisations must be a list"
    assert len(randomisations) == input_array.shape[0], "number of randomisations must match number of subjects"

    #swops for odd subjects
    test_correct = np.zeros(input_array.shape)
    for subject, rand in zip(range(0, input_array.shape[0]), randomisations):
        #even subjects have randomisation 0 and don't need to be changed
        if rand == 0:
            test_correct[subject,:,:] = input_array[subject,:,:]
        #odd subjects have randomisation 1 and need to be changed
        else:
            for trial in range(0,input_array.shape[1]):
                if trial % 4 < 2:
                    test_correct[subject,trial,:] = input_array[subject,trial+2,:]
                else:
                    test_correct[subject,trial,:] = input_array[subject,trial-2,:]
    return test_correct

def flatten_subject_first_over_two_lists(input_list_0, input_list_1, window_index):
    """
    Flattens data from two lists for each subject and trial for specified window index
    Allows to construct data matrix such that cross-validation can be performed on subject level

    Args:
        input_list_0 (list): 
        input_list_1 (list):
        window_index (int): index of window to be flattened from [60,45,30,20,10,5,2,1]

    Returns:
        np.array: subject-first flattened data
    """
    assert len(input_list_0) == len(input_list_1), f"lists must have same length but have {len(input_list_0)} and {len(input_list_1)}"
    flattended_data = []
    for subject in range(0,len(input_list_0)):
        for input_list in [input_list_0, input_list_1]:
            for trial in range(0,len(input_list[0])):
                if isinstance(input_list[subject][trial][window_index], np.ndarray):
                    #assert input_list_0[subject][trial][window_index].shape == input_list_1[subject][trial][window_index].shape, f"arrays must have same shape but have {input_list_0[subject][trial][window_index].shape} and {input_list_1[subject][trial][window_index].shape}"
                    for val in input_list[subject][trial][window_index].tolist():
                        flattended_data.append(val)
                else:
                    #assert len(input_list_0[subject][trial][window_index]) == len(input_list_1[subject][trial][window_index]), f"arrays must have same shape but have {len(input_list_0[subject][trial][window_index])} and {len(input_list_1[subject][trial][window_index])}"
                    for val in input_list[subject][trial][window_index]:
                        flattended_data.append(val)
    return np.array(flattended_data)

def flatten_subject_raw_data(input_list, window_index):
    """
    Flattens data for each subject and trial for specified window index
    Note list should have trials in standardized order of ranozimaiton 0.

    Args:
        input_list (list): list of reconstruction scores in shape (subjects, trials, window_sizes, reconstruction_scores)
        window_index (int): index of window to be flattened from [60,45,30,20,10,5,2,1]

    Returns:
        np.array: flattened data
    """
    flattended_data = []
    for subject in range(0,len(input_list)):
        for trial in range(0,len(input_list[0])):
            if isinstance(input_list[subject][trial][window_index], np.ndarray):
                for val in input_list[subject][trial][window_index].tolist():
                    flattended_data.append(val)
            else:
                for val in input_list[subject][trial][window_index]:
                    flattended_data.append(val)
    return np.array(flattended_data)

def get_elb_pol_attended(input_list):
    """
    Note input list should have trials in standardized order of ranozimaiton 0.
    Takes raw reconstruction scores and return one list containing 
    trials where elbenwald was attended another where Polarnacht is attended
    Both of shape (subjects, trials, window_sizes, reconstruction_scores)

    Args:
        input_list (list): list of reconstruction scores in shape (subjects, trials, window_sizes, reconstruction_scores)

    Returns:
        tuple: (elbenwald_data, polarnacht_data)
    """
    elb = []
    pol = []
    for subj in range(0, len(input_list)):
        subj_elb_tmp = []
        subj_pol_tmp = []
        for trial in range(0, len(input_list[subj])):
            if trial % 4 < 2:
                subj_elb_tmp.append(input_list[subj][trial][:])
            elif trial % 4 >= 2:
                subj_pol_tmp.append(input_list[subj][trial][:])
        elb.append(subj_elb_tmp)
        pol.append(subj_pol_tmp)
    return elb, pol

def get_randomisation(subject, base_dir):
    """
    Get randomisation from psychopy log file

    Args:
        subject (int): subject number
        base_dir (string): base directory

    Returns:
        int: randomisation: 0 or 1
    """
    csv_log_file = get_csv_log_file(base_dir, subject)
    psychopy_log_file = read_csv(csv_log_file, sep=',')

    if 'randomisation' in psychopy_log_file.columns.tolist():
        randomisation = psychopy_log_file.loc[0,'randomisation']
    else:
        randomisation = int(input('Specifiy randomisation (0 or 1), because it is missing in log file'))
    return randomisation

def correct_dataframe(df):
    """
    Some answeres were wrong in the psychopy experiment.
    To ensure correct analysis the answers are corrected with this function.

    Args:
        df (pandas dataframe): pandas datafram containing the experiment data
    """
    df.loc[df['trigger'] == 217, 'answer_1'] = 'c'
    if df.loc[df['trigger'] == 217, 'subject_answer_1'].item() == 'c':
        df.loc[df['trigger'] == 217, 'answer_1_score'] = 1
    else:
        df.loc[df['trigger'] == 217, 'answer_1_score'] = 0

    df.loc[df['trigger'] == 218, 'answer_1'] = 'a'
    if df.loc[df['trigger'] == 218, 'subject_answer_1'].item() == 'a':
        df.loc[df['trigger'] == 218, 'answer_1_score'] = 1
    else:
        df.loc[df['trigger'] == 218, 'answer_1_score'] = 0

    df.loc[df['trigger'] == 219, 'answer_1'] = 'c'
    if df.loc[df['trigger'] == 219, 'subject_answer_1'].item() == 'c':
        df.loc[df['trigger'] == 219, 'answer_1_score'] = 1
    else:
        df.loc[df['trigger'] == 219, 'answer_1_score'] = 0

    df.loc[df['trigger'] == 219, 'answer_2'] = 'a'
    if df.loc[df['trigger'] == 219, 'subject_answer_2'].item() == 'a':
        df.loc[df['trigger'] == 219, 'answer_2_score'] = 1
    else:
        df.loc[df['trigger'] == 219, 'answer_2_score'] = 0
    
    return df