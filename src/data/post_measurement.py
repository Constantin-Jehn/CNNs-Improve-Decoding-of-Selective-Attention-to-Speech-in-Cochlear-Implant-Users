#Code to run after a measurement to analyse alignment and to extract the experiment info
#Raw data must lay in data/raw_input/subject

from eeg_measurement import EegMeasurement
from eeg_attention_pytorch_dataset import EegAttentionDataset
import numpy as np
import git
import os
import json
from tqdm import tqdm



def create_dataset_22_09():

    #Specify newly measured subjects
    subjects = list(range(101,115))
    aux_channels_correct = np.ones(14,dtype=int).tolist()
    #subject 103 has switched aux channels
    aux_channels_correct[2] = 0
    l_freq = 1.0
    h_freq = 16.0
    output_freq = 125
    data_filename = 'ci_l_1_h_16_out_125.hdf5'
    base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    for subject, aux_correct in zip(subjects, aux_channels_correct):
        eeg_measurement = EegMeasurement(subject, base_dir, aux_correct)

        #write stimulus data only once
        if subject == 101:
            eeg_measurement.write_stimulus_data(data_filename = data_filename, l_freq_env=l_freq, h_freq_env = h_freq, output_freq = output_freq)
        
        if subject > 111:
            #for new data only
            eeg_measurement.extract_experiment_info()
            eeg_measurement.analyse_alignment()
            eeg_measurement.analyse_drift()

        eeg_measurement.write_subjects_eeg_data(data_filename = data_filename, l_freq_eeg = l_freq, h_freq_eeg = h_freq, output_freq = output_freq)

def create_dataset_28_09():

    #Specify newly measured subjects
    subjects = list(range(101,115))
    aux_channels_correct = np.ones(14, dtype=int).tolist()
    #subject 103 has switched aux channels
    aux_channels_correct[2] = 0

    l_freq_env = 1.0
    h_freq_env = 50.0

    l_freq_eeg = 1.0
    h_freq_eeg = 32

    output_freq = 125
    data_filename = 'ci_l_1,1_h_32,50_out_125_114.hdf5'
    base_dir = base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    for subject, aux_correct in zip(subjects, aux_channels_correct):
        eeg_measurement = EegMeasurement(subject, base_dir, aux_correct)

        #write stimulus data only once
        if subject == 101:
            eeg_measurement.write_stimulus_data(data_filename = data_filename, l_freq_env=l_freq_env, h_freq_env = h_freq_env, output_freq = output_freq)

        eeg_measurement.write_subjects_eeg_data(data_filename = data_filename, l_freq_eeg = l_freq_eeg, h_freq_eeg = h_freq_eeg, output_freq = output_freq)

def create_dataset_11_10():
    #create dataset with ica cleaned data

    #Specify newly measured subjects
    subjects = list(range(101,115))
    subjects.append(116)
    aux_channels_correct = np.ones(15, dtype=int).tolist()
    #subject 103 has switched aux channels
    aux_channels_correct[2] = 0

    l_freq_env = 1.0
    h_freq_env = 50.0

    l_freq_eeg = 1.0
    h_freq_eeg = 32

    output_freq = 125
    data_filename = 'ci_l_1,1_h_32,50_out_125_116_incl_ica.hdf5'
    base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    for subject, aux_correct in zip(subjects, aux_channels_correct):
        eeg_measurement = EegMeasurement(subject, base_dir, aux_correct)

        #write stimulus data only once
        if subject == 101:
            eeg_measurement.write_stimulus_data(data_filename = data_filename, l_freq_env=l_freq_env, h_freq_env = h_freq_env, output_freq = output_freq)

        eeg_measurement.write_subjects_eeg_data(data_filename = data_filename, l_freq_eeg = l_freq_eeg, h_freq_eeg = h_freq_eeg, output_freq = output_freq)

def create_dataset_08_01():
    #create dataset with ica cleaned data

    #Specify newly measured subjects

    #leave out 115 and 117
    # subjects = list(range(101,115))
    # subjects = subjects + list(range(116,117))
    # subjects = subjects + list(range(118,126))

    #for debugging only
    subjects = [124, 125]

    aux_channels_correct = np.ones(len(subjects), dtype=int).tolist()
    #subject 103 has switched aux channels
    #aux_channels_correct[2] = 0

    l_freq_env = 1.0
    h_freq_env = 50.0

    l_freq_eeg = 1.0
    h_freq_eeg = 32

    output_freq = 125
    data_filename = 'ci_l_1,1_h_32,50_out_125_125.hdf5'
    base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    for subject, aux_correct in zip(subjects, aux_channels_correct):
        eeg_measurement = EegMeasurement(subject, base_dir, aux_correct)

        #write stimulus data only once
        if subject == 101:
            eeg_measurement.write_stimulus_data(data_filename = data_filename, l_freq_env=l_freq_env, h_freq_env = h_freq_env, output_freq = output_freq)
        
        if subject > 116:
            #for new data only
            eeg_measurement.extract_experiment_info()
            eeg_measurement.analyse_alignment()
            eeg_measurement.analyse_drift()
        eeg_measurement.write_subjects_eeg_data(data_filename = data_filename, l_freq_eeg = l_freq_eeg, h_freq_eeg = h_freq_eeg, output_freq = output_freq, use_ica=False)
        #print('Subject {} done'.format(subject))

def create_dataset_09_01():
    #create dataset without rerereferencing
    #create dataset with ica cleaned data

    #Specify newly measured subjects

    #leave out 115 and 117
    subjects = list(range(101,115))
    subjects = subjects + list(range(116,117))
    subjects = subjects + list(range(118,126))

    aux_channels_correct = np.ones(len(subjects), dtype=int).tolist()
    #subject 103 has switched aux channels
    aux_channels_correct[2] = 0

    l_freq_env = 1.0
    h_freq_env = 50.0

    l_freq_eeg = 1.0
    h_freq_eeg = 32

    output_freq = 125
    data_filename = 'ci_l_1,1_h_32,50_out_125_125_no_reref.hdf5'
    base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    for subject, aux_correct in zip(subjects, aux_channels_correct):
        eeg_measurement = EegMeasurement(subject, base_dir, aux_correct, eeg_reference=None)

        #write stimulus data only once
        if subject == 101:
            eeg_measurement.write_stimulus_data(data_filename = data_filename, l_freq_env=l_freq_env, h_freq_env = h_freq_env, output_freq = output_freq)
        
        if subject > 116:
            #for new data only
            eeg_measurement.extract_experiment_info()
            eeg_measurement.analyse_alignment()
            eeg_measurement.analyse_drift()
        eeg_measurement.write_subjects_eeg_data(data_filename = data_filename, l_freq_eeg = l_freq_eeg, h_freq_eeg = h_freq_eeg, output_freq = output_freq, use_ica=False)

def create_dataset_10_01():
    #create dataset with ica cleaned data, now ica is calculated on re-referenced data, which led to cleaner TRFs

    #Specify newly measured subjects

    #leave out 115 and 117
    subjects = list(range(101,115))
    subjects = subjects + list(range(116,117))
    subjects = subjects + list(range(118,126))


    aux_channels_correct = np.ones(len(subjects), dtype=int).tolist()
    #subject 103 has switched aux channels
    aux_channels_correct[2] = 0

    l_freq_env = 1.0
    h_freq_env = 50.0

    l_freq_eeg = 1.0
    h_freq_eeg = 32

    output_freq = 125
    data_filename = 'ci_l_1,1_h_32,50_out_125_125_incl_ica.hdf5'
    base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    for subject, aux_correct in zip(subjects, aux_channels_correct):
        eeg_measurement = EegMeasurement(subject, base_dir, aux_correct, eeg_reference='avg')

        #write stimulus data only once
        if subject == 101:
            eeg_measurement.write_stimulus_data(data_filename = data_filename, l_freq_env=l_freq_env, h_freq_env = h_freq_env, output_freq = output_freq)
        eeg_measurement.write_subjects_eeg_data(data_filename = data_filename, l_freq_eeg = l_freq_eeg, h_freq_eeg = h_freq_eeg, output_freq = output_freq, use_ica=True)

def debug_stim_writing():
    #create dataset with ica cleaned data

    #Specify newly measured subjects
    subjects = [116,120]
    aux_channels_correct = np.ones(len(subjects), dtype=int).tolist()

    l_freq_env = 1.0
    h_freq_env = 50.0

    l_freq_eeg = 1.0
    h_freq_eeg = 32

    output_freq = 125
    data_filename = 'test_stim.hdf5'
    base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    for subject, aux_correct in zip(subjects, aux_channels_correct):
        eeg_measurement = EegMeasurement(subject, base_dir, aux_correct)

        #write stimulus data only once
        #if subject == 120:
        eeg_measurement.write_stimulus_data(data_filename = data_filename, l_freq_env=l_freq_env, h_freq_env = h_freq_env, output_freq = output_freq)
        
        if subject > 116:
            #for new data only
            eeg_measurement.extract_experiment_info()
            eeg_measurement.analyse_alignment()
            eeg_measurement.analyse_drift()
        eeg_measurement.write_subjects_eeg_data(data_filename = data_filename, l_freq_eeg = l_freq_eeg, h_freq_eeg = h_freq_eeg, output_freq = output_freq, use_ica=False)
        print('Subject {} done'.format(subject))

def check_data_set():
    # subjects = list(range(116,117))
    # subjects = subjects + list(range(118,123))
    subjects = [116,120]
    subjects = [str(s) for s in subjects]
    base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    dir_h5 = os.path.join(base_dir, 'data', 'processed', 'ci_l_1,1_h_32,50_out_125_125_incl_ica.hdf5')
    trials = list(range(0,20))
    torch_datasets = []
    for sub in subjects:
        torch_datasets.append(EegAttentionDataset(dir_h5, sub, trials, window_size_training=60))
    pass

def debug_subject_summaries():
    #leave out 115 and 117
    subjects = [127, 128, 130]

    aux_channels_correct = np.ones(len(subjects), dtype=int).tolist()

    base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    for subject, aux_correct in zip(subjects, aux_channels_correct):
        eeg_measurement = EegMeasurement(subject, base_dir, aux_correct)
        eeg_measurement.extract_experiment_info()
        print(f'subject {subject} processed')

def create_intermediate_database():
    #create dataset with ica cleaned data, now ica is calculated on re-referenced data, which led to cleaner TRFs

    #Specify newly measured subjects

    #leave out 115 and 117
    subjects = [127, 128, 130]


    aux_channels_correct = np.ones(len(subjects), dtype=int).tolist()

    l_freq_env = 1.0
    h_freq_env = 50.0

    l_freq_eeg = 1.0
    h_freq_eeg = 32

    output_freq = 125
    data_filename = 'intermediate_127_128_130_noreref.hdf5'
    base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    for subject, aux_correct in zip(subjects, aux_channels_correct):
        eeg_measurement = EegMeasurement(subject, base_dir, aux_correct, eeg_reference=None)

        #write stimulus data only once
        if subject == 127:
            eeg_measurement.write_stimulus_data(data_filename = data_filename, l_freq_env=l_freq_env, h_freq_env = h_freq_env, output_freq = output_freq)
        eeg_measurement.write_subjects_eeg_data(data_filename = data_filename, l_freq_eeg = l_freq_eeg, h_freq_eeg = h_freq_eeg, output_freq = output_freq, use_ica=False)

def create_dataset_14_03():
    #create dataset with ica cleaned data, now ica is calculated on re-referenced data, which led to cleaner TRFs
    #This is the dataset for final evaluation used in the paper

    #Specify newly measured subjects

    #leave out 101, 115 and 117, 126, 129
    subjects = list(range(102,115))
    subjects = subjects + list(range(116,117))
    subjects = subjects + list(range(118,126))
    subjects = subjects + [127, 128, 130]


    aux_channels_correct = np.ones(len(subjects), dtype=int).tolist()
    #subject 103 has switched aux channels
    aux_channels_correct[1] = 0

    l_freq_env = 1.0
    h_freq_env = 50.0

    l_freq_eeg = 1.0
    h_freq_eeg = 32

    output_freq = 125
    #data_filename = 'ci_attention_final_l_1,1_h_32,50_out_125_130_incl_ica.hdf5'
    data_filename='test'
    base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    for subject, aux_correct in zip(subjects, aux_channels_correct):
        eeg_measurement = EegMeasurement(subject, base_dir, aux_correct, eeg_reference='avg')

        #write stimulus data only once
        if subject == 102:
            eeg_measurement.write_stimulus_data(data_filename = data_filename, l_freq_env=l_freq_env, h_freq_env = h_freq_env, output_freq = output_freq)
        eeg_measurement.write_subjects_eeg_data(data_filename = data_filename, l_freq_eeg = l_freq_eeg, h_freq_eeg = h_freq_eeg, output_freq = output_freq, use_ica=True)

def create_BASEN_dataset():
    #create dataset with comparable filterings to the BASEN dataset

    #Specify newly measured subjects

    #leave out 101, 115 and 117, 126, 129
    subjects = list(range(102,115))
    subjects = subjects + list(range(116,117))
    subjects = subjects + list(range(118,126))
    subjects = subjects + [127, 128, 130]


    aux_channels_correct = np.ones(len(subjects), dtype=int).tolist()
    #subject 103 has switched aux channels
    aux_channels_correct[1] = 0

    l_freq_env = 1.0
    h_freq_env = 50.0

    l_freq_eeg = 0.1
    h_freq_eeg = 45

    output_freq = 128
    #data_filename = 'ci_attention_final_l_1,1_h_32,50_out_125_130_incl_ica.hdf5'
    data_filename='CI_BASEN'
    base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    for subject, aux_correct in zip(subjects, aux_channels_correct):
        eeg_measurement = EegMeasurement(subject, base_dir, aux_correct, eeg_reference='avg')

        #write stimulus data only once
        if subject == 102:
            eeg_measurement.write_stimulus_data(data_filename = data_filename, l_freq_env=l_freq_env, h_freq_env = h_freq_env, output_freq = output_freq)
        eeg_measurement.write_subjects_eeg_data(data_filename = data_filename, l_freq_eeg = l_freq_eeg, h_freq_eeg = h_freq_eeg, output_freq = output_freq, use_ica=True)

def create_control_dataset():
    subjects = list(range(301,329))
    #308 seems to need aux channels inverted
    aux_channels_correct = np.ones(len(subjects),dtype=int).tolist()
    if 308 in subjects:
        aux_channels_correct[subjects.index(308)] = 0
    
    l_freq_env = 1.0
    h_freq_env = 50.0

    # in accordance to BASEN paper
    l_freq_eeg = 1.0
    h_freq_eeg = 45

    output_freq = 128

    data_filename = 'control_group_301_328.hdf5'
    base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    alignment_doc = {}
    for subject, aux_correct in zip(subjects, aux_channels_correct):
        eeg_measurement = EegMeasurement(subject, base_dir, aux_correct, eeg_reference='avg', cohort='control')
        alignment_doc[str(subject)] = eeg_measurement.using_trigger
        #eeg_measurement.analyse_alignment()
        #write stimulus data only once
        if subject == subjects[0]:
            eeg_measurement.write_stimulus_data(data_filename = data_filename, l_freq_env=l_freq_env, h_freq_env = h_freq_env, output_freq = output_freq)
        eeg_measurement.write_subjects_eeg_data(data_filename = data_filename, l_freq_eeg = l_freq_eeg, h_freq_eeg = h_freq_eeg, output_freq = output_freq, use_ica=True)
    with open(os.path.join(base_dir,'data', 'processed', data_filename[:-5]+'.json'), 'w') as f:
        json.dump(alignment_doc, f)

def create_ci_rejection_dataset():
    # create dataset with 1kHz EEG to test artifact rejection at higher frequencie

    #Specify newly measured subjects

    #leave out 101, 115 and 117, 126, 129
    subjects = list(range(102,115))
    subjects = subjects + list(range(116,117))
    subjects = subjects + list(range(118,126))
    subjects = subjects + [127, 128, 130]


    aux_channels_correct = np.ones(len(subjects), dtype=int).tolist()
    #subject 103 has switched aux channels
    aux_channels_correct[1] = 0

    l_freq_env = 1.0
    h_freq_env = 50.0

    l_freq_eeg = 1.0
    h_freq_eeg = 280

    output_freq = 1000
    data_filename = 'ci_artifact_dataset_1kHz.hdf5'
    base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    for subject, aux_correct in zip(subjects, aux_channels_correct):
        eeg_measurement = EegMeasurement(subject, base_dir, aux_correct, eeg_reference='avg')

        #write stimulus data only once
        if subject == 102:
            eeg_measurement.write_stimulus_data(data_filename = data_filename, l_freq_env=l_freq_env, h_freq_env = h_freq_env, output_freq = output_freq)
        eeg_measurement.write_subjects_eeg_data(data_filename = data_filename, l_freq_eeg = l_freq_eeg, h_freq_eeg = h_freq_eeg, output_freq = output_freq, use_ica=False)

def create_dataset_05_09():
    #create dataset with ica cleaned data, based SNR criterion for paper revision

    subjects = list(range(102,115))
    subjects = subjects + list(range(116,117))
    subjects = subjects + list(range(118,126))
    subjects = subjects + [127, 128, 130]


    aux_channels_correct = np.ones(len(subjects), dtype=int).tolist()
    #subject 103 has switched aux channels
    aux_channels_correct[1] = 0

    l_freq_env = 1.0
    h_freq_env = 50.0

    l_freq_eeg = 1.0
    h_freq_eeg = 32

    output_freq = 125
    #data_filename = 'ci_attention_final_l_1,1_h_32,50_out_125_130_incl_ica.hdf5'
    data_filename='ci_attention_final_SNR_crit.hdf5'
    base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    for subject, aux_correct in zip(subjects, aux_channels_correct):
        eeg_measurement = EegMeasurement(subject, base_dir, aux_correct, eeg_reference='avg')

        #write stimulus data only once
        if subject == 102:
            eeg_measurement.write_stimulus_data(data_filename = data_filename, l_freq_env=l_freq_env, h_freq_env = h_freq_env, output_freq = output_freq)
        eeg_measurement.write_subjects_eeg_data(data_filename = data_filename, l_freq_eeg = l_freq_eeg, h_freq_eeg = h_freq_eeg, output_freq = output_freq, use_ica=True)


def create_dataset_05_09_snr_15():
    #create dataset with ica cleaned data, based SNR criterion for paper revision

    subjects = list(range(102,115))
    subjects = subjects + list(range(116,117))
    subjects = subjects + list(range(118,126))
    subjects = subjects + [127, 128, 130]


    aux_channels_correct = np.ones(len(subjects), dtype=int).tolist()
    #subject 103 has switched aux channels
    aux_channels_correct[1] = 0

    l_freq_env = 1.0
    h_freq_env = 50.0

    l_freq_eeg = 1.0
    h_freq_eeg = 32

    output_freq = 125

    snr_thres = 15
    #data_filename = 'ci_attention_final_l_1,1_h_32,50_out_125_130_incl_ica.hdf5'
    data_filename='ci_attention_final_SNR_crit_15dB.hdf5'
    base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    for subject, aux_correct in zip(subjects, aux_channels_correct):
        eeg_measurement = EegMeasurement(subject, base_dir, aux_correct, eeg_reference='avg')

        #write stimulus data only once
        if subject == 102:
            eeg_measurement.write_stimulus_data(data_filename = data_filename, l_freq_env=l_freq_env, h_freq_env = h_freq_env, output_freq = output_freq)
        eeg_measurement.write_subjects_eeg_data(data_filename = data_filename, l_freq_eeg = l_freq_eeg, h_freq_eeg = h_freq_eeg, output_freq = output_freq, use_ica=True, snr_thres = snr_thres)


def create_dataset_05_09_snr_10():
    #create dataset with ica cleaned data, based SNR criterion for paper revision

    subjects = list(range(102,115))
    subjects = subjects + list(range(116,117))
    subjects = subjects + list(range(118,126))
    subjects = subjects + [127, 128, 130]


    aux_channels_correct = np.ones(len(subjects), dtype=int).tolist()
    #subject 103 has switched aux channels
    aux_channels_correct[1] = 0

    l_freq_env = 1.0
    h_freq_env = 50.0

    l_freq_eeg = 1.0
    h_freq_eeg = 32

    output_freq = 125

    snr_thres = 10
    #data_filename = 'ci_attention_final_l_1,1_h_32,50_out_125_130_incl_ica.hdf5'
    data_filename='ci_attention_final_SNR_crit_10dB.hdf5'
    base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    for subject, aux_correct in zip(subjects, aux_channels_correct):
        eeg_measurement = EegMeasurement(subject, base_dir, aux_correct, eeg_reference='avg')

        #write stimulus data only once
        if subject == 102:
            eeg_measurement.write_stimulus_data(data_filename = data_filename, l_freq_env=l_freq_env, h_freq_env = h_freq_env, output_freq = output_freq)
        eeg_measurement.write_subjects_eeg_data(data_filename = data_filename, l_freq_eeg = l_freq_eeg, h_freq_eeg = h_freq_eeg, output_freq = output_freq, use_ica=True, snr_thres = snr_thres)

def create_ci_dataset_eye_art():
    #create dataset from ci patient, remove only eye artifacts, rm ci artifacts in subsequent step

    subjects = list(range(102,115))
    subjects = subjects + list(range(116,117))
    subjects = subjects + list(range(118,126))
    subjects = subjects + [127, 128, 130]


    aux_channels_correct = np.ones(len(subjects), dtype=int).tolist()
    #subject 103 has switched aux channels
    aux_channels_correct[1] = 0

    l_freq_env = 1.0
    h_freq_env = 50.0

    l_freq_eeg = 1.0
    h_freq_eeg = 32

    output_freq = 125
    #data_filename = 'ci_attention_final_l_1,1_h_32,50_out_125_130_incl_ica.hdf5'
    data_filename='ci_attention_artifact_removal_125Hz.hdf5'
    base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    for subject, aux_correct in zip(subjects, aux_channels_correct):
        eeg_measurement = EegMeasurement(subject, base_dir, aux_correct, eeg_reference='avg')

        #write stimulus data only once
        if subject == 102:
            eeg_measurement.write_stimulus_data(data_filename = data_filename, l_freq_env=l_freq_env, h_freq_env = h_freq_env, output_freq = output_freq)
        eeg_measurement.write_subjects_eeg_data(data_filename = data_filename, l_freq_eeg = l_freq_eeg, h_freq_eeg = h_freq_eeg, output_freq = output_freq, use_ica=True, rm_eye_muscle = True, log_ica_comp=True)


def create_ci_dataset_art():
    #create dataset from ci patient, remove only eye and muscle artifacts, rm ci artifacts in subsequent step

    subjects = list(range(102,115))
    subjects = subjects + list(range(116,117))
    subjects = subjects + list(range(118,126))
    subjects = subjects + [127, 128, 130]


    aux_channels_correct = np.ones(len(subjects), dtype=int).tolist()
    #subject 103 has switched aux channels
    aux_channels_correct[1] = 0

    l_freq_env = 1.0
    h_freq_env = 50.0

    l_freq_eeg = 1.0
    h_freq_eeg = 280

    output_freq = 1000
    #data_filename = 'ci_attention_final_l_1,1_h_32,50_out_125_130_incl_ica.hdf5'
    data_filename='ci_attention_artifact_removal.hdf5'
    base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    for subject, aux_correct in zip(subjects, aux_channels_correct):
        eeg_measurement = EegMeasurement(subject, base_dir, aux_correct, eeg_reference='avg')

        #write stimulus data only once
        if subject == 102:
            eeg_measurement.write_stimulus_data(data_filename = data_filename, l_freq_env=l_freq_env, h_freq_env = h_freq_env, output_freq = output_freq)
        eeg_measurement.write_subjects_eeg_data(data_filename = data_filename, l_freq_eeg = l_freq_eeg, h_freq_eeg = h_freq_eeg, output_freq = output_freq, use_ica=True, rm_eye_muscle = True)


def create_snr_threshold_datasets():
    # create datasets with different SNR thresholds

    subjects = list(range(102,115))
    subjects = subjects + list(range(116,117))
    subjects = subjects + list(range(118,126))
    subjects = subjects + [127, 128, 130]


    aux_channels_correct = np.ones(len(subjects), dtype=int).tolist()
    #subject 103 has switched aux channels
    aux_channels_correct[1] = 0

    l_freq_env = 1.0
    h_freq_env = 50.0

    l_freq_eeg = 1.0
    h_freq_eeg = 32

    output_freq = 125
    snr_thres_list = [10, 15, 20, 25, 27, 30]
    #data_filename = 'ci_attention_final_l_1,1_h_32,50_out_125_130_incl_ica.hdf5'
    # data_filename='ci_attention_final_SNR_crit_eye_20dB.hdf5'
    for snr_thres in tqdm(snr_thres_list):
        data_filename = f'ci_attention_final_SNR_crit_{snr_thres}dB.hdf5'
        base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
        for subject, aux_correct in zip(subjects, aux_channels_correct):
            eeg_measurement = EegMeasurement(subject, base_dir, aux_correct, eeg_reference='avg')

            #write stimulus data only once
            # if subject == 102:
            #     eeg_measurement.write_stimulus_data(data_filename = data_filename, l_freq_env=l_freq_env, h_freq_env = h_freq_env, output_freq = output_freq)
            eeg_measurement.write_subjects_eeg_data(data_filename = data_filename, l_freq_eeg = l_freq_eeg, h_freq_eeg = h_freq_eeg, output_freq = output_freq, use_ica=True, rm_eye_muscle = False, rm_ci = True, snr_thres = snr_thres)

def create_snr_threshold_datasets_5ms():
    # create datasets with different SNR thresholds +/- 5ms to onset

    subjects = list(range(102,115))
    subjects = subjects + list(range(116,117))
    subjects = subjects + list(range(118,126))
    subjects = subjects + [127, 128, 130]


    aux_channels_correct = np.ones(len(subjects), dtype=int).tolist()
    #subject 103 has switched aux channels
    aux_channels_correct[1] = 0

    l_freq_env = 1.0
    h_freq_env = 50.0

    l_freq_eeg = 1.0
    h_freq_eeg = 32

    output_freq = 125
    snr_thres_list = [0, 5, 7.5, 10, 12.5, 15, 20]
    #data_filename = 'ci_attention_final_l_1,1_h_32,50_out_125_130_incl_ica.hdf5'
    # data_filename='ci_attention_final_SNR_crit_eye_20dB.hdf5'
    for snr_thres in tqdm(snr_thres_list):
        data_filename = f'ci_attention_SNR_5ms_{int(snr_thres)}dB.hdf5'
        base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
        for subject, aux_correct in zip(subjects, aux_channels_correct):
            eeg_measurement = EegMeasurement(subject, base_dir, aux_correct, eeg_reference='avg')

            # write stimulus data only once
            if subject == 102:
                eeg_measurement.write_stimulus_data(data_filename = data_filename, l_freq_env=l_freq_env, h_freq_env = h_freq_env, output_freq = output_freq)
            eeg_measurement.write_subjects_eeg_data(data_filename = data_filename, l_freq_eeg = l_freq_eeg, h_freq_eeg = h_freq_eeg, output_freq = output_freq, use_ica=True, rm_eye_muscle = False, rm_ci = True, snr_thres = snr_thres)

if __name__ == '__main__':
    create_snr_threshold_datasets_5ms()