from src.data.eeg_measurement import EegMeasurement
from src.data.eeg_attention_pytorch_dataset import EegAttentionDataset
from src.evaluation.eval_vanilla_ridge import RidgeEvaluator
import numpy as np
import git
import os
import pandas
import pickle

import argparse

parser = argparse.ArgumentParser(description='Train CNN model on EEG attention dataset.')

parser.add_argument('-model_id', type=str, help='model id')
args = parser.parse_args()

####
# 1. create dataset with ica cleaned data and onset stimulus
####


def create_dataset_19_10():
    #create dataset with ica cleaned data

    #Specify newly measured subjects
    subjects = list(range(101,115))
    subjects.append(116)
    aux_channels_correct = np.ones(15, dtype=int).tolist()
    #subject 103 has switched aux channels
    aux_channels_correct[2] = 0

    l_freq_env = 1.0
    h_freq_env = 32.0
    l_freq_eeg = 1.0
    h_freq_eeg = 32

    output_freq = 125

    data_filename = 'ci_l_1,1_h_32,32_out_125_116_incl_ica_onset.hdf5'

    base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir

    for subject, aux_correct in zip(subjects, aux_channels_correct):
        eeg_measurement = EegMeasurement(subject, base_dir, aux_correct)
        if subject == 101:
            eeg_measurement.write_stimulus_data(data_filename = data_filename, l_freq_env = l_freq_env, h_freq_env = h_freq_env, output_freq = output_freq)
        eeg_measurement.write_subjects_eeg_data(data_filename = data_filename, l_freq_eeg = l_freq_eeg, h_freq_eeg = h_freq_eeg, output_freq = output_freq)

####
# 2.Fit backward model on:
#       a) raw data
#       b) ica cleaned data
#       c) raw data with onset stimulus
#       d) ica cleaned data with onset stimulus
####

def cross_val_models():
    data_filename = 'ci_l_1,1_h_32,32_out_125_116_incl_ica_onset.hdf5'
    #data_filename = 'test_ci.hdf5'
    random_state = 672
    competing_indices_book_0 = np.arange(8,19,2, dtype=int)
    competing_indices_book_1 = np.arange(9,20,2, dtype=int)
    competing_indcies = np.arange(8,20,1,dtype=int)
    accuracy_window_size = 55
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

    #for debugging
    #test_indices  = [test_indices[0]]
    #val_indices = [val_indices[0]]

    freq_range = '1-32Hz'

    #reference model raw data on envelope
    model_id = '003'

    eval_raw_env = RidgeEvaluator(-200, 800, database_filename= data_filename, training_scheme='concat', 
                                        freq_range= freq_range, use_ica_data=False, speech_feature= 'env', model_id=model_id)
    eval_raw_env.eval_competing_speaker_cross_val(test_indices_list = test_indices, val_indices_list = val_indices, accuracy_window_size = accuracy_window_size)

    #model using ica cleaned eeg data
    model_id = '004'
    eval_ica_env = RidgeEvaluator(-200, 800, database_filename= data_filename, training_scheme='concat', 
                                        freq_range= freq_range, use_ica_data=True, speech_feature= 'env', model_id=model_id)
    eval_ica_env.eval_competing_speaker_cross_val(test_indices_list = test_indices, val_indices_list = val_indices, accuracy_window_size = accuracy_window_size)

    #model using raw data and onset stimulus
    model_id = '005'
    eval_raw_onset_env = RidgeEvaluator(-200, 800, database_filename= data_filename, training_scheme='concat', 
                                        freq_range= freq_range, use_ica_data=False, speech_feature= 'onset_env', model_id=model_id)
    eval_raw_onset_env.eval_competing_speaker_cross_val(test_indices_list = test_indices, val_indices_list = val_indices, accuracy_window_size = accuracy_window_size)


def cross_val_ica_onset():
    data_filename = 'ci_l_1,1_h_32,32_out_125_116_incl_ica_onset.hdf5'
    #data_filename = 'test_ci.hdf5'
    random_state = 672
    competing_indices_book_0 = np.arange(8,19,2, dtype=int)
    competing_indices_book_1 = np.arange(9,20,2, dtype=int)
    competing_indcies = np.arange(8,20,1,dtype=int)
    accuracy_window_size = 55
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

    #for debugging
    #test_indices  = [test_indices[0]]
    #val_indices = [val_indices[0]]

    freq_range = '1-32Hz'

    #model using raw data and onset stimulus
    model_id = '006'
    eval_raw_onset_env = RidgeEvaluator(-200, 800, database_filename= data_filename, training_scheme='concat', 
                                        freq_range= freq_range, use_ica_data=True, speech_feature= 'onset_env', model_id=model_id)
    eval_raw_onset_env.eval_competing_speaker_cross_val(test_indices_list = test_indices, val_indices_list = val_indices, accuracy_window_size = accuracy_window_size)


#####
# 3.Apply ica on coeficients on standard model
######
def coef_ica():
    base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    standard_model_id = '003'
    data_filename = 'ci_l_1,1_h_32,32_out_125_116_incl_ica_onset.hdf5'
    model_id_comp_ica = '006'

    #getting the test indices from the standard model
    random_state = 672
    competing_indices_book_0 = np.arange(8,19,2, dtype=int)
    competing_indices_book_1 = np.arange(9,20,2, dtype=int)
    competing_indcies = np.arange(8,20,1,dtype=int)
    accuracy_window_size = 55
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

    subjects = list(range(101,115))
    subjects.append(116)

    for subject in subjects:
        model_path = os.path.join(base_dir, "models", "ridge", "concat", standard_model_id, str(subject))
        list_models = [os.path.join(model_path, mdl) for mdl in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, mdl))]
        
        list_models = list(np.roll(np.array(sorted(list_models)),1))
        subj_attended_scores, subj_distractor_scores, subj_accuracies = [],[],[]

        for model, val_index, test_index in zip(list_models, val_indices, test_indices):
            mdl = pandas.read_pickle(model)
            ####
            #get coefs
            # apply ica
            # evaluate that model
            #####

####
# 4. compare results on different methods
####    a) correlation coefficients
####    b) accuracies
####  


#####
# 6. Visualize results
#####

def cross_val_models_complete():
    """
    Crossval over all competing trials
    """
    data_filename = 'ci_l_1,1_h_32,32_out_125_116_incl_ica_onset.hdf5'
    accuracy_window_size = 55

    #convert to list of arrays - needed this way by evaluator class
    test_indices = [np.array([i]) for i in range(8,20)]
    val_indices = np.roll(np.array(test_indices), 1).reshape(-1).tolist()
    val_indices = [np.array([i]) for i in val_indices]

    freq_range = '1-32Hz'

    #reference model raw data on envelope
    model_id = '007'
    eval_raw_env = RidgeEvaluator(-200, 800, database_filename= data_filename, training_scheme='concat', 
                                        freq_range= freq_range, use_ica_data=False, speech_feature= 'env', model_id=model_id)
    eval_raw_env.eval_competing_speaker_cross_val(test_indices_list = test_indices, val_indices_list = val_indices, accuracy_window_size = accuracy_window_size)

    #model using ica cleaned eeg data
    model_id = '008'
    eval_ica_env = RidgeEvaluator(-200, 800, database_filename= data_filename, training_scheme='concat', 
                                        freq_range= freq_range, use_ica_data=True, speech_feature= 'env', model_id=model_id)
    eval_ica_env.eval_competing_speaker_cross_val(test_indices_list = test_indices, val_indices_list = val_indices, accuracy_window_size = accuracy_window_size)

    #model using raw data and onset stimulus
    model_id = '009'
    eval_raw_onset_env = RidgeEvaluator(-200, 800, database_filename= data_filename, training_scheme='concat', 
                                        freq_range= freq_range, use_ica_data=False, speech_feature= 'onset_env', model_id=model_id)
    eval_raw_onset_env.eval_competing_speaker_cross_val(test_indices_list = test_indices, val_indices_list = val_indices, accuracy_window_size = accuracy_window_size)
    
    model_id = '010'
    eval_raw_onset_env = RidgeEvaluator(-200, 800, database_filename= data_filename, training_scheme='concat', 
                                        freq_range= freq_range, use_ica_data=True, speech_feature= 'onset_env', model_id=model_id)
    eval_raw_onset_env.eval_competing_speaker_cross_val(test_indices_list = test_indices, val_indices_list = val_indices, accuracy_window_size = accuracy_window_size)


####
#7. investigate per story model - with intent of learnable thresholding
###

def cross_val_models_complete_per_story():
    base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    data_filename = 'ci_l_1,1_h_32,32_out_125_116_incl_ica_onset.hdf5'
    accuracy_window_size = [55, 20]
    model_id = '011'

    #convert to list of arrays - needed this way by evaluator class
    test_indices = [np.array([i]) for i in range(8,20)]
    val_indices = np.roll(np.array(test_indices), 1).reshape(-1).tolist()
    val_indices = [np.array([i]) for i in val_indices]

    freq_range = '1-32Hz'
    eval_per_story = RidgeEvaluator(-200, 800, database_filename= data_filename, training_scheme='concat', 
                                        freq_range= freq_range, use_ica_data=False, speech_feature= 'env', model_id=model_id)
    
    elb_scores_list, pol_scores_list, accuracies_list, labels_list = [],[],[],[]

    #for debugging
    # test_indices  = [test_indices[0]]
    # val_indices = [val_indices[0]]
    for val_index, test_index in zip(val_indices, test_indices):
        elb_scores, pol_scores, label, accuracies = eval_per_story.run_story_model_eval(val_indices=val_index, test_indices=test_index, accuracy_window_size=accuracy_window_size, debug=False)
        elb_scores_list.append(elb_scores)
        pol_scores_list.append(pol_scores)
        labels_list.append(label)
        accuracies_list.append(accuracies)
    
    #save results
    #shape: testindicex x subjects x acc_window
    #with datadimension inhomogeneous 

    metric_path = os.path.join(base_dir, "reports", "metrics", "ridge", model_id)
    os.makedirs(metric_path, exist_ok=True)
    #pickle dump the results
    with open(os.path.join(metric_path, 'overall_elb_scores.pkl'), 'wb') as f:
        pickle.dump(elb_scores_list, f)
    with open(os.path.join(metric_path, 'overall_pol_scores.pkl'), 'wb') as f:
        pickle.dump(pol_scores_list, f)
    with open(os.path.join(metric_path, 'overall_labels.pkl'), 'wb') as f:
        pickle.dump(labels_list, f)
    with open(os.path.join(metric_path, 'overall_accuracies.pkl'), 'wb') as f:
        pickle.dump(accuracies_list, f)

####
#8. investigate on new data
###

def cross_val_models_complete_dataset():
    """
    Crossval over all competing trials
    """
    data_filename = 'ci_attention_final_SNR_crit_30dB.hdf5'
    accuracy_window_size = 55

    #convert to list of arrays - needed this way by evaluator class
    test_indices = [np.array([i]) for i in range(8,20)]
    val_indices = np.roll(np.array(test_indices), 1).reshape(-1).tolist()
    val_indices = [np.array([i]) for i in val_indices]

    freq_range = '1-32Hz'

    #reference model raw data on envelope
    if args.model_id == '030':
        model_id = args.model_id
        eval_raw_env = RidgeEvaluator(-200, 800, database_filename= data_filename, training_scheme='concat', 
                                            freq_range= freq_range, use_ica_data=False, speech_feature= 'env', model_id=model_id)
        eval_raw_env.eval_competing_speaker_cross_val(test_indices_list = test_indices, val_indices_list = val_indices, accuracy_window_size = accuracy_window_size)

    #model using ica cleaned eeg data
    elif args.model_id == '031':
        model_id = args.model_id
        eval_ica_env = RidgeEvaluator(-200, 800, database_filename= data_filename, training_scheme='concat', 
                                            freq_range= freq_range, use_ica_data=True, speech_feature= 'env', model_id=model_id)
        eval_ica_env.eval_competing_speaker_cross_val(test_indices_list = test_indices, val_indices_list = val_indices, accuracy_window_size = accuracy_window_size)

    #model using raw data and onset stimulus
    elif args.model_id == '032':
        model_id = args.model_id
        eval_raw_onset_env = RidgeEvaluator(-200, 800, database_filename= data_filename, training_scheme='concat', 
                                            freq_range= freq_range, use_ica_data=False, speech_feature= 'onset_env', model_id=model_id)
        eval_raw_onset_env.eval_competing_speaker_cross_val(test_indices_list = test_indices, val_indices_list = val_indices, accuracy_window_size = accuracy_window_size)
    
    elif args.model_id == '033':
        model_id = args.model_id
        eval_raw_onset_env = RidgeEvaluator(-200, 800, database_filename= data_filename, training_scheme='concat', 
                                            freq_range= freq_range, use_ica_data=True, speech_feature= 'onset_env', model_id=model_id)
        eval_raw_onset_env.eval_competing_speaker_cross_val(test_indices_list = test_indices, val_indices_list = val_indices, accuracy_window_size = accuracy_window_size)

def snr_ablation_study():
    # study the effect of different snr levels for ci artifact rejection on the performance on the linear decoding method
    accuracy_window_size = 55
    test_indices = [np.array([i]) for i in range(8,20)]
    val_indices = np.roll(np.array(test_indices), 1).reshape(-1).tolist()
    val_indices = [np.array([i]) for i in val_indices]
    freq_range = '1-32Hz'
    dB_list = [0, 5, 7.5, 10, 12.5, 15, 20]
    model_ids = [str(x) for x in range(50, 57)]

    for dB, model_id in zip(dB_list, model_ids):
        data_filename = f'ci_attention_SNR_5ms_{int(dB)}dB.hdf5'
        eval_raw_env = RidgeEvaluator(-200, 800, database_filename= data_filename, training_scheme='concat', 
                                            freq_range= freq_range, use_ica_data=True, speech_feature= 'env', model_id=model_id)
        eval_raw_env.eval_competing_speaker_cross_val(test_indices_list = test_indices, val_indices_list = val_indices, accuracy_window_size = accuracy_window_size)
        
if __name__ == '__main__':
    #create_dataset_19_10()
    snr_ablation_study()