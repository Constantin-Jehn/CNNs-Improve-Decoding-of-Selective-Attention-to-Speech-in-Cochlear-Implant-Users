#file to run evaluation of CNN models

import torch
#Quantization modules
from torch.ao.quantization import (
  get_default_qconfig_mapping,
  get_default_qat_qconfig_mapping,
  QConfigMapping,
)

import torch.ao.quantization.quantize_fx as quantize_fx
import numpy as np
from src.models.dnn import CNN, CNN_2
from src.data.eeg_attention_pytorch_dataset import EegAttentionDataset
import os
import git
import pickle
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from src.evaluation.eval_vanilla_ridge import RidgeEvaluator
import copy
from tqdm import tqdm
import random
import argparse

parser = argparse.ArgumentParser(description='Evaluate CNN models')
parser.add_argument('-model_id', type=int, help='model identifier either 76,86,96')
parser.add_argument('-dB', type=float, help='SNR threshold of CI artifact rejection')
parser.add_argument('-quantize', type=int, help='0 for normal evaluation, 1 for quantized evaluation')
parser.add_argument('-break_calibration', type=int, help='Introduce too large or small number in calibration to test quantization')
args = parser.parse_args()

#whether to use post-training quantization
#https://pytorch.org/docs/stable/quantization.html
quantize = args.quantize
break_calibration = args.break_calibration
qconfig_mapping = get_default_qat_qconfig_mapping('qnnpack')
#check this option to see if activations are quantized
#qconfig_mapping = get_default_qat_qconfig_mapping('qnnpack')


use_ica = False
if args.model_id == 152:
    model = CNN
    #Pretraining + Finetuning
    model_ids_152 = [f'{i:03}' for i in range(140,151,2)]
    metrics_id_152 = '152'
    
    #Subject Specific
    model_ids_153 = [f'{i:03}' for i in range(141,152,2)]
    metrics_id_153 = '153'

    #env + ica
    model_ids_176 = [f'{i:03}' for i in range(170,176)]
    metrics_id_176 = '176'

    #onset + raw eeg
    model_ids_186 = [f'{i:03}' for i in range(180,186)]
    metrics_id_186 = '186'

    #ica + onset
    model_ids_196 = [f'{i:03}' for i in range(190,196)]
    metrics_id_196 = '196'

    master_model_id_list = [model_ids_152, model_ids_153, model_ids_176, model_ids_186, model_ids_196]
    master_metrics_id = [metrics_id_152, metrics_id_153, metrics_id_176, metrics_id_186, metrics_id_196]
elif args.model_id in [200, 202, 204, 206]:
    #ReLU model
    #all subjects in same model_id:
    master_model_id_list = [[f'{args.model_id:03}' for _ in range (0,6)]]
    if quantize:
        if break_calibration:
            master_metrics_id = [f'{args.model_id+3:03}']
        else:
            #standard calibration
            master_metrics_id = [f'{args.model_id+2:03}']
    else:
        master_metrics_id = [f'{args.model_id+1:03}']
    model = CNN_2
elif args.model_id == 154:
    #quantized ELU model
    quantize = 1
    master_model_id_list = [[f'{i:03}' for i in range(140,151,2)]]
    master_metrics_id = ['154']
elif args.model_id in [210, 212, 214]:
    quantize = 0
    master_model_id_list = [[str(args.model_id)]]
    subject_strings_list = [['102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112','113', '114', '116', '118', '119', '120', '121', '122', '123', '124','125', '127', '128', '130']]
    master_metrics_id = [str(args.model_id +1)]
    model = CNN
    use_ica = True
elif (args.model_id in list(range(215, 221)) or args.model_id in list(range(300, 307))):
    quantize = 0
    master_model_id_list = [[str(args.model_id)]]
    subject_strings_list = [['102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112','113', '114', '116', '118', '119', '120', '121', '122', '123', '124','125', '127', '128', '130']]
    master_metrics_id = [str(args.model_id)]
    model = CNN
    use_ica = True
else:
    print('Invalid model id')
    exit()

#subjects for the model ids
# subject_strings_list = [['102', '103', '104', '105', '106'],['107', '108', '109', '110'],['111', '112','113', '114'],['116', '118', '119', '120'],['121', '122', '123', '124'], ['125', '127', '128', '130']]

srate = 125
base_dir = base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
if args.dB is not None:
    data_dir = os.path.join(base_dir, f"data/processed/ci_attention_SNR_5ms_{int(args.dB)}dB.hdf5")

#subject string for model ids
if args.model_id == 200:
    activation_fct = 'LeakyReLU'
    conv_bias = 1
elif args.model_id == 202:
    activation_fct = 'ELU'
    conv_bias = 0
elif args.model_id == 204:
    activation_fct = 'LeakyReLU'
    conv_bias = 0
elif args.model_id == 206:
    activation_fct = 'ELU'
    conv_bias = 0
elif args.model_id in [210, 212, 214, 215, 216, 217, 218, 219, 220, 300, 301, 302, 303, 304, 305, 306]:
    activation_fct = 'ELU'
    conv_bias = 0

if isinstance(model, CNN_2):
    cnn_hyperparameters = {"dropout_rate": 0.35,"F1": 8,"D": 8,"F2": 64, "input_length": 100, "num_input_channels": 31, "activation": activation_fct, "conv_bias": conv_bias}
else:
    cnn_hyperparameters = {"dropout_rate": 0.35,"F1": 8,"D": 8,"F2": 64, "input_length": 100, "num_input_channels": 31, "activation": activation_fct}
cnn_train_params = {"data_dir": data_dir, "lr": 0.00005, "batch_size": 256, "weight_decay": 1e-08, "epochs": 20, "early_stopping_patience": 7}

#decision windows to evaluate models on
dec_window_sizes = [60,45,30,20,10,5,2,1]

# **Check model size**
cnn = model(**cnn_hyperparameters)
window_size_training = cnn_hyperparameters['input_length']
total_params = sum(p.numel() for p in cnn.parameters())
print("Total number of parameters: ", total_params)


# **Set test indices: For complete cross-validation**
test_indices = [np.array([i]) for i in range(8,20,1)]
test_indices_string = [str(i.item()) for i in test_indices]

def get_dnn_predictions(dnn, loader, device='cuda'): # move to pipeline

     predictions = []
     corr_class, false_class = 0, 0

     pearson_attended_list, pearson_distractor_list = [], []
    
     dnn.eval()

     dnn = dnn.to(device)

     with torch.no_grad():
        for x, y_attended, y_distractor in loader:
            x = x.to(device, dtype=torch.float)
            y_hat = dnn(x)
            predictions.append(y_hat.cpu().numpy())

            pearson_attended, _ = pearsonr(y_hat, y_attended)
            pearson_distractor, _ = pearsonr(y_hat, y_distractor)

            pearson_attended_list.append(pearson_attended)
            pearson_distractor_list.append(pearson_distractor)

            if pearson_attended > pearson_distractor:
                corr_class+=1
            elif pearson_attended < pearson_distractor:
                false_class+=1
            else:
                corr_class+=1
                false_class+=1
        acc = corr_class / (corr_class + false_class)
     return predictions, acc, np.mean(np.array(pearson_attended_list)), pearson_attended_list, np.mean(np.array(pearson_distractor_list)), pearson_distractor_list


def get_model_params(model_id, subject_string, test_index_string):
    """
    Get model parameters from model_kwargs.pkl file
    Args:
        model_id (string): model identifier e.g. '045'
        subject_string (string): subject identfier e.g. '101'
        test_index_string (string): test index as string e.g. '8'

    Returns:
        dict: model parameters
    """
    param_path = os.path.join(base_dir, 'models', 'cnn', 'checkpoints', model_id, subject_string, 'test_ind_'+test_index_string, 'model_kwargs.pkl')
    with open(param_path, 'rb') as f:
        model_kwargs = pickle.load(f)
    return model_kwargs

def get_best_model_path(model_id, subject, test_index_string):
    """
    Get path to best model in checkpoint folder

    Args:
        model_id (string): model identifier e.g. '045'
        subject (string): subject identfier e.g. '101'
        test_index_string (string): test index as string e.g. '8'

    Returns:
        string: path to best model
    """
    model_path = os.path.join(base_dir, "models/cnn/checkpoints", model_id, subject, 'test_ind_' + test_index_string)
    #get all files starting with epoch
    file_list = os.listdir(model_path)
    file_list = [os.path.join(model_path, file) for file in file_list if file.startswith("epoch")]
    #get file with latest creation date
    if len(file_list) == 0:
        latest_file = os.path.join(model_path, "best_model.ckpt")
        if not os.path.exists(latest_file):
            latest_file = None
    else:
        modification_times = np.array(([os.stat(path_to_file).st_mtime for path_to_file in file_list]))
        latest_file = file_list[np.argmax(modification_times)]
    return latest_file

def get_best_mdoel_dual_speaker(model_id, subject, test_index_string):
    """
    Get path to best model ind dual speaker scenario

    Args:
        model_id (string): model identifier e.g. '045'
        subject (string): subject identfier e.g. '101'
        test_index_string (string): test index as string e.g. '8'

    Returns:
        string: path to best model
    """
    return os.path.join(base_dir, "models/cnn/checkpoints", model_id, subject, 'test_ind_' + test_index_string, 'best_model_dual_speaker.ckpt')


def get_best_model_single_speaker(model_id, subject, test_index_string):
    """
    Get path to best model ind single speaker scenario

    Args:
        model_id (string): model identifier e.g. '045'
        subject (string): subject identfier e.g. '101'
        test_index_string (string): test index as string e.g. '8'

    Returns:
        string: path to best model
    """
    return os.path.join(base_dir, "models/cnn/checkpoints", model_id, subject, 'test_ind_' + test_index_string, 'best_model_single_speaker.ckpt')


if __name__ == "__main__":
    #initialize lists for metrics
    for model_id_list, metrics_id in zip(master_model_id_list, master_metrics_id):
        overall_acc = []
        #average for each subject
        overall_r_attended_mean = []
        overall_r_distractor_mean = []
        #raw correlations
        overall_r_attended_raw = []
        overall_r_distractor_raw = []

        #to evaluate multiple models on multiple subjects
        for model_id, subject_strings in zip(model_id_list, subject_strings_list):
            print(f"Evaluating model {model_id} on {len(subject_strings)} subjects")
            
            #evaluate all subjects contained in that model
            for subject in subject_strings:
                subject_acc = []
                subject_r_attended_mean = []
                subject_r_distractor_mean = []
                subject_r_attended_raw = []
                subject_r_distractor_raw = []

                for test_index, test_index_string in zip(test_indices, test_indices_string):
                    model_dict_path = get_best_model_path(model_id, subject, test_index_string)
                    state_dic = torch.load(model_dict_path, map_location=torch.device('cpu'))
    
                    if state_dic:
                        pass
                    else:
                        print(f'No model found for subject {subject} and test index {test_index_string}')
                        model_invalid = True
                        n = 0
                        while model_invalid and n < 15:
                            replace_subj = random.choice(subject_strings)
                            model_dict_path = get_best_model_path(model_id, replace_subj, test_index_string)
                            state_dic = torch.load(model_dict_path, map_location=torch.device('cpu'))
                            if state_dic:
                                model_invalid = False
                            n+=1
                        if model_invalid:
                            print(f'No model found for subject {subject} and test index {test_index_string}')
                            continue
                        else:
                            print(f'Using model from subject {replace_subj} instead')

                    #load model
                    trained_CNN = model(**cnn_hyperparameters)
                    trained_CNN.load_state_dict(state_dic)

                    test_dataset = EegAttentionDataset(dir_h5 = data_dir, subj_string=subject, trials_indices=test_index, window_size_training=window_size_training, sampling_rate = srate, ica = use_ica)
                    
                    if quantize:
                        #retrieve data to calibrate quantization parameters
                        calibration_trials = np.setdiff1d(np.arange(20), test_index)
                        train_dataset = EegAttentionDataset(dir_h5 = data_dir, subj_string=subject, trials_indices=calibration_trials, window_size_training=window_size_training, sampling_rate = srate, ica = use_ica)
                        calibration_indices = np.random.choice(len(train_dataset), 100)
                        calibration_dataset = torch.utils.data.Subset(train_dataset, calibration_indices)
                        calibration_dataloader = DataLoader(train_dataset, batch_size= 20 * srate, shuffle=False, drop_last=True)
                        
                        trained_CNN.eval()
                        #prepares model for post training quantizationn
                        model_prepared = quantize_fx.prepare_fx(trained_CNN, qconfig_mapping, calibration_dataset)
                        #converts the prepared model to a quantized model and return torch.nn module
                        
                        #calibrate the quantization parameters
                        with torch.inference_mode():
                            for eeg, _, _ in calibration_dataloader:
                                if break_calibration:
                                    eeg = eeg * 0.00000001
                                model_prepared(eeg.float())
                        trained_CNN = quantize_fx.convert_fx(model_prepared)


                    #predictions for decreasing window sizes
                    window_acc, window_r_attended, window_r_distractor = [], [], []
                    window_r_attended_raw, window_r_distractor_raw = [], []
                    for decision_window_s in dec_window_sizes:
                        test_dataloader = DataLoader(test_dataset, batch_size= decision_window_s * srate, shuffle=False, drop_last=True)
                        prediction, acc, pearson_attended_mean, pearson_attended_list, pearson_distractor_mean, pearson_distractor_list = get_dnn_predictions(trained_CNN, test_dataloader, device='cpu')
                        
                        window_acc.append(acc)
                        window_r_attended.append(pearson_attended_mean)
                        window_r_distractor.append(pearson_distractor_mean)

                        window_r_attended_raw.append(pearson_attended_list)
                        window_r_distractor_raw.append(pearson_distractor_list)


                    subject_acc.append(window_acc)
                    subject_r_attended_mean.append(window_r_attended)
                    subject_r_distractor_mean.append(window_r_distractor)

                    subject_r_attended_raw.append(window_r_attended_raw)
                    subject_r_distractor_raw.append(window_r_distractor_raw)


                print(f'subject {subject} accuracies on 60,45,30,15,10,5,2,1s: {np.mean(np.array(subject_acc), axis=0)}')

                overall_acc.append(subject_acc)
                overall_r_attended_mean.append(subject_r_attended_mean)
                overall_r_distractor_mean.append(subject_r_distractor_mean)

                overall_r_attended_raw.append(subject_r_attended_raw)
                overall_r_distractor_raw.append(subject_r_distractor_raw)
        
        #save metrics as lists using pickle
        path = os.path.join(base_dir, 'reports', 'metrics', 'cnn', metrics_id + 'pickled_raw')
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'accuracies.pkl'), 'wb') as f:
            pickle.dump(overall_acc, f)
        with open(os.path.join(path, 'attended_scores_mean.pkl'), 'wb') as f:
            pickle.dump(overall_r_attended_mean, f)
        with open(os.path.join(path, 'distractor_scores_mean.pkl'), 'wb') as f:
            pickle.dump(overall_r_distractor_mean, f)
        with open(os.path.join(path, 'attended_scores_raw.pkl'), 'wb') as f:
            pickle.dump(overall_r_attended_raw, f)
        with open(os.path.join(path, 'distractor_scores_raw.pkl'), 'wb') as f:
            pickle.dump(overall_r_distractor_raw, f)



