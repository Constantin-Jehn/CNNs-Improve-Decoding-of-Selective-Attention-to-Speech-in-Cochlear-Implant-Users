from src.models.dnn import CNN
from src.models.ridge import Ridge
import torch
import os
from src.data.eeg_attention_pytorch_dataset import EegAttentionDataset
from torch.utils.data import DataLoader
from torch.optim import NAdam, Adam
import numpy as np
import pickle
from collections.abc import Iterable
import operator
import functools
import optuna
import copy

# set seeds
torch.manual_seed(0)
np.random.seed(0)

# check if cuda is available
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def correlation(x, y, eps=1e-8):
    """Compute correlation between two tensors.
    """
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + eps)
    return corr

def correlation_difference(x_hat, x_attended, x_unattended):
    """
    Compute difference in correlation between attended and unattended condition.
    """
    corr_attended = correlation(x_hat, x_attended)
    corr_unattended = correlation(x_hat, x_unattended)
    return corr_attended - corr_unattended

def train_dnn(data_dir,
              subject_string_train,
              train_indices,
              subject_string_val,
              val_indices,
              ica=False,
              feature = 'env',
              checkpoint_path=None,
              model_handle=CNN,
              epochs=1,
              lr=3e-4,
              weight_decay=0.0001,
              batch_size=128,
              early_stopping_patience=torch.inf,
              optuna_trial=None,
              seed=0,
              workers=16,
              loss_fnc = 'corr',
              mdl_checkpointing=False,
              optimizer_handle=NAdam,
              use_pretrained=False,
              pretrained_model = None,
              **mdl_kwargs):
    """Train a DNN model on the EEG attention dataset.

    args:
        data_dir (str): path to the EEG attention dataset
        subject_string_train (str or list): subject identifier for training: e.g. '108' or ['108', '109']
        train_indices (list): list of trials to take for training [0,1,2,3....,19]. e.g. [0,1,2,3,4,5,6]
        subject_string_val (str or list): subject identifier for validation: e.g. '108' or ['108', '109']
        val_indices (list): list of trials to take for validation [0,1,2,3....,20]. e.g. [0,1,2,3,4,5,6]
        ica (bool, optional): whether to use ICA preprocessed EEG data. Defaults to False.
        feature (str, optional): feature to use. Defaults to 'env'. Must be either 'env' or 'onset_env'.
        checkpoint_path (str, optional): path to save model checkpoints. Defaults to None.
        window_size (int, optional): number of data points used predict a single envelope data point. Is not the same as decision window_size used for classification. Defaults to 60.
        epochs (int, optional): number of epochs to train. Defaults to 1.
        lr (float, optional): learning rate. Defaults to 3e-4.
        weight_decay (float, optional): weight decay. Defaults to 0.0001.
        batch_size (int, optional): batch size. Defaults to 128.
        early_stopping_patience (int, optional): number of epochs to wait for improvement before stopping. Defaults to torch.inf.
        optuna_trial (optuna.Trial, optional): optuna trial to report intermediate results to. Defaults to None.
        seed (int, optional): random seed. Defaults to 0.
        workers (int, optional): number of workers for data loading. Defaults to 16.
        loss_fnc (str, optional): loss function to use. Defaults to 'corr'. must bei either 'corr' or 'corr_diff'
        use_pretrained (bool, optional): whether to use pretrained model
        pretrained_model (torch model): pretrained model to use
        **mdl_kwargs: keyword arguments to pass to the model constructor
    """
    assert loss_fnc in ['corr', 'corr_diff'], f'loss_fnc must be either corr or corr_diff, but is {loss_fnc}'
    assert feature in ['env', 'onset_env'], f'feature must be either env or onset_env, but is {feature}'

    window_size = mdl_kwargs['input_length']
    if mdl_checkpointing:
        assert os.path.isdir(checkpoint_path), f"{checkpoint_path} is not a directory. Please provide valid directory."
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # configure model and optimizers
    if use_pretrained:
        mdl = copy.deepcopy(pretrained_model)
        mdl.to(device)
    else:
        mdl = model_handle(**mdl_kwargs)
        mdl.to(device)
    
    optimizer = optimizer_handle(mdl.parameters(), lr=lr, weight_decay=weight_decay)

    # configure dataloaders
    assert isinstance(train_indices, Iterable) and type(train_indices) is not dict
    assert isinstance(val_indices, Iterable) and type(val_indices) is not dict

    if loss_fnc == 'corr':
        train_dataset = EegAttentionDataset(dir_h5=data_dir, subj_string=subject_string_train, trials_indices=train_indices, window_size_training = window_size, ica=ica, feature=feature)
        train_loader_complete = DataLoader(train_dataset, batch_size=batch_size, sampler = torch.randperm(len(train_dataset)), num_workers=workers, pin_memory=True)
    
    #if loss function is correlation difference, split train indices into single and dual speaker, as single speaker does not have a competing speaker track
    elif loss_fnc == 'corr_diff':
        #get all train indices smaller than 8 (single speaker)
        train_indices_single_speaker = [index for index in train_indices if index < 8]
        #get remaining indices (dual speaker)
        train_indices_dual_speaker = [index for index in train_indices if index >= 8]
        #train dataset and train loader for single speaker
        train_dataset_single_speaker = EegAttentionDataset(dir_h5=data_dir, subj_string=subject_string_train, trials_indices=train_indices_single_speaker, window_size_training = window_size, ica=ica, feature=feature)
        train_loader_single_speaker = DataLoader(train_dataset_single_speaker, batch_size=batch_size, sampler = torch.randperm(len(train_dataset_single_speaker)), num_workers=workers, pin_memory=True)
        #train dataset and train loader for dual speaker
        train_dataset_dual_speaker = EegAttentionDataset(dir_h5=data_dir, subj_string=subject_string_train, trials_indices=train_indices_dual_speaker, window_size_training = window_size, ica=ica, feature=feature)
        train_loader_dual_speaker = DataLoader(train_dataset_dual_speaker, batch_size=batch_size, sampler = torch.randperm(len(train_dataset_dual_speaker)), num_workers=workers, pin_memory=True)

    if loss_fnc == 'corr':
        train_loader_list = [train_loader_complete]
        scenarios = ['complete']
    #single speaker cannot be trainined on correlation difference
    elif loss_fnc == 'corr_diff':
        train_loader_list = [train_loader_single_speaker, train_loader_dual_speaker]
        scenarios = ['single_speaker', 'dual_speaker']

    val_dataset = EegAttentionDataset(dir_h5=data_dir, subj_string=subject_string_val, trials_indices = val_indices, window_size_training = window_size, ica=ica, feature=feature)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, sampler = torch.randperm(len(val_dataset)), num_workers=workers, pin_memory=True)

    #train_loss_tracker = np.array([])
    for train_loader, scenario in zip(train_loader_list, scenarios):
        best_correlation=0
        best_val_acc = -0.1
        best_epoch=0
        #initialize best state dict with initial model
        best_state_dict=mdl.state_dict()
        #report loss
        val_corr_att_tracker = np.array([])
        val_corr_distr_tracker = np.array([])
        val_acc_tracker = np.array([])
        train_loss_tracker = np.array([])
        
        #training loop
        for epoch in range(epochs):
            loss_tracker_epoch = np.array([])
            if epoch > best_epoch + early_stopping_patience:
                print(f'Trained for {epoch} epochs. Early stopping training after {early_stopping_patience} epochs without improvement.')
                break

            mdl.train()

            for batch, (eeg, env_attended, env_unattended) in enumerate(train_loader):

                eeg = eeg.to(device, dtype=torch.float)
                env_attended = env_attended.to(device, dtype=torch.float)
                env_unattended = env_unattended.to(device, dtype=torch.float)

                env_attended_hat = mdl(eeg)
                if loss_fnc == 'corr' or (loss_fnc == 'corr_diff' and scenario == 'single_speaker'):
                    loss = -correlation(env_attended, env_attended_hat)
                elif loss_fnc == 'corr_diff' and scenario == 'dual_speaker':
                    loss = -correlation_difference(env_attended_hat, env_attended, env_unattended)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_tracker_epoch = np.append(loss_tracker_epoch, loss.item())

            mdl.eval()
            val_correlations_att = []
            val_correlations_dist = []
            
            n_correct = 0
            n_total = 0
            n_false = 0

            with torch.no_grad():
                for batch, (eeg, env_attended, env_unattended) in enumerate(val_loader):
                    eeg = eeg.to(device, dtype=torch.float)
                    env_attended = env_attended.to(device, dtype=torch.float)
                    env_unattended = env_unattended.to(device, dtype=torch.float)

                    env_attended_hat = mdl(eeg)
                    if scenario == 'dual_speaker':
                        val_corr_att = correlation_difference(env_attended_hat, env_attended, env_unattended)
                    else:
                        val_corr_att = correlation(env_attended, env_attended_hat)
                        val_corr_distr = correlation(env_unattended, env_attended_hat)
                        
                        if val_corr_att > val_corr_distr:
                            n_correct += 1
                        else:
                            n_false += 1
                        n_total += 1

                        val_correlations_dist.append(val_corr_distr)
                    val_correlations_att.append(val_corr_att)

            
            mean_val_correlation_att = torch.mean(torch.hstack(val_correlations_att)).item()
            if loss_fnc == 'corr':
                mean_val_correlation_dist = torch.mean(torch.hstack(val_correlations_dist)).item()

            if epoch == 0:
                train_loss_tracker = loss_tracker_epoch
            else:
                train_loss_tracker = np.vstack((train_loss_tracker, loss_tracker_epoch)) 
            val_corr_att_tracker = np.append(val_corr_att_tracker, mean_val_correlation_att)
            
            if loss_fnc == 'corr':
                val_corr_distr_tracker = np.append(val_corr_distr_tracker, mean_val_correlation_dist)
            
            val_acc = n_correct/n_total
            val_acc_tracker = np.append(val_acc_tracker, val_acc)

            print(f'Epoch: {epoch} mean validation correlation: {mean_val_correlation_att}')

            if optuna_trial is not None:
                optuna_trial.report(mean_val_correlation_att, epoch)
                if optuna_trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            # early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_state_dict = mdl.state_dict()
                
                #save new best model
                if mdl_checkpointing:
                    torch.save(
                        mdl.state_dict(),
                        os.path.join(checkpoint_path, f'epoch={epoch}_correlation={mean_val_correlation_att}.ckpt')
                        )

        training_params = {'lr': lr,
                            'weight_decay': weight_decay,
                            'batch_size': batch_size,
                            'epochs': epochs,
                            'early_stopping_patience': early_stopping_patience,
                            'seed': seed,
                            'workers': workers,
                            'loss_fnc': loss_fnc,
                            'mdl_kwargs': mdl_kwargs,
                            'model_handle': model_handle,
                            'train_indices': train_indices,}
                           
        #save loss curves and models
        if os.path.isdir(checkpoint_path):
            pickle.dump(mdl_kwargs, open(os.path.join(checkpoint_path, 'model_kwargs.pkl'), 'wb'))
            pickle.dump(training_params, open(os.path.join(checkpoint_path, 'training_params.pkl'), 'wb'))
            
            if loss_fnc == 'corr':
                np.save(os.path.join(checkpoint_path, 'train_loss.npy'), train_loss_tracker)
                np.save(os.path.join(checkpoint_path, 'val_corr_att.npy'), val_corr_att_tracker)
                np.save(os.path.join(checkpoint_path, 'val_corr_distr.npy'), val_corr_distr_tracker)
                np.save(os.path.join(checkpoint_path, 'val_acc.npy'), val_acc_tracker)
                torch.save(best_state_dict, os.path.join(checkpoint_path, 'best_model.ckpt'))
            elif loss_fnc == 'corr_diff':
                if scenario == 'single_speaker':
                    np.save(os.path.join(checkpoint_path, 'train_loss_single_speaker.npy'), train_loss_tracker)
                    np.save(os.path.join(checkpoint_path, 'val_loss_single_speaker.npy'), val_corr_att_tracker)
                    torch.save(best_state_dict, os.path.join(checkpoint_path, 'best_model_single_speaker.ckpt'))
                elif scenario == 'dual_speaker':
                    np.save(os.path.join(checkpoint_path, 'train_loss_dual_speaker.npy'), train_loss_tracker)
                    np.save(os.path.join(checkpoint_path, 'val_loss_dual_speaker.npy'), val_corr_att_tracker)
                    torch.save(best_state_dict, os.path.join(checkpoint_path, 'best_model_dual_speaker.ckpt'))

    return best_correlation, best_state_dict