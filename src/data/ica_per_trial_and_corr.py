import mne
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import h5py
import git
from mne.preprocessing import ICA
import scipy
from scipy.signal import decimate, correlate, correlation_lags, hilbert, coherence
from scipy.stats import zscore
import json
from memory_profiler import profile


base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
montage_file = os.path.join(base_dir, "CACS-32_NO_REF_NO_CZ.bvef")
montage = mne.channels.read_custom_montage(montage_file)
raw_path = os.path.join(base_dir, "data", "raw_input")
h5_dir = os.path.join(base_dir, 'data', 'processed', 'ci_attention_artifact_removal.hdf5')
target_dir = os.path.join(base_dir, 'data', 'ica', 'corr_criterium')

# for final analyses
saving_dir = target_dir

subjects = list(range(102,115))
subjects = subjects + list(range(116,117))
subjects = subjects + list(range(118,126))
subjects = subjects + [127, 128, 130]

trials = list(range(1, 21))
fs=1000

# whether to use data where eye artifacts were previously removed --> then use ica data of that dataset
prev_eye_rm = False

# subjects = [102]
# trials = [10]

def peak_snr(correlation):
    """
    Calculating snr of correlation peak in relevant time lags (+/- 5ms)
    """
    n_samples = correlation.shape[0]
    central_lags = np.linspace(n_samples//2-int(0.005*fs), n_samples//2+int(0.005*fs), int(0.01 * fs), dtype=int).tolist()
    peak_value = np.max(correlation[central_lags])
    peak_value_index = np.argmax(correlation[central_lags]) + central_lags[0]
    signal_power = peak_value ** 2
    cross_corr_no_peak = np.delete(correlation, peak_value_index)
    noise_power = np.mean(cross_corr_no_peak**2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

#@profile
def ica_computation():
    with h5py.File(h5_dir, 'r') as f:
        for subject in subjects:
            subject_dict = {}
            subj_dir = os.path.join(target_dir, str(subject))
            subj_saving_dir = os.path.join(saving_dir, str(subject))
            if not os.path.exists(subj_dir):
                os.makedirs(subj_dir)
            if not os.path.exists(subj_saving_dir):
                os.makedirs(subj_saving_dir)
            raw_filepath = os.path.join(raw_path, str(subject), str(subject)+".vhdr")
            raw_template = mne.io.read_raw_brainvision(raw_filepath, preload=True)
            raw_template.drop_channels(['Aux1', 'Aux2'])
            raw_template.set_montage(montage, match_case=False)
            info = raw_template.info

            for trial in trials:
                peaking_channels = []
                snr_values = []
                trial_dir = os.path.join(subj_dir, str(trial))
                trial_saving_dir = os.path.join(subj_saving_dir, str(trial))
                if not os.path.exists(trial_dir):
                    os.makedirs(trial_dir)
                if not os.path.exists(trial_saving_dir):
                    os.makedirs(trial_saving_dir)
                
                if prev_eye_rm:
                    eeg_path = f'eeg_ica/{str(subject)}/{str(trial)}'
                else:
                    eeg_path = f'eeg/{str(subject)}/{str(trial)}'
                data = f[eeg_path][:]
                eeg = data[:31, :]
                audio = data[31:, :]

                raw = mne.io.RawArray(eeg, info)
                raw.set_montage(montage)

                ica_path = os.path.join(trial_dir, 'ica' + str(subject) + '_' + str(trial)+'.fif')
                ica_saving_path = os.path.join(trial_saving_dir, 'ica' + str(subject) + '_' + str(trial)+'.fif')
                
                if os.path.exists(ica_path) and os.path.isfile(ica_path):
                    ica = mne.preprocessing.read_ica(ica_path)
                else:
                    ica = ICA(n_components=30, max_iter='auto', random_state=1337, method='infomax')
                    ica.fit(raw)
                    #save ica
                    ica.save(ica_path, overwrite=True)
                fig = ica.plot_components(nrows=5, ncols=6, show=False, title = "ICA components for subject " + str(subject))
                extensions = ['.pdf', '.png', '.svg']
                for ext in extensions:
                    fig.savefig(os.path.join(trial_saving_dir, 'ica_components_' + str(subject) + '_' + str(trial) + ext))
                ica.save(ica_saving_path, overwrite=True)
                # get time series of components

                sources = ica.get_sources(raw)
                sources_np = sources.get_data()

                # calculate correlation between components and audio
                corrs = []
                lags_max_corr = []
                central_deviations_list = []
                for i in range(0,30):
                    # normalize sources before cross-correlating for comparable results
                    a, b = sources_np[i,:], audio[0,:] + audio[1,:]
                    a_norm, b_norm = a / np.linalg.norm(a), b / np.linalg.norm(b)
                    corr = correlate(a_norm, b_norm)
                    corrs.append(corr)
                    # deviation from central lag
                    central_deviation = len(corr)//2 - np.argmax(corr)
                    central_deviations_list.append(int(central_deviation))
                    lags_max_corr.append(int(np.argmax(corr)))
                    snr = peak_snr(corr)
                    snr_values.append(snr)


                n_samples = len(corrs[0])
                fig, ax = plt.subplots(5, 6, figsize=(12, 10))
                n_samples = len(corrs[0])
                lags = np.linspace(-n_samples/2, n_samples/2, n_samples)
                for i in range(30):
                    if snr_values[i] > 20:
                        color = 'red'
                        peaking_channels.append(i)
                    else:
                        color = 'blue'
                    ax[i//6, i%6].plot(lags, corrs[i], color=color)
                    ax[i//6, i%6].set_title(f'IC {i}')
                    ax[i//6, i%6].text(lags[0], 0.08, f'SNR: {int(snr_values[i])} dB', fontsize = 'small')
                    ax[i//6, i%6].grid(True)
                    ax[i//6, i%6].set_ylim(-0.1, 0.1)
                fig.suptitle(f'Correlation between ICs and audio for subject {subject} trial {trial}')
                fig.tight_layout()
                extensions = ['.pdf']
                for ext in extensions:
                    fig.savefig(os.path.join(trial_saving_dir, 'snr_pm_5ms' + str(subject) + '_' + str(trial) + ext))
                
                plt.clf()
                del fig, ax
                subject_dict[f'trial_{str(trial)}']={}
                subject_dict[f'trial_{str(trial)}']['peaking_channels'] = peaking_channels
                subject_dict[f'trial_{str(trial)}']['snr'] = snr_values
                subject_dict[f'trial_{str(trial)}']['lags_max_corr'] = lags_max_corr
                subject_dict[f'trial_{str(trial)}']['central_deviations'] = central_deviations_list
            
            # save peak channels as json
            with open(os.path.join(subj_saving_dir, f'subj{str(subject)}_SNR_analysis.json'), 'w') as f_json:
                json.dump(subject_dict, f_json)

if __name__=='__main__':
    ica_computation()