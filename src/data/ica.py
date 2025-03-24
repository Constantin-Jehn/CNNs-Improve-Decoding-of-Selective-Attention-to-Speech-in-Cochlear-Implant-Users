import mne
import os
from mne.preprocessing import ICA
import git
import h5py
import numpy as np


subjects = list(range(301,329))

work_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
etard_data_dir = "/Users/constantin/PhD_Data/EtardBrainstemAndComprehension"
ci_data_dir = os.path.join(work_dir, "data/processed/ci_final_l_1,1_h_32,50_out_125_130_incl_ica.hdf5")



raw_path = os.path.join(work_dir, "data", "raw_input")
montage_file = os.path.join(work_dir, "CACS-32_NO_REF_NO_CZ.bvef")
figure_filepath = os.path.join(work_dir, "figures")
sphere=(0, 0, 0.035, 0.094)

def create_directory(work_dir, *args):
    """ Create a directory if it doesn't exist """

    path = os.path.join(work_dir, *args)

    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created.")

    return path

def filepath_etard(subject):

    header_paths = []
    data_dir_subject = os.path.join(etard_data_dir, "eeg", "YH" + f"{subject:02}")

    files = [f for f in os.listdir(data_dir_subject) if os.path.isfile(os.path.join(data_dir_subject, f))]
    for file in files:
        if file.endswith(".vhdr"):
            header_paths.append(os.path.join(data_dir_subject, file))
    return header_paths

def ica_CI_preprocessed():
    """
    Do ICA on preprocessed data (re-referenced, filtered, downsampled)
    """
    montage = mne.channels.read_custom_montage(montage_file)
    with h5py.File(ci_data_dir, 'r') as f:
        for subject in subjects:
            for trial in range(1,21):
                eeg_path = f'eeg/{str(subject)}/{str(trial)}'
                if trial == 1:
                    subject_eeg_data = f[eeg_path][:31,:]
                else:
                    subject_eeg_data = np.concatenate((subject_eeg_data, f[eeg_path][:31,:]), axis=1)

            #to get accurate info
            raw_filepath = os.path.join(raw_path, str(subject), str(subject)+".vhdr")
            raw_template = mne.io.read_raw_brainvision(raw_filepath, preload=True)
            raw_template.drop_channels(['Aux1', 'Aux2'])
            raw_template.set_montage(montage, match_case=False)
            info = raw_template.info

            raw = mne.io.RawArray(subject_eeg_data, info)
            raw.set_montage(montage, match_case=False)

            # Prepare and fit ICA
            filt_raw = raw.copy().filter(l_freq=1.0, h_freq=100.0)
            # Using n_components=31 (resulting in n_components_=31) may lead to an unstable mixing matrix estimation 
            # because the ratio between the largest (0.0057) and smallest (5.4e-09) variances is too large (> 1e6); consider setting n_components=0.999999 or an integer <= 30
            ica = ICA(n_components=30, max_iter='auto', random_state=1337, 
                        method='fastica')
            ica.fit(filt_raw)

            solution_filepath = create_directory(work_dir, 'data', 'ica', 're_ref', str(subject))
            #create directory if not exists
            if not os.path.exists(solution_filepath):
                os.makedirs(solution_filepath)
                print(f"Directory '{solution_filepath}' created.")
            ica.save(os.path.join(solution_filepath, str(subject) + 'ica'), overwrite=True)

            # Plot components
            #figure_filepath = create_directory(work_dir, 'figures', str(subject), 'ica')
            fig = ica.plot_components(sphere=sphere, nrows=5, ncols=6, show=False, title = "ICA components for subject " + str(subject))
            # for i in range(0, len(fig)):
            #     fig[i].savefig(os.path.join(solution_filepath, str(subject) + "component_%02d" % i))
            extensions = ['.pdf', '.svg', '.png']
            for ext in extensions:
                fig.savefig(os.path.join(solution_filepath, str(subject) + ext))
            print(f"ICA finished for subject {subject}.")

            subject_eeg_data = 0


def ica_CI():
    #on raw data
    for subject in subjects:

        raw_filepath = os.path.join(raw_path, str(subject), str(subject)+".vhdr")
        raw = mne.io.read_raw_brainvision(raw_filepath, preload=True)
        #raw.drop_channels(['Aux1', 'Aux2'])
        montage = mne.channels.read_custom_montage(montage_file)
        raw.set_montage(montage, match_case=False)

        # Prepare and fit ICA
        filt_raw = raw.copy().filter(l_freq=1.0, h_freq=100.0)
        # Using n_components=31 (resulting in n_components_=31) may lead to an unstable mixing matrix estimation 
        # because the ratio between the largest (0.0057) and smallest (5.4e-09) variances is too large (> 1e6); consider setting n_components=0.999999 or an integer <= 30
        ica = ICA(n_components=30, max_iter='auto', random_state=1337, 
                    method='fastica')
        ica.fit(filt_raw)
        solution_filepath = create_directory(work_dir, 'data', 'ica','preprocessed', str(subject))
        ica.save(os.path.join(solution_filepath, "ica"), overwrite=True)

        # Plot components
        figure_filepath = create_directory(work_dir, 'figures', str(subject), 'ica')
        fig = ica.plot_components(sphere=sphere, nrows=5, ncols=6, show=False, title = "ICA components for subject " + str(subject))
        fig.savefig(figure_filepath+"/ica_components" )
        print(f"ICA finished for subject {subject}.")

def ica_control_group():
    #on raw data
    for subject in subjects:
        #load data
        raw_filepath = os.path.join(raw_path, str(subject), str(subject)+".vhdr")
        raw = mne.io.read_raw_brainvision(raw_filepath, preload=True)
        raw.drop_channels(['Aux1', 'Aux2'])
        montage = mne.channels.read_custom_montage(montage_file)
        raw.set_montage(montage, match_case=False)

        # lof algorithm needs data to be filtered between 1 and 32 Hz
        # see e.g. https://doi.org/10.3390/s22197314
        filt_raw_lof = raw.copy().filter(l_freq=1.0, h_freq=32.0)
        
        # ic label algorithm needs data to be filtered between 1 and 100 Hz 
        # see https://doi.org/https://doi.org/10.1016/j.neuroimage.2019.05.026
        filt_raw_ica = raw.copy().filter(l_freq=1.0, h_freq=100.0)

        # Identify bad channels using the lof algorithm
        bad_channels, scores_subj = mne.preprocessing.find_bad_channels_lof(filt_raw_lof, n_neighbors = 10, threshold=1.6, return_scores= True)
        
        # Prepare and fit ICA
        filt_raw_ica.drop_channels(bad_channels)
        # ic label algorithm is evaluated for infomax 
        ica = ICA(n_components=20, max_iter='auto', random_state=1337, 
                    method="infomax")
        ica.fit(filt_raw_ica)

        solution_filepath = create_directory(work_dir, 'data', 'ica','preprocessed', str(subject))
        ica.save(os.path.join(solution_filepath, "ica"), overwrite=True)

        # Plot components
        figure_filepath = create_directory(work_dir, 'figures', str(subject), 'ica')
        fig = ica.plot_components(sphere=sphere, nrows=5, ncols=6, show=False, title = "ICA components for subject " + str(subject))
        fig.savefig(figure_filepath+"/ica_components" )
        print(f"ICA finished for subject {subject}.")

def ica_Etard():
    """
    ICA for Etard data as comparison
    """

    print("ICA for Etard data")
    subjects = list(range(6,20))
    for subject in subjects:

        raw_filepath_list = filepath_etard(subject)

        raws = [ mne.io.read_raw_brainvision(raw_filepath, preload=True) for raw_filepath in raw_filepath_list]
        for raw in raws:
            raw.drop_channels(['Sound'])
        raw = mne.concatenate_raws(raws)
        #raw.drop_channels(['Aux1', 'Aux2'])
        #montage = mne.channels.read_custom_montage(montage_file)
        #raw.set_montage(montage, match_case=False)

        # Prepare and fit ICA
        filt_raw = raw.copy().filter(l_freq=1.0, h_freq=100.0)
        # Using n_components=31 (resulting in n_components_=31) may lead to an unstable mixing matrix estimation 
        # because the ratio between the largest (0.0057) and smallest (5.4e-09) variances is too large (> 1e6); consider setting n_components=0.999999 or an integer <= 30
        ica = ICA(n_components=30, max_iter='auto', random_state=1337, 
                    method='fastica')
        ica.fit(filt_raw)

        solution_filepath = create_directory(work_dir, 'data', 'etard_ica', 'processed', str(subject))
        
        #chech if solution_filepath exists and create otherwise
        if not os.path.exists(solution_filepath):
            os.makedirs(solution_filepath)
            print(f"Directory '{solution_filepath}' created.")

        ica.save(os.path.join(solution_filepath, "ica"))

        # Plot components
        #figure_filepath = create_directory(work_dir, 'figures', 'ica', str(subject), 'ica')
        
        fig = ica.plot_components(sphere=sphere, show=False)
        for i in range(0, len(fig)):
            fig[i].savefig(solution_filepath + "/component_%02d" % i)
        
        print(f"ICA finished for subject {subject}.")

if __name__ == "__main__":
    #ica_Etard()
    #ica_Etard()
    #ica_CI_preprocessed()
    #ica_CI()
    ica_control_group()
