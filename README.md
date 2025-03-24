model_dev_aad_semeco
==============================

Attention decoding for subjects with hearing impairment,thus wearing CIs or hearing aids.
The aim of the development process is a model that is deployable in hardware while reaching stable and high decoding accuracies.

Getting started
------------

**Setting up the environment**

I recommend using a virtual conda environment for the project. [Here's a tutorial](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/)

Under /conda_env you find .yml files for Linux and Linux with which you you can [create an enviroÇnment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

**Installing src as module**

Run `python setup.py install` in the root directory

**Download hdf5 database file**


Preprocessed data can be downloaded from [zenode](https://zenodo.org/records/10980117)


Project Organization
------------

    ├── LICENSE
    ├── README.md      
    ├── data
    │
    ├── docs          
    |   |──  hdf5_dataset_info.txt   <- description of the structure of the database file.
    │
    │
    ├── notebooks     
    │
    │
    ├── reports            
    │   └── figures        <- Generated graphics and figures to be used in reporting. 
    │                            Please use subfolders for the different models.
    │
    ├── conda_env          <- Files for setting up the conda environment. 
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data   
    │   ├── evaluation  
    │   ├── models
    │   ├── training



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
