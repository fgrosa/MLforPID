# Repository for PID studies with ML

## Production of data frames
* Run PID task on grid with RunAnalysisAODVertexingHFPIDsyst.C passing a yaml configuration file for your desired data sample
* Run getdataframesfromroot.py script to convert trees to pandas dataframe and store them in files:  
```py
python3 getdataframesfromroot.py input_root_file name_TDirectoryFile output_directory 
```

## Basic examples
* Binary classification with BTD (using [interpret](https://github.com/microsoft/interpret) library): test_BDTinterpret.py
* Multi-class classification with SVM using the [scikit-learn](https://scikit-learn.org/stable/modules/svm.html) library and with the [xgboost](https://xgboost.readthedocs.io/en/latest/) library: test_multiclass.py
* One vs. rest classification with [xgboost](https://xgboost.readthedocs.io/en/latest/) classifier using the [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html) library for the one vs. rest classification: test_BDTOneVsRest.py

## Plots
* dE/dx vs. p representation in 2D plots (both data or MC):
```
python3 plot_hist2d.py --mc (--data) name_directory_with_files 
```
where ```name_directory_with_files``` is a directory that should contain the data or MC files

* plot purity of tagged samples (from MC truth only):
```
python3 plot_purity.py name_directory_with_files_MC 
```
where ```name_directory_with_files_MC``` is a directory that should contain the MC files

## Grid search:
* Grid search for one vs. rest classifier with [xgboost](https://xgboost.readthedocs.io/en/latest/) binary classifiers:
```
python3 grid_search_OvR.py name_directory_with_files
```
* Grid search for multi-class classifier with [xgboost](https://xgboost.readthedocs.io/en/latest/):
```
python3 grid_search_multiclass.py name_directory_with_files
```
in both cases ```name_directory_with_files``` is a directory that should contain the files for the grid search (either data or MC)

## Training and testing:
* Training and testing for one vs. rest classifier with [xgboost](https://xgboost.readthedocs.io/en/latest/) binary classifiers:
```
python3 OvsR_classifier_train_test.py name_directory_with_files
```
where ```name_directory_with_files``` is a directory that should contain the data or MC files

## Data samples:
* Data: LHC17pq_cent
* MC (general purpose): LHC17l3b_cent
* MC (with injected nuclei): LHC18b5a_cent

## Setup git
* ``` git config --global user.name "<Firstname> <Lastname>" ```
* ``` git config --global user.email <your-email-address> ``` 
* ``` git config --global user.github <your-github-username> ``` 