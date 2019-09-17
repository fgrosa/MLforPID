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
```py
python3 plot_hist2d.py --mc (--data) name_directory_with_files 
```

## Data samples:
* Data: LHC17pq_cent
* MC (general purpose): LHC17l3b_cent
* MC (with injected nuclei): LHC18b5a_cent

## Setup git
* ``` git config --global user.name "<Firstname> <Lastname>" ```
* ``` git config --global user.email <your-email-address> ``` 
* ``` git config --global user.github <your-github-username> ``` 
master
