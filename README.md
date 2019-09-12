# Repository for PID studies with ML

 branch_matteo
#Data samples:
Data: LHC17pq_cent
MC: LHC17l3b_cent
MC: LHC18a5b_cent

#plot
hist2d to represent data, remember to put directory,'--mc' for MC files and '--data' for data files

#add_colum
function to add a new column of the dataframe which it will be used later for training and testing

## Production of data frames
* Run PID task on grid with RunAnalysisAODVertexingHFPIDsyst.C passing a yaml configuration file for your desired data sample
* Run getdataframesfromroot.py script to convert trees to pandas dataframe and store them in files:  
``` python3 getdataframesfromroot.py input_root_file name_TDirectoryFile output_directory ```

## Basic examples
* Binary classification with BTD (using [interpret](https://github.com/microsoft/interpret) library): test_BDTinterpret.py
* Multi-class classification with SVM using the [scikit-learn](https://scikit-learn.org/stable/modules/svm.html) library and with the [xgboost](https://xgboost.readthedocs.io/en/latest/) library: test_multiclass.py
* One vs. rest classification with [xgboost](https://xgboost.readthedocs.io/en/latest/) classifier using the [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html) library for the one vs. rest classification: test_BDTOneVsRest.py

## Data samples:
* Data: LHC17pq_cent
* MC (general purpose): LHC17l3b_cent
* MC (with injected nuclei): LHC18b5a_cent

## Setup git
* ``` git config --global user.name "<Firstname> <Lastname>" ```
* ``` git config --global user.email <your-email-address> ``` 
* ``` git config --global user.github <your-github-username> ``` 
master
