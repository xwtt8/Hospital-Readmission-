# Hospital-Readmission-
This is the code for solving hospital readmission problem using machine learning tools

The python version I use is 2.76
Need to install hmmlearn from github https://github.com/hmmlearn/hmmlearn
Need to install python sklearn to train model

In a linux terminal, change directory to the Term_project_deliver
The step that produce the best prediction is:

Processing training data
Set proc_train = True and start_col = 10 in both hmmporcessing.py and preprocessing.py
1. Use HMM to get hidden states sequences by command “python hmm processing.py”
2. Doing preprocessing and generate features by command “python preprocessing.py”

Processing testing data
Set proc_train = False and start_col = 9 in both hmmporcessing.py and preprocessing.py
2. change the pro_train to false
2. python Gboosting.py

Run ensemble model to get benchmark for gaggle submission
You can either run “python randomforestPredict.py” or “python Boosting.py”.
The resulting prediction is in the file “MyBenchmark.csv”

There are also an existing “MyBenchmark” which will achieve 68% AUC on Kaggle