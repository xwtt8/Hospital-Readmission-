# AdaBoost Algorithm
import numpy as np
import csv_io
import postprocessing

from sklearn import ensemble
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error


def runAdaBoost(train_file, train_label_file, test_file):	
##############################################	
# Load Data
	train_data = csv_io.read_data(train_file)
	train_label = csv_io.read_data(train_label_file)

	train_data = np.array( [x[0:] for x in train_data] )
	train_label = np.array( [x[0] for x in train_label] )


##############################################
# Fit regression model
	#params = {DecisionTreeClassifier(max_depth=1), 'algorithm':"SAMME", 'n_estimators': 100, 'learning_rate': 1.0}
	#rf = ensemble.AdaBoostClassifier(**params)
	
	rf = ensemble.AdaBoostClassifier(ensemble.RandomForestClassifier(min_samples_split=2,max_depth = 5),
						 algorithm="SAMME.R",
						 n_estimators=100)
	
	# Doing cross validation using 10 folds
	cv = cross_validation.KFold(len(train_data), n_folds=10)
	average_importance = 0
	average_score = 0
	for traincv, testcv in cv:
		print train_data[traincv].shape, train_label[traincv].shape

		rf.fit(train_data[traincv], train_label[traincv])

		average_score += rf.score(train_data[testcv],train_label[testcv])
		#print rf.predict(train_data[testcv])
		print rf.feature_importances_

		mse = mean_squared_error(train_label[testcv], rf.predict(train_data[testcv]))
		print("MSE: %.4f" % mse)

	print "average score is"+str(average_score/10)
	# End doing cross validaiton
	
	
runAdaBoost('train_data.csv','train_label.csv', 'test_data.csv');
