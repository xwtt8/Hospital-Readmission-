# Gradient Boosting classifier 
import numpy as np
import csv_io
import postprocessing

from sklearn import ensemble
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from sklearn.grid_search import GridSearchCV


def runGradientBoosting(train_file, train_label_file, test_file, test_label_file):	
##############################################	
# Load Data
	train_data = csv_io.read_data(train_file)
	train_label = csv_io.read_data(train_label_file)
	
	train_data = np.array( [x[0:] for x in train_data] )
	train_label = np.array( [x[0] for x in train_label] )
##############################################
# Fit regression model
	params = {'n_estimators': 50, 'max_depth': 10, 'min_samples_split': 2,
			'learning_rate': 0.15}
	rf = ensemble.GradientBoostingRegressor(**params)
	
	# Doing cross validation using 10 folds
	cv = cross_validation.KFold(len(train_data), n_folds=3)
	average_importance = 0
	average_score = 0
	for traincv, testcv in cv:
		print train_data[traincv].shape, train_label[traincv].shape
		
		rf.fit(train_data[traincv], train_label[traincv])
		
		average_score += rf.score(train_data[testcv],train_label[testcv])
		#print rf.predict(train_data[testcv])
		print rf.feature_importances_
		
		mse = mean_squared_error(train_label[testcv], rf.predict(train_data[testcv]))
		mse_train = mean_squared_error(train_label[traincv], rf.predict(train_data[traincv]))
		print("MSE: %.4f" % mse)
		print("MSE: %.4f" % mse_train)
	
	print "average score is"+str(average_score/10)
	# End doing cross validaiton

# Predicting using test data
	test_data = csv_io.read_data(test_file)
	test_data = np.array( [x[0:] for x in test_data] )
	
	#test_label = csv_io.read_data(test_label_file)
	#test_label = np.array( x[1] for x in test_label )
	
	max_probs = []
	min_mse = 1
	for i in range(1,20):
		rf.fit(train_data, train_label)
		predicted_probs = rf.predict_proba(test_data)
		mse_train = mean_squared_error(train_label, rf.predict(train_data))
		if min_mse > mse_train:
			min_mse = mse_train
			max_probs = predicted_probs
			
	print np.array(max_probs).shape
	#print mean_squared_error(test_label, max_probs)
	postprocessing.getBenchmark(max_probs)

	
runGradientBoosting('train_data.csv','train_label.csv', 'test_data.csv', 'benchMark2.csv');