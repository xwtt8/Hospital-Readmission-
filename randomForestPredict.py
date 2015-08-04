import numpy as np
import pandas as pd
import csv_io

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

import postprocessing

'''
input:
	train_data: data used to train
	test_data: data used to test
'''
def randomForestPredict(train_file, train_label_file, test_file):	
	train_data = csv_io.read_data(train_file)
	train_label = csv_io.read_data(train_label_file)
	
	train_data = np.array( [x[0:] for x in train_data] )
	train_label = np.array( [x[0] for x in train_label] )
		
	rf = RandomForestClassifier(n_estimators=100,min_samples_split=2,max_depth = 6)
	
	# Doing cross validation using 10 folds
	cv = cross_validation.KFold(len(train_data), n_folds=10)	
	i = 0
	average_importance = 0
	average_score = 0
	for traincv, testcv in cv:
		print train_data[traincv].shape, train_label[traincv].shape
		rf.fit(train_data[traincv], train_label[traincv])
		probas = rf.predict_proba(train_data[testcv])		
		print i		
		#If the prob is <0.5, the label will be 0 
		result = [x[1] for x in rf.predict_proba(train_data[testcv])]
				
		#print np.array(result)
		print rf.score(train_data[testcv],train_label[testcv])
		print rf.feature_importances_
		average_score += rf.score(train_data[testcv], train_label[testcv])
		average_importance += rf.feature_importances_[-1]

		#print train_data[testcv].shape
		#print np.array(probas).shape
		i+=1	
				
	print "average importance is " + str(average_importance)
	print "average score is " + str(average_score/i)
	# End doing cross validaiton
	
	
	# Predicting using test data
	test_data = csv_io.read_data(test_file)
	test_data = np.array( [x[0:] for x in test_data] )
	
	print np.array(test_data).shape
	
	#print test_data.shape
	
	final_prob = []
	for i in range(0,10):
		max_probs = []
		max_score = 0
		for i in range(1,20):
			rf.fit(train_data, train_label)
			predicted_probs = rf.predict_proba(test_data)
			predicted_probs = ["%f" % x[1] for x in predicted_probs]
			if max_score < rf.score(train_data, train_label):
				max_score = rf.score(train_data, train_label)
				max_prob = predicted_probs
	
		print max_score, rf.feature_importances_
		if final_prob == []:
			final_prob = max_prob
		else:
			final_prob = [float(final_prob[i])+float(max_prob[i]) for i in range(0,len(max_prob))]
			
	index = 0
	for v in final_prob:
		final_prob[index] = float(v)/10
		index+=1
	
	print final_prob
	postprocessing.getBenchmark(final_prob)


# end
randomForestPredict('train_data.csv','train_label.csv','test_data.csv')
#multiGroupsRandomForestPredict('train_data.csv', 'train_label.csv', 'test_data.csv')

