import numpy as np
import csv_io

from sklearn.cluster import KMeans
from sklearn.mixture import GMM

'''
input a file name in string
'''
def runKmeans(data_file):
	train_data = csv_io.read_data(data_file)
	print len(train_data)
	num_clusters = 10
	model = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
	
	max_score = 0
	iteration = 2
	best_classification = []
	for i in range(1,iteration): 
		print "Iteration number "+str(i)
		model.fit(train_data)
		score = model.score(train_data)
		
		if i == 1 or score > max_score:
			max_score = score
			best_classification =  model.predict(train_data)
	
	print len(best_classification.tolist())
	return best_classification.tolist()


'''
input a list of train data
'''
def _runKmeans(train_data):
	#print ('Running Kmeans Clustering...')
	num_clusters = 5
	model = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
	
	max_score = 0
	iteration = 20
	best_classification = []
	for i in range(1,iteration): 
	#	print "Iteration number "+str(i)
		model.fit(train_data)
		score = model.score(train_data)
		
		if i == 1 or score > max_score:
			max_score = score
			best_classification =  model.predict(train_data)
	
	#print ("Done!")
	return best_classification.tolist()
	
def _runGMM(train_data):
	print ('Running GMM...')
	g = GMM(covariance_type='diag', init_params='wmc', min_covar=0.001,
					n_components=10, n_init=1, n_iter=100, params='wmc',
					random_state=None, thresh=None, tol=0.001)
	g.fit(train_data)
	classification = g.predict(train_data)
	print classification		
	
	return classification


#runKmeans('train_states.csv')