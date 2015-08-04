"""
Gaussian HMM of patients' condition
"""

import datetime
import numpy as np
import pylab as pl
from hmmlearn.hmm import GaussianHMM

'''
input:
	patient_record: a dictionary that contains date and health conditional of oragn and sympton
	date_list: a list of date in ascending order corresponding to the patient record
'''

def runHmm(patient_record,date_list,group_id,empirical_states):
###############################################################################
# Processing the data
	max_state_number = (group_id+1)*10
	
	X = np.zeros(shape=(max(len(patient_record),2),20))
	index = 0
	for date in date_list:
		tmp_list = []
		#print(date)
		for key, value in patient_record[date].iteritems():
			tmp_list.append(value)
		X[index] = np.array(tmp_list)
		index+=1
		
	# if no lab test is available, train with an all zero array
	if X.shape[0]  == 0:
		X = np.zeros(shape=(2,20))
	elif X.shape[0] == 1:
		X[1] = np.zeros(shape=(1,20))
		
	#print(X)	
	#print(X.shape)
	
###############################################################################
# Run Gaussian HMM
	print("fitting to HMM and decoding ...")
	n_components = 2
	
	# make an HMM instance and execute fit
	model = GaussianHMM(n_components, covariance_type="diag", n_iter=1000)
	
	# Train n number of HMM to avoid loacl minimal 
	max_score = 0
	max_proba_states = []
	transmat = [[]]
	n = 2
	for i in range(1,n):
		model.fit([X])
		score = model.decode(X)[0]
		if i==1 or max_score < score:
			max_score = score
			max_proba_states = model.predict(X)
			transmat = model.transmat_
		
		'''	
		print "score", score
		# predict the optimal sequence of internal hidden state
		hidden_states = model.predict(X)
		print hidden_states
		'''
	# end multiple training
	
	#print max_score, max_proba_states, transmat
	
	# Compare the state with empirical states
	max_proba_states = max_proba_states.tolist()
	max_proba_states_inver = []
	for s in max_proba_states:
		max_proba_states_inver.append(0 if s == 1 else 1)
	
	#print empirical_states, max_proba_states, max_proba_states_inver
	
	difference_state = np.subtract(np.array(max_proba_states),np.array(empirical_states)).tolist()
	difference_state_inver = np.subtract(np.array(max_proba_states_inver),np.array(empirical_states)).tolist()
	
	difference = np.sum(np.power(difference_state,2))
	difference_inver = np.sum(np.power(difference_state_inver,2))
	
	#print difference, difference_inver
	
	if(difference_inver < difference):
		max_proba_states = max_proba_states_inver
	# end switch bits
	
	# Predict future state
	future_states_proba = np.dot([0,1],transmat)
	future_state = 0
	if future_states_proba[1] > future_states_proba[0]:
		future_state = 1	
	# End
	
	result_states = max_proba_states+[future_state for i in range(0,max_state_number-len(max_proba_states))];
	
	return result_states
	'''
	state = [0,1]
	transmat = np.array(model.transmat_)
	
	print np.dot(state,transmat)
	
	print np.array(model.transmat_)
	
	#print (hidden_states)
	#print (hidden_states.shape)
	'''
		
	print("done\n")