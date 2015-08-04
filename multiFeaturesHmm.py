'''
Gaussian HMM of patients' conditions
Create one sequence of states for every symptom and organs per patient
'''

import numpy as np
import pylab as pl
from hmmlearn.hmm import GaussianHMM

# List of symptons that can be reflected by lab tests
symptom_list = ['inflammation','cardiovascular','diabetes','anemia','polycythemia',\
		   'hypochloremia','hyperchloremia','leukopenia','leukocytosis']

# List of organ whose's condition can be reflected by lab tests
organ_list = ['kidney','protein','WBC','heart','liver','bone',\
		 	'pancreas','muscle','RBC','nutrition','lung']
		 
def runHmm(patient_record,date_list,group_id,empirical_states):
###############################################################################
# Processing the data
	result = {}
	state_dict = {}
	
	for s in symptom_list:
		state_dict[s] = np.zeros(shape=(max(len(patient_record),2),1))
		result[s] = []
		
	for o in organ_list:
		state_dict[o] = np.zeros(shape=(max(len(patient_record),2),1))
		result[s] = []
	
	index = 0
	for date in date_list:
		for key, value in patient_record[date].iteritems():
			state_dict[key][index] = np.array([value])
		index+=1
	
	print("fitting to HMM and decoding ...")
	for s in symptom_list+organ_list:
		result[s] = predict_states(state_dict[s],group_id,empirical_states[s])
		
	print("done\n")
	return result
	

###############################################################################
# Predict Gaussian HMM
def predict_states(X,group_id,empirical_states):
	#print("fitting to HMM and decoding ...")
	max_state_number = (group_id+1)*10
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
	print("done\n")