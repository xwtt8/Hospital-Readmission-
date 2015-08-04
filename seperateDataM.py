# Convert train.csv into usable form. Outputs csv file including only numbers.
# Converts every field into a number ({0,1}, int, float).
# Author: Xiaotian Wang
#
# Numeric conversion rules:
# gender: M -> 0, F -> 1
# admission_type: Planned -> 0, Emergency -> 1, Other Hospital -> 2, Other -> 3
# outcome: Home -> 0, Care Home -> 1, Other Hospital -> 2, Died ->3, Other -> 4

from datetime import datetime 
import numpy as np

import testprocessing
import unsplearnHmm
import multiFeaturesHmm
import Clustering

# Maps attributes to row positions
# TODO: figure out lab tests
field = {'label':0, 'admission_id':1, 'patient_id':2, 'age':3, 'gender':4, \
'admission_type':5, 'outcome':6, 'start_date':7,'discharge_date':8, \
'num_prior_readmissions':9}

# List of fields we are currently storing
# TODO: add all useful fields
use_field = ['age', 'gender','admission_type','outcome','start_date',\
'discharge_date', 'num_prior_readmissions']

# List of symptoms that can be reflected by lab tests
symptom_list = ['inflammation','cardiovascular','diabetes','anemia','polycythemia',\
		   'hypochloremia','hyperchloremia','leukopenia','leukocytosis']

# List of organ whose's condition can be reflected by lab tests
organ_list = ['kidney','protein','WBC','heart','liver','bone',\
		 'pancreas','muscle','RBC','nutrition','lung']

# The max number of sequence of states for patient
max_states_number = 79
num_conditions = len(symptom_list)+len(organ_list)

# The group is decided by the number of days a patient has lab test
group_id = 0

# Key is the index of the admission and value is the number of days tests are performed
tests_number = {}

# Key is the group id and value index
groups = {}

# Key is the index and value is the label(0 or 1)
train_labels = {}

# Key is the index and value is the admission_id
admit_id = {}

# list of train data. This can be seen as a matrix whoes row is the train data for each patient
train_data = []

# list of test data. This can be seen as a matrix whoes row is the test data for each patient
test_data = []

# list of state sequences. 
state_sequences = []

"""
	Converts input line into all numbers and writes to a new line in writer
	Input:  
			classification      the class for a particular patient's health sequence
			statewriter csv file to write patient's state to
			test_data   a list with patient's lab test value
			diag_dict   a dictionary with organ or symptom as key and normal range as value
			row         a list of strings
			datawriter  csv file to write data to
			labelwriter csv file to write label to 
			infowriter  csv file to write information that doesn't predict result
			train       true for processing train data (starting with label);
						false for test data
	Output: none
	Effect: Writes a new line to the writer if successfully processed
"""

def convert_line(group_id, index, latest_test_data, diag_dict, row, datawriter, labelwriter, infowriter, train):
	datarow = []
	start_column = 0
	if train: # write label
		#labelwriter.writerow(row[field.get('label')])
		start_column = 10
	else: # writer info
		#infowriter.writerow([row[field.get('admission_id')-1]])
		start_column = 9

	age_index = 0 # 0 for adult men, 1 for adult women, and 2 for children

	# write other useful fields listed in use_field
	for i in use_field:
		if train:
			thing = row[field.get(i)]
			train_writer = []
		else:
			thing = row[field.get(i)-1]
			test_writer = []
			
		if i =='age':
			#datarow.append(thing)
			if(int(thing)<18):
				age_index = 2

		if i == 'gender':
			datarow.append(0) if thing =='M' else datarow.append(1)
			if(age_index != 3):
				if thing =='M':
					age_index = 0
				else:
					age_index = 1

		elif i == 'admission_type':
			if thing == 'Planned': 
				datarow.append(0)
			elif thing == 'Emergency':
				datarow.append(1)
			elif thing == 'Other Hospital':
				datarow.append(2)
			elif thing == 'Other': 
				datarow.append(3)
			else: 
				print 'illegal admission_type', thing
		elif i == 'outcome':
			if thing == 'Home':
				datarow.append(0)
			elif thing == 'Care Home':
				datarow.append(1)
			elif thing == 'Other Hosp':
				datarow.append(2)
			elif thing =='Died':
				datarow.append(3)
			elif thing == 'Other':
				datarow.append(4)
			else:
				print 'illegal outcome', thing
		elif i == 'start_date' or i == 'discharge_date':
			start_date = datetime.strptime(thing, "%Y-%m-%d")
			# print start_data.toordinal()

		elif i == 'discharge_date':
			finish_date = datetime.strptime(thing, "%Y-%m-%d")
			# print finish_date.toordinal()
		else:
			datarow.append(thing)

	# write in additional data from computation
	# write total days in hospital
	if train:
		d0 = datetime.strptime(row[field.get('start_date')], "%Y-%m-%d")
		d1 = datetime.strptime(row[field.get('discharge_date')], "%Y-%m-%d")
	else:
		d0 = datetime.strptime(row[field.get('start_date')-1], "%Y-%m-%d")
		d1 = datetime.strptime(row[field.get('discharge_date')-1], "%Y-%m-%d")
	datarow.append((d1-d0).days)


	# Start handling the lab test data
	# Create an dictionary that use the symptom and organ as key to indicate the health condition
	# the health condition are repersent by float 
	condition_dict = {} # This dictionary is used for copy
	for value in (symptom_list+organ_list):
	   condition_dict[value] = 0.0

	# This dictionary use the date of tests as key and the dictionary of the health 
	# condition of organs and symptom as value for each patient
	patient_records = {}
	
	# This dictionary is used to store the different between the lab test result and 
	# the normal lab test result for a patient in different date
	compared_records = {}
	
	current_date = 0
	i = start_column

	######################
	# Get the total number of dates the when patient took tests
	# Because the test data is not in sequence order, we need to reorder it into sequence order
	date_dict = {}
	while(i<len(row)):
		date = row[i+1].split()[0]
		if date not in patient_records:
			date_dict[date] = 0
		i+=3

	# use list to sort date into ascending order
	date_list = []
	for key,value in date_dict.iteritems():
		date_list.append(key)
	date_list.sort();
	
	# Add the num of date into a dictionary for future
	#tests_number[index] = len(date_list)
	current_id = len(date_list)/10
	
	if len(date_list)/10 in groups:
		groups[len(date_list)/10].append(index)
	else:
		groups[len(date_list)/10] = [index]
	tests_number[index] = len(date_list)
	
	# Only write the label from the patients is in group 2
	if current_id == group_id:
		if train: # write label
			train_writer.append(row[field.get('label')])
			#labelwriter.writerow(row[field.get('label')])
			
		else: # writer info
			test_writer.append(row[field.get('admission_id')-1])
			#infowriter.writerow([row[field.get('admission_id')-1]])
		
	# create patient_record dictionary based on date
	for d in date_list:
		patient_records[d] = condition_dict.copy()
		compared_records[d] = condition_dict.copy()
	# end reordering date and initial patient_records with date
	
	######################
	i = start_column
	while(i<len(row)):
		lab_test = row[i]
		current_date = row[i+1].split()[0]
		test_result = float(row[i+2])

		if diag_dict[lab_test] != []:
		# Check whether the test result is below, above, or within the normal range
			#print diag_dict[lab_test]
			normal_range = diag_dict[lab_test][age_index]
			if test_result < float(normal_range[0]):
				if diag_dict[lab_test][3] != '':
					for test in diag_dict[lab_test][3]:
						patient_records[current_date][test] += float(test_result)
						compared_records[current_date][test] += (float(normal_range[0])-test_result)
																#/float(normal_range[0])
			elif test_result > float(normal_range[1]):
				#print diag_dict[lab_test]
				if diag_dict[lab_test][4] !='':
					for test in diag_dict[lab_test][4]:
						patient_records[current_date][test] += float(test_result)
						compared_records[current_date][test] += (test_result-float(normal_range[1]))
																#/float(normal_range[1])
		i+=3
	
	# states in a list that we used to compared with the state generated by HMM to 
	# tell whether the states have flipped
	empirical_states = {} # This dictionary is used for copy
	for value in (symptom_list+organ_list):
	   empirical_states[value] = []
	
	for d in date_list:
		score = 0
		for key,value in compared_records[d].iteritems():
			if(value == 0):
				empirical_states[key].append(0)
			else:
				empirical_states[key].append(1)
	
	# handle the situation when the number of states are not enough
	if date_list == []:
		for value in (symptom_list+organ_list):
			empirical_states[value] = [0,0]
	if len(date_list) == 1:
		for value in (symptom_list+organ_list):
			empirical_states[value].append(0)
	
	######################
	# Return one sequence of state based on all twenty features
	'''
	if(group_id == current_id):
		states = unsplearnHmm.runHmm(patient_records,date_list,group_id,empirical_states)		
		state_sequences.append(states)
	'''			
	# Return twenty sequence of state for every feature 
	state_dict = {}
	if(group_id == current_id):
		print("fitting to HMM and decoding ...")
		state_dict = multiFeaturesHmm.runHmm(patient_records, date_list,group_id,empirical_states)
	######################

	#append the latest test data to the patient info
	if date_list == []:
		datarow+= [0 for i in range(0,20)]
	else:
		for key,value in compared_records[date_list[-1]].iteritems():
			datarow.append(value)
	
	#append all the data to train/test writer
	print current_id, group_id
	if current_id == group_id:
		if train: # write label
			train_writer += datarow
			train_data.append(train_writer)
		else: # writer info
			test_writer += datarow
			test_data.append(test_writer)
		#datawriter.writerow(datarow)
	
	return state_dict
	
# Process the data.
# This would overwrite files if those files existed!
import csv
proc_train = True
start_col = 10;
if proc_train:
	rfile = open('train.csv','rb')
	dfile = open('train_data.csv','wb')
	lfile = open('train_label.csv','wb')
	ifile = open('train_info.csv', 'wb')
	sfile = open('train_states.csv','wb')
	cfile = open('train_classes.csv','wb')
	state_file = 'train_states.csv'	
	
else:
	rfile = open('test.csv','rb')
	dfile = open('test_data.csv','wb')
	lfile = open('test_label.csv','wb')
	ifile = open('test_info.csv', 'wb')
	sfile = open('test_states.csv','wb')
	cfile = open('test_classes.csv','wb')
	state_file = 'test_states.csv'

try:
	reader = csv.reader(rfile, delimiter=',')
	datawriter = csv.writer(dfile, delimiter=',')
	labelwriter = csv.writer(lfile, delimiter=',')
	infowriter = csv.writer(ifile, delimiter=',')
	classwriter = csv.writer(cfile,delimiter=',')
	statewriter = csv.writer(sfile, delimiter=',')
	
	lcount = 1 #line count

	# get test name into a dictionary
	test_dict = testprocessing.get_test(reader,start_col)
	rfile.seek(0)
	# get diagnostic info into a dictionary
	diag_dict = testprocessing.read_labtestcvs("LabTestsInfo.csv", test_dict)
	# get the latest lab test result for every patient
	latest_test_data = testprocessing.convert_test_line(reader,test_dict,start_col)
	rfile.seek(0)

	# Get the class label for each state with respect to patient's health states
	#classification = Clustering.runKmeans(state_file)

	# append a empty row to the first row
	infowriter.writerow([])
	datawriter.writerow([])
	labelwriter.writerow([])
	statewriter.writerow([])
	classwriter.writerow([])
	
	while group_id < 8:
		#number of samples from each group
		sample_num = 0
		# Dictionary that hold the hmm state sequences
		state_dict = {}
		for s in symptom_list:
			state_dict[s] = []			
		for o in organ_list:
			state_dict[o] = []
			
		for index, row in enumerate(reader):
			if proc_train:
				states = convert_line(group_id, index,latest_test_data[index], diag_dict, row, datawriter, labelwriter, infowriter, True)
				'''
				if index >10:
					break
				'''
				if states != []:
					sample_num +=1
					# State_dict will store all the patients' record on each organ and symptom
					# This dictionary will be used to do clustering later
					for key, value in states.iteritems():
						state_dict[key].append(value)
			else:
				states = convert_line(group_id, index, latest_test_data[index], diag_dict, row, datawriter, labelwriter, infowriter, False)
				'''
				if index >10:
					break	
				'''
				if states != []:
					sample_num +=1
					# The same as the train hmm states
					for key, value in states.iteritems():
						state_dict[key].append(value)
						
			print 'converted line', lcount
			lcount += 1
		
		class_label = []
		if(sample_num > 4):
		# Run kmeans clustering and write the data into csv (patient number, symptom number)
			print('running Kmeans clustering algorithm')
			for key,value in state_dict.iteritems():
				class_label.append(Clustering._runKmeans(value))

			class_label = np.array(class_label).transpose().tolist()
			
			index = 0
			for cl in class_label:
				class_label[index] = [c+10*group_id for c in cl]
				index+=1
		else:
			for i in range(0,sample_num):
				class_label.append([0 for j in range(0,20)])
				
		# write data into csv file
		index = 0
		for c in class_label:
			if proc_train == True:
				print index
				print np.array(train_data).shape
				labelwriter.writerow(train_data[index][0])
				datawriter.writerow(train_data[index][1:]+c)
			else:
				print np.array(test_data).shape
				infowriter.writerow([test_data[index][0]])
				datawriter.writerow(test_data[index][1:]+c)
			index+=1
		
		group_id += 1 
		# initial variables
		lcount = 1
		train_data = []
		test_data = []
		state_sequences = []
		rfile.seek(0)	
	'''
	for key, value in groups.iteritems():
		print key,len(value)
	'''
finally:
	rfile.close()
	dfile.close()
	lfile.close()
