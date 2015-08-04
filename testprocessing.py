#
# Description: Convert the every type of lab test data into test value and number of test pair.
# Rules: test value is the average value among different tests.
# Usage: call get_test() to get a dictionary which contains the name of all tests
#		 call convert_test_line() to get a list of row vectors that contains test value and number of test #pair
#		 for each tests	
# Author: Xiaotian Wang

import csv

# symptons that can be reflected by the lab test
sympton = ['cardiovascular','diabetes','anemia','polycythemia',\
		   'hypochloremia','hyperchloremia','leukopenia','leukocytosis']
		
# parts of body whose's condition can be reflected by the lab test
parts = ['kidney','protein','WBC','heart','liver','bone',\
		 ' pancreas','muscle','RBC','nutrition','lung']

# Indication from the lab test can both indicate the condition of body as well as reveal the symptons 
indication = {'CRP':['cardiovascular'],'UREA':['protein','kindey'],'BASCONT':['WBC'],\
			  'NEUTS':['WBC'],'TROPI':['heart'],'CAI':[],'EOSINCNT':['WBC'],\
			  'MONOCNT':['WBC'],'HB':[],'GLUC':['diabetes'],'GGT':['liver'],\
			  'PHOS':['liver','kidney','bone'], 'AST':['liver'],'NA':[],\
			  'POT':['kidney'],'BILICON':[],'TP':['kidney','liver'],'AMYLASE':['pancreas'],\
			  'CK':['muscle','heart'],'RDW':['RBC']
			}

# dictionary that holds the normal test value range for each lab test
normal_result = {'CRP':(1,3),'UREA':(12,24),'BASOCNT':(),'NEUTS':(),'TROPI':(0,0.15),
				 'CAI':(),'EOSINCNT':(),'MONOCNT':(),\
				 'HB':(13.8,17.2),'GLUC':(0,15),'GGT':(0,51),'PHOS':(2.4,4.1),'AST':(10,34),
				 'NA':(),'POT':(3.5,5.1),'BILICON':(),\
				 'TP':(39,51),'AMYLASE':(23,85),'CK':(10,120),'RDW':(11,15),\
				 'HCT':(0.4,0.5),'CL':(96,106),'CA':(),'URATE':(250,750),'LACTATE':(0.5,2.2),'CAUNCOR':(),\
				 'MG':(),'PLT':(),		 \
				 'CREAT':(),'MCH':(27,31),'OSM':(),'MCHC':(32,36),'ALB':(),'BILI':(),'MCV':(),'ALT':(),\
				 'LYMPCNT':(), 'ALP':(),	'WBC':()}

# The valid range for each lab test, discard a result if the lab test result is outside the range 
valid_result = {'CRP':(1,3),'UREA':(12,24),'BASOCNT':(),'NEUTS':(),'TROPI':(0,0.15),'CAI':(),\
				'EOSINCNT':(),'MONOCNT':(),
				'HB':(),'GLUC':(),'GGT':(0,51),'PHOS':(),'AST':(),'NA':(),'POT':(),'BILICON':(),\
				'TP':(),'AMYLASE':(),\
				'CK':(),'RDW':(),'HCT':(),'CL':(),'CA':(),'URATE':(),'LACTATE':(),'CAUNCOR':(),\
				'MG':(),'PLT':(),		 \
				'CREAT':(),'MCH':(),'OSM':(),'MCHC':(),'ALB':(),'BILI':(),'MCV':(),'ALT':(),\
				'LYMPCNT':(), 'ALP':(),	'WBC':()}

# @param reader: csv.reader 
# @return test_dict: dictionary
def get_test(reader,start_col):
	test_dict = {}
	for row in reader:
		i = start_col
		while i<len(row):
			if (test_dict.has_key(row[i]) == False):
				# list contain all the test data of a corresponding test
				new_list = []
				new_list.append(len(test_dict))
				new_list.append(float(row[i+2]))
				test_dict[row[i]] = new_list
			else:
				test_dict[row[i]].append(float(row[i+2]));
				
			i += 3				
	#print_testValueRange(test_dict)	
	return test_dict

# @param reader: csv.reader
# @param test_dict: dictionary
# @return data: list contain all row list of converted test data
def convert_test_line(reader,test_dict,start_col):
	data = []
	for row in reader:
		data_row = []
		for i in range(0, len(test_dict)):
			data_row.append(0);
			
		# store only the latest lab test value in the data
		i = start_col
		while i<len(row):
			data_row[test_dict[row[i]][0]] = float(row[i+2])
			i+=3	
		''''
		i = start_col
		while i<len(row):
			data_row[2*test_dict[row[i]][0]] += float(row[i+2])
			data_row[2*(test_dict[row[i]][0])+1] += 1
			i += 3
		# compute the average value for conducted tests
		i = 0
		while i<len(data_row):
			if(data_row[i+1] != 0): data_row[i] = data_row[i]/data_row[i+1]
			i +=2
		'''
		data.append(data_row)
	return data

# @param	 file_name: a csv file that contains the lab test data
# @param test_dict: an dictionary that contains the test name as key 
# 				    and all the value of that test as value
# @return: an dictionary that contains a test name as key and a list of
# noraml range for man, normal range for women, normal range for children
# sympton or parts that are affected if the result is less than normal
# sympton or parts that are affected if the result is greater than normal as value
def read_labtestcvs(file_name,test_dict):
	diag_dict = {}
	# a dictionary that reflect index to test name
	help_dict = {}
	
	i = 0
	for key,value in test_dict.iteritems():	
		diag_dict[key] = []
		help_dict[i] = key
		i+=1

	#print help_dict

	new_file = open(file_name,"rb")
	new_reader = csv.reader(new_file,delimiter=',')
	for index,row in enumerate(new_reader):
		# store the normal range value for different group of people
		for i in range(0,3):
			if row[i] != "":
				t1 = row[i].split()[0]
				t2 = row[i].split()[1]
				diag_dict[help_dict[index]].append((t1,t2))
		# store the affected result
		for i in range(3,5):
			if diag_dict[help_dict[index]] == []:
				break;
			elif	 row[i] != "":
				diag_dict[help_dict[index]].append(row[i].split())
			else:
				diag_dict[help_dict[index]].append([])
	
	#print diag_dict
	return diag_dict
	
# @param test_dict: a dictionary whose keys are the name of the lab test
# whose values are the test result
def print_testValueRange(test_dict):
	print test_dict['WBC'] 
		
	for key, value in test_dict.iteritems():
		print str(key) + "     " + str(min(value[1:len(value)-1])) +\
		"     "+str(max(value[1:len(value)-1])) + "    "+\
		str(sum(value[1:len(value)-1])/(len(value)-1))

			
# process the lab test data

rfile = open("test.csv","rb")
t_reader = csv.reader(rfile,delimiter=',')
test_dict = get_test(t_reader,9)
rfile.seek(0)
read_labtestcvs("LabTestsInfo.csv", test_dict)
#data = convert_test_line(t_reader,test_dict,9)