#
# Convert train.csv into usable form. Outputs csv file including only numbers.
# Converts every field into a number ({0,1}, int, float).
# Author: Yi Hua
#
# Numeric conversion rules:
# gender: M -> 0, F -> 1
# admission_type: Planned -> 0, Emergency -> 1, Other Hospital -> 2, Other -> 3
# outcome: Home -> 0, Care Home -> 1, Other Hospital -> 2, Died ->3, Other -> 4

from datetime import datetime 
import csv_io
import testprocessing
import unsplearnHmm
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

# List of symptons that can be reflected by lab tests
sympton_list = ['inflammation','cardiovascular','diabetes','anemia','polycythemia',\
           'hypochloremia','hyperchloremia','leukopenia','leukocytosis']

# List of organ whose's condition can be reflected by lab tests
organ_list = ['kidney','protein','WBC','heart','liver','bone',\
         'pancreas','muscle','RBC','nutrition','lung']

# The max number of sequence of states for patient
max_states_number = 79


"""
    Converts input line into all numbers and writes to a new line in writer
    Input:  
            classification      the class for a particular patient's health sequence
            test_data   a list with patient's lab test value
            diag_dict   a dictionary with organ or sympton as key and normal range as value
            row         a list of strings
            datawriter  csv file to write data to
            labelwriter csv file to write label to 
            infowriter  csv file to write information that doesn't predict result
            train       true for processing train data (starting with label);
                        false for test data
    Output: none
    Effect: Writes a new line to the writer if successfully processed
"""

def convert_line(classification, test_data, diag_dict, row, datawriter, labelwriter, infowriter, train):
    datarow = []
    start_column = 0
    if train: # write label
        labelwriter.writerow(row[field.get('label')])
        start_column = 10
    else: # writer info
        infowriter.writerow([row[field.get('admission_id')-1]])
        start_column = 9
    
    age_index = 0 # 0 for adult men, 1 for adult women, and 2 for children
    
    # write other useful fields listed in use_field
    for i in use_field:
        if train:
            thing = row[field.get(i)]
        else:
            thing = row[field.get(i)-1]
            
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
            year = float(thing.split('-')[0])
            month = float(thing.split('-')[1])
            datarow.append(year)
            datarow.append(month)
            
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
    
    #append the latest test data to the patient info
    datawriter.writerow(datarow+[classification])
    #datawriter.writerow(datarow)
# Process the data.
# This would overwrite files if those files existed!
import csv
proc_train = False
start_col = 9;
if proc_train:
    rfile = open('train.csv','rb')
    dfile = open('train_data.csv','wb')
    lfile = open('train_label.csv','wb')
    ifile = open('train_info.csv', 'wb')
    state_file = 'train_states.csv'
    classreader = csv_io.read_data('train_classes.csv')
else:
    rfile = open('test.csv','rb')
    dfile = open('test_data.csv','wb')
    lfile = open('test_label.csv','wb')
    ifile = open('test_info.csv', 'wb')
    state_file = 'test_states.csv'
    classreader = csv_io.read_data('test_classes.csv')
    
try:
    reader = csv.reader(rfile, delimiter=',')
    datawriter = csv.writer(dfile, delimiter=',')
    labelwriter = csv.writer(lfile, delimiter=',')
    infowriter = csv.writer(ifile, delimiter=',')
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
    classification = Clustering.runKmeans(state_file)
    
    # append a empty row to the first row 
    infowriter.writerow([])
    datawriter.writerow([])
    labelwriter.writerow([])
    
    for index, row in enumerate(reader):
        # if lcount < 20:
        if proc_train:
            convert_line(classification[index],latest_test_data[index], diag_dict, row, datawriter, labelwriter, infowriter, True)
            '''
            if index >20:
                break
            '''
        else:
            convert_line(classification[index],latest_test_data[index], diag_dict, row, datawriter, labelwriter, infowriter, False)
            
        print 'converted line', lcount
        lcount += 1
        
finally:
    rfile.close()
    dfile.close()
    lfile.close()