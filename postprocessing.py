import csv
import csv_io
import numpy as np

def getBenchmark(test_label):
    tfile = open('MyBenchmark.csv','wb')
    test_info = csv_io.read_data('test_info.csv')
    
    try:
        twriter =  csv.writer(tfile, delimiter=',')
        twriter.writerow(['Id','Prediction'])
        index = 0;
        
        print np.array(test_info)
        print test_info[0][0], test_label[0]
        
        #print np.array(test_info)        
        for id in test_info:
            #print 'converted line', index
            twriter.writerow( [int(id[0]), test_label[index]] )
            index +=1
            
    finally:
        tfile.close()
                

'''   
ffile = open('test_regLinearRegPred_lambda0.csv','rb')
ifile = open('test_info.csv', 'rb')
tfile = open('ptest_regLinearRegPred_lambda0.csv','wb')
try:
    freader = csv.reader(ffile, delimiter=',')
    ireader = csv.reader(ifile, delimiter=',')
    twriter = csv.writer(tfile, delimiter=',')
    twriter.writerow(['Id','Prediction'])
    lcount = 1 #line count
    linef = freader.next()
    linei = ireader.next()
    while linef and linei:
    # for i in xrange(len(ireader)):
        # if lcount < 20:
        twriter.writerow([linei[0], linef[0]])
        print 'converted line', lcount
        linef = freader.next()
        linei = ireader.next()
        lcount += 1
        
finally:
    ffile.close()
    ifile.close()
    tfile.close()
'''
