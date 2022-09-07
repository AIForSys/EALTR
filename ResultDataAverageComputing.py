import pandas as pd
import numpy as np
from configuration_file import *
from Processing import Processing

resultalldata=['dataset','recall','precision','pmi','ifma','f1x','recallpmi','popt','PofB','PofBpmi']

resultalldataheader=['dataset','recall','precision','pmi','ifma','f1x','recallpmi','popt','PofB','PofBpmi']
resultmedianalldata=[]
resultmedianalldata.append(resultalldataheader)

for i in range(1,21):

    resultdatapath = 'E:/PytcharmProject/CrossProjectExperimentResult/' + 'Unsupervised_allDataSetTest_ManualUp_20220503_crossproject_' + str(i) + '.xlsx'
    data1=pd.read_excel(resultdatapath,engine='openpyxl')
    origin_data1 = data1.iloc[:, :]
    origin_data1 = np.array(origin_data1)
    resultalldata=np.vstack((resultalldata,origin_data1))


resultalldata=np.delete(resultalldata,0,axis=1)



rowdataset=['ant-1.3.csvant-1.4.csv','jedit-3.2.csvjedit-4.0.csv',
 'jedit-4.0.csvjedit-4.1.csv', 'jedit-4.1.csvjedit-4.2.csv',
 'jedit-4.2.csvjedit-4.3.csv' ,'log4j-1.0.csvlog4j-1.1.csv',
 'log4j-1.1.csvlog4j-1.2.csv' ,'lucene-2.0.csvlucene-2.2.csv',
 'lucene-2.2.csvlucene-2.4.csv', 'poi-1.5.csvpoi-2.0.csv',
 'poi-2.0.csvpoi-2.5.csv', 'ant-1.4.csvant-1.5.csv',
 'poi-2.5.csvpoi-3.0.csv', 'synapse-1.0.csvsynapse-1.1.csv',
 'synapse-1.1.csvsynapse-1.2.csv', 'velocity-1.4.csvvelocity-1.5.csv',
 'velocity-1.5.csvvelocity-1.6.csv', 'xalan-2.4.csvxalan-2.5.csv',
 'xalan-2.5.csvxalan-2.6.csv', 'xalan-2.6.csvxalan-2.7.csv',
 'xerces-1.1.csvxerces-1.2.csv' ,'xerces-1.2.csvxerces-1.3.csv',
 'ant-1.5.csvant-1.6.csv', 'xerces-1.3.csvxerces-1.4.csv',
 'ant-1.6.csvant-1.7.csv' ,'camel-1.0.csvcamel-1.2.csv',
 'camel-1.2.csvcamel-1.4.csv', 'camel-1.4.csvcamel-1.6.csv',
 'ivy-1.1.csvivy-1.4.csv', 'ivy-1.4.csvivy-2.0.csv']





 



for j in range(1,31):
    resultrowdata=resultalldata[j]
    resultmedianrowdata=[]
    crossversion=j-1

    resultmedianrowdata.append(rowdataset[crossversion])

    for a in range(1,20):
        num=j+30*a
        resultrowdata = np.vstack((resultrowdata, resultalldata[num]))

    for b in range(0,9):
        resultLdata = resultrowdata[:, b]
        meannum=np.mean(resultLdata)
        resultmedianrowdata.append(meannum)
    resultmedianalldata.append(resultmedianrowdata)


result_path = configuration_file().save_paper_result
result_csv_name = "Unsupervised_allDataSetTest_ManualUp_20220503_crossproject_"+"mean.xlsx"
result_path = os.path.join(result_path, result_csv_name)
Processing().write_excel(result_path, resultmedianalldata)






