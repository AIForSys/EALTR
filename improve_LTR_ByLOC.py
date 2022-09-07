# coding=utf-8
import numpy as np
import pandas as pd
from Processing import Processing
from configuration_file import configuration_file

from rankboostYin import *
# import xlrd
import shutil
from PerformanceMeasure import PerformanceMeasure

import warnings

from LTR_New import *

from LTR_New import LTR
import joblib
from imblearn.under_sampling import RandomUnderSampler




warnings.filterwarnings('ignore')
header = ["dataset", "recall", "precision", "pmi", "ifma", "f1x", "recallpmi", "popt","PofB","Pofbpmi"]


def transform_data(original_data):
    original_data = original_data.iloc[:, :]

    original_data = np.array(original_data)

    k = len(original_data[0])



    original_data = np.array(original_data)
    original_data_X = original_data[:, 0:k - 1]

    original_data_y = original_data[:, k - 1]
    y_list = []
    for i in original_data_y:
        if i >= 1:
            y_list.append(1)
        else:
            y_list.append(0)
    return original_data_X, y_list

def transform_data1(original_data):

    original_data = original_data.iloc[:, :]

    original_data = np.array(original_data)

    k = len(original_data[0])



    original_data = np.array(original_data)
    original_data_X = original_data[:, 0:k - 1]

    original_data_y = original_data[:, k - 1]

    return original_data_X, original_data_y



def newdata_training(training_data_X, training_data_y,index):

    want_row = [i for i in range(len(training_data_X))]
    new_train_data_x = training_data_X[want_row]

    want_col = [j for j in range(0, index)] + [j for j in range(index + 1, len(training_data_X[0]))]
    print('wantcol',want_col)
    new_train_data_x = new_train_data_x[:, want_col]


    loc_train = training_data_X[:, [index]].squeeze()

    new_train_data_y = [i for i in training_data_y]

    for i in range(len(training_data_y)):
        if (loc_train[i] > 0.0):
            new_train_data_y[i] = training_data_y[i] / loc_train[i]
        else:
            new_train_data_y[i] = 0



    return new_train_data_x, new_train_data_y




if __name__ == '__main__':


    '''
    
    '''

for p in range(1, 21):
    dataset_train = pd.core.frame.DataFrame()
    dataset_test = pd.core.frame.DataFrame()

    folder_path = configuration_file().crossversiondatafolderPath + '/'
    resultlist = []
    resultlist.append(header)

    for root, dirs, files, in os.walk(folder_path):

        if root == folder_path:

            thisroot = root
            for dir in dirs:
                dir_path = os.path.join(thisroot, dir)

                for root, dirs, files, in os.walk(dir_path):
                    if (files[0][-7:-4] < files[1][-7:-4]):
                        file_path_train = os.path.join(dir_path, files[0])
                        file_path_test = os.path.join(dir_path, files[1])
                        trainingfile = files[0]
                        testingfile = files[1]
                    else:
                        file_path_train = os.path.join(dir_path, files[1])
                        file_path_test = os.path.join(dir_path, files[0])
                        trainingfile = files[1]
                        testingfile = files[0]

                    print('files[0][-7:-4]', files[0][-7:-4])
                    print('files[1][-7:-4]', files[1][-7:-4])
                    print(files[0][-7:-4], '>', files[1][-7:-4])
                    print('train', file_path_train)
                    print('test', file_path_test)
                    print('***********************************')

                    dataset_train = pd.read_csv(file_path_train)
                    dataset_test = pd.read_csv(file_path_test)

                    training_data_x, training_data_y = transform_data1(
                        dataset_train)
                    testing_data_x, testing_data_y = transform_data1(
                        dataset_test)


                    new_training_data_x,new_training_data_y= newdata_training(training_data_x,training_data_y,10)
                    new_testing_data_x, new_testing_data_y = newdata_training(testing_data_x, testing_data_y, 10)


                    traincodeN = training_data_x[:, 10]




                    cost = traincodeN
                    de = LTR(X=new_training_data_x, y=new_training_data_y, cost=cost, costflag='loc',
                             logorlinear='linear')
                    modelsavepath = 'E:/PytcharmProject/Effort-Aware Defect Prediction/NewEALTRModel/' + 'EALTRModel_' + trainingfile + '_version_' + str(p) + '.pkl'

                    Ltr_w = joblib.load(modelsavepath)


                    testingcodeN = testing_data_x[:, 10]
                    Ltr_linear_pred = de.predict(new_testing_data_x, Ltr_w)
                    EALTR_pred=np.array(Ltr_linear_pred)
                    EALTR_pred_topIndex=np.argsort(-EALTR_pred)



                    train_LOC=[]
                    for i in range(len(training_data_y)):
                        if training_data_y[i]==0:
                            train_LOC.append(traincodeN[i])
                    median_train_loc=np.median(train_LOC)
                    print("###median_train_loc###")
                    print(median_train_loc)


                    EALTR_trainingdata=[]
                    LTR_DP_lTR = []

                    EALTR_trainingdata_pred = de.predict(new_training_data_x, Ltr_w)

                    for i in range(len(EALTR_trainingdata_pred)):
                        EALTR_trainingdata.append(EALTR_trainingdata_pred[i]*traincodeN[i])

                    print(len(training_data_y))
                    print(len(EALTR_trainingdata))
                    print(len(traincodeN))
                    traindataPrecisionx, traindataRecallx, traindataF1x, traindataIFMA, traindataPMI, traindatarecallPmi, traindataPofB, traindataPofbpmi=PerformanceMeasure(training_data_y, EALTR_trainingdata,traincodeN, 0.2,'density','loc').Performance()

                    print(traindataPrecisionx)
                    print(traindataIFMA)
                    if traindataPrecisionx < 0.1 or traindataIFMA >1:

                        for j in range(25):
                            topIndexNum=EALTR_pred_topIndex[j]
                            if testingcodeN[topIndexNum] < median_train_loc:
                                topIndexValue=Ltr_linear_pred[topIndexNum]
                                Ltr_linear_pred[topIndexNum]=topIndexValue-100000000


                    for j in range(len(Ltr_linear_pred)):

                            LTR_DP_lTR.append(Ltr_linear_pred[j]*testingcodeN[j])


                    Precisionx, Recallx, F1x, IFMA, PMI, recallPmi, PofB, Pofbpmi = PerformanceMeasure(testing_data_y, LTR_DP_lTR,
                                                                                            testingcodeN, 0.2,
                                                                                        'density',
                                                                                        'loc').Performance()
                    wholenormOPT = PerformanceMeasure(testing_data_y, LTR_DP_lTR, testingcodeN, 0.2, 'density',
                                                      'loc').POPT()
                    Results = []
                    dataset = trainingfile + testingfile

                    Results.append(dataset)
                    Results.append(Recallx)
                    Results.append(Precisionx)
                    Results.append(PMI)
                    Results.append(IFMA)
                    Results.append(F1x)
                    Results.append(recallPmi)
                    Results.append(wholenormOPT)
                    Results.append(PofB)
                    Results.append(Pofbpmi)
                    resultlist.append(Results)

                    print("**********Recallx**********")
                    print(Recallx)
                    print("**********Precisionx**********")
                    print(Precisionx)
                    print("**********PMI**********")
                    print(PMI)
                    print("**********IFMA**********")
                    print(IFMA)
                    print("**********F1x**********")
                    print(F1x)
                    print("**********recallPmi**********")
                    print(recallPmi)
                    print("**********wholenormOPT**********")
                    print(wholenormOPT)
                    print("**********PofB**********")
                    print(PofB)
                    print("**********PofBpmi**********")
                    print(Pofbpmi)

    result_path = configuration_file().save_paper_result
    result_csv_name = "Improve_EALTR_ByPrecisionIFALoc_0.1_1_25_20220503_"+str(p)+".xlsx"

    result_path = os.path.join(result_path, result_csv_name)
    Processing().write_excel(result_path, resultlist)









