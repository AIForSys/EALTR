# coding=utf-8
import numpy as np
import pandas as pd
from Processing import Processing
from configuration_file import configuration_file

from sklearn.metrics import make_scorer

from rankboostYin import *

from PerformanceMeasure import PerformanceMeasure

from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier


from sklearn.ensemble import RandomForestRegressor

from LTR_New import *

from deepforest import CascadeForestClassifier
from gcforest.gcforest import GCForest
from LTR_New import LTR
import joblib  # joblib是sklearn的一个外部模块
from imblearn.under_sampling import RandomUnderSampler




#warnings.filterwarnings('ignore')
header = ["dataset", "recall", "precision", "pmi", "ifma", "f1x", "recallpmi", "popt","PofB"]




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


for p in range(1,21):
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
                    testingcodeN=testing_data_x[:,10]



                    cost = traincodeN
                    de = LTR(X=new_training_data_x, y=new_training_data_y, cost=cost, costflag='loc',
                             logorlinear='linear')
                    Ltr_w = de.process()

                    #Ltr_linear_pred = de.predict(new_testing_data_x, Ltr_w)


                    modelsavepath='E:/PycharmProjects/Effort-Aware Defect Prediction/NewEALTRModel/'+'EALTRModel_'+trainingfile+'_version_'+str(p)+'.pkl'
                    joblib.dump(Ltr_w, modelsavepath)
                    '''

                    LTR_pred=[]

                    for j in range(len(Ltr_linear_pred)):

                            LTR_pred.append((Ltr_linear_pred[j]+100000)*testingcodeN[j])



                    Precisionx, Recallx, F1x, IFMA, PMI, recallPmi,PofB = PerformanceMeasure(testing_data_y, LTR_pred,
                                                                                            testingcodeN, 0.2,
                                                                                        'density',
                                                                                        'loc').Performance()
                    wholenormOPT = PerformanceMeasure(testing_data_y, LTR_pred, testingcodeN, 0.2, 'density',
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

    result_path = configuration_file().save_PredBugDensityResult_dir
    result_csv_name = "ltr_num_pofb_100000.xlsx"
    result_path = os.path.join(result_path, result_csv_name)
    Processing().write_excel(result_path, resultlist)
    '''








