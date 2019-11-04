import os
import pandas as pd

data_dir= 'C:/Users/user/Desktop/class/maching learning/project/group/Don-t_Overfit_PolyU/dataset'


#observe train data
filenameTrain=os.path.join(data_dir,'train.csv')
train=pd.read_csv(filenameTrain)
#print(train.info) #把所有col. row列印出來
#print(train.describe()) #每行的統計數據
#print(train.head(3)) #可以顯示5筆資料（預設是5筆）
#print(train.shape) #總結col.row的數量

#observe test data
filenameTest=os.path.join(data_dir,'test.csv')
test=pd.read_csv(filenameTest)
#print(test.info) #把所有col. row列印出來
#print(test.describe()) #每行的統計數據
#print(test.head()) #可以顯示5筆資料（預設是5筆）
#print(test.shape) #總結col.row的數量

#load prepared data
def get_train_data():
    train=pd.read_csv(filenameTrain,index_col=0) #去掉第一行的預設值
    x_train=train.iloc[:,1:]#每一行,列(除了target)
    y_train=train['target']
    return x_train,y_train

def get_test_data():
    test=pd.read_csv(filenameTest,index_col=0) #去掉第一行的預設值
    return test
    
    
x_train0,y_train=get_train_data()
test=get_test_data()
#print(x_train0.shape)
#print(test.shape)


#使用Logistic Regression之前需要先對資料做特徵縮放
#使用sklearn.preprocessing.StandardScaler類，使用該類的好處在於可以保存訓練集中的參數（均值、方差）直接使用其對象轉換測試集數據。
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=pd.DataFrame(scaler.fit_transform(x_train0),columns=x_train0.columns,index=x_train0.index)
test=pd.DataFrame(scaler.fit_transform(test),columns=test.columns,index=test.index)


#SVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,StratifiedKFold

n_fold = 20
folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
svc = SVC(probability=True, gamma='scale')

parameter_grid = {'C': [0.001, 0.01, 0.1, 1.0, 10.0],
                  'kernel': ['linear', 'poly', 'rbf'],
                 }

grid_search = GridSearchCV(svc, param_grid=parameter_grid, cv=folds, scoring='roc_auc', n_jobs=-1)
grid_search.fit(x_train,y_train)


#result
prediction = grid_search.predict(test)
prediction = pd.DataFrame(prediction)
prediction.index += 250
prediction.columns = ['target']
prediction.to_csv('C:/Users/user/Desktop/class/maching learning/project/group/Don-t_Overfit_PolyU/result/SVC(GridSearchCV).csv', index_label='id', index=True)
