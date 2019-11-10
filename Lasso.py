import os
import pandas as pd

data_dir= 'C:/Users/user/Desktop/class/maching learning/project/group/Don-t_Overfit_PolyU/dataset'

#observe train data
filenameTrain=os.path.join(data_dir,'train.csv')
train=pd.read_csv(filenameTrain)


#observe test data
filenameTest=os.path.join(data_dir,'test.csv')
test=pd.read_csv(filenameTest)

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


#使用sklearn.preprocessing.StandardScaler類，使用該類的好處在於可以保存訓練集中的參數（均值、方差）直接使用其對象轉換測試集數據。
from sklearn.preprocessing import StandardScaler,RobustScaler
scaler=RobustScaler()
x_train=pd.DataFrame(scaler.fit_transform(x_train0),columns=x_train0.columns,index=x_train0.index)
test=pd.DataFrame(scaler.fit_transform(test),columns=test.columns,index=test.index)

#losso
from sklearn.linear_model import Lasso
'''
lasso=Lasso(
        alpha=1.0, 
        copy_X=True, 
        fit_intercept=True, 
        max_iter=1000,
        normalize=False, 
        positive=False, 
        precompute=False, 
        random_state=None,
        selection='cyclic', 
        tol=0.0001, 
        warm_start=False)
'''


from sklearn.metrics import  make_scorer,roc_auc_score #mean_squared_error, mean_absolute_error, roc_auc_score, r2_score,
# define roc_auc_metric robust to only one class in y_pred
def scoring_roc_auc(y, y_pred):
    try:
        return roc_auc_score(y, y_pred)
    except:
        return 0.5
    
robust_roc_auc = make_scorer(scoring_roc_auc)

lasso = Lasso(alpha=0.031, tol=0.01, random_state=213, selection='random')

from sklearn.feature_selection import RFECV
select=RFECV(lasso, min_features_to_select=12, scoring=robust_roc_auc, step=15, verbose=0, cv=20, n_jobs=-1)
select.fit(x_train,y_train)

'''
if(select.support_=='true'):
    print(select.support_)
print("N_features %s" % select.n_features_)
print("Support is %s" % select.support_)
print("Ranking %s" % select.ranking_)
print("Grid Scores %s" % select.grid_scores_)
'''


#temp = pd.DataFrame({"Support":select.support_, "Ranking":select.ranking_})
#temp.to_csv("C:/Users/user/Desktop/1.csv")


#result
prediction = select.predict(test)
prediction = pd.DataFrame(prediction)
prediction.index += 250
prediction.columns = ['target']
prediction.to_csv('C:/Users/user/Desktop/class/maching learning/project/group/Don-t_Overfit_PolyU/result/Lasso(RobustScaler).csv', index_label='id', index=True)











