import os
import pandas as pd

data_dir= 'C:/Users/user/Desktop/class/maching learning/project/group/Don-t_Overfit_PolyU/dataset'

features=['0','9','15','16','17','24','33','39','43','45','63','65','73','80','89','90','91','98','101',
 '105','117','133','134','143','150','156','164','176','183','189','194','199','209','214','215',
 '217','221','227','228','230','237','239','240','244','253','258','276','295','298']

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
    x_train=train.iloc[:,1:]#.iloc[:,1:]#每一行,列(除了target)
    y_train=train['target']
    return x_train,y_train

def get_test_data():
    test=pd.read_csv(filenameTest,index_col=0) #去掉第一行的預設值
    #test=test[features]
    test=test[features]
    return test
    
    
x_train0,y_train=get_train_data()
test=get_test_data()
#print(x_train0.shape)
#print(test.shape)


#使用Logistic Regression之前需要先對資料做特徵縮放
#使用sklearn.preprocessing.StandardScaler類，使用該類的好處在於可以保存訓練集中的參數（均值、方差）直接使用其對象轉換測試集數據。
from sklearn.preprocessing import StandardScaler,RobustScaler
scaler=RobustScaler()
x_train=pd.DataFrame(scaler.fit_transform(x_train0),columns=x_train0.columns,index=x_train0.index)
test=pd.DataFrame(scaler.fit_transform(test),columns=test.columns,index=test.index)


#SVC
from sklearn.svm import SVC
svc = SVC(
        C=1.0,  
        kernel='linear', #'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' 
        degree=3, 
        gamma='auto', #'scale'
        coef0=0.0, 
        shrinking=True, 
        probability=False, #是否使用概率估計
        tol=0.001, #0.001殘差收斂條件
        cache_size=200, #緩衝大小
        class_weight=None, #{0:0.4,1:0.6}, 'balance'
        verbose=False, 
        max_iter=-1, 
        decision_function_shape=None, #'ovo', 'ovr' or None
        random_state=None #數據洗牌時的種子值
        )

svc.fit(x_train,y_train)


#result
prediction = svc.predict(test)
prediction = pd.DataFrame(prediction)
prediction.index += 250
prediction.columns = ['target']
prediction.to_csv('C:/Users/user/Desktop/class/maching learning/project/group/Don-t_Overfit_PolyU/result/SVC(RobustScaler)(probing).csv', index_label='id', index=True)


#RFECV                                                                           
['4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22',
'23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40',
'41','43','50','63','65','73','80','82','90','91','101','108','117','129','133','134','138',
'139','40','165','183','189','194','199','201','217','226','227','234','235','236','237','238',
'239','240','241','242','243','244','245','247','248','249','250','251','252','253','258','276',
'293','294','295','298']


#pearson's correlation
['0','3','7','12','13','14','17','18','21','24','25','27','29','30','33','36','42','44','46',
'47','50','53','54','59','60','65','67','69','71','72','74','78','79','81','84','88','89','92',
'99','100','101','103','104','105','107','112','114','115','119','122','126','130','131','137',
'145','152','154','157','160','162','163','164','168','170','171','174','176','177','183','184',
'185','186','187','188','190','193','199','200','201','203','204','205','207','208','214','215',
'218','219','221','222','224','225','226','234','235','240','241','244','246','247','248','249',
'253','255','256','257','259','263','264','265','266','267','271','272','273','274','278','280',
'282','283','285','289','290','291','296','297']

#probing
['0','9','15','16','17','24','33','39','43','45','63','65','73','80','89','90','91','98','101',
 '105','117','133','134','143','150','156','164','176','183','189','194','199','209','214','215',
 '217','221','227','228','230','237','239','240','244','253','258','276','295','298']