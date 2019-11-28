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
    x_train=train[features]#.iloc[:,1:]#每一行,列(除了target)
    print(x_train.head(2))
    y_train=train['target']
    return x_train,y_train

def get_test_data():
    test=pd.read_csv(filenameTest,index_col=0) #去掉第一行的預設值
    test=test[features]
    return test
    
    
x_train0,y_train=get_train_data()
test=get_test_data()


#使用Logistic Regression之前需要先對資料做特徵縮放
#使用sklearn.preprocessing.StandardScaler類，使用該類的好處在於可以保存訓練集中的參數（均值、方差）直接使用其對象轉換測試集數據。
from sklearn.preprocessing import StandardScaler,RobustScaler
scaler=RobustScaler()
x_train=pd.DataFrame(scaler.fit_transform(x_train0),columns=x_train0.columns,index=x_train0.index)
test=pd.DataFrame(scaler.fit_transform(test),columns=test.columns,index=test.index)

'''
from sklearn.decomposition import PCA
pca = PCA(n_components=100)
pca.fit(x_train)
print(x_train.shape)
'''

#logistic regression
from sklearn.linear_model import LogisticRegression
log = LogisticRegression(
        penalty='l1', #l1:拉普拉斯分佈; l2:高斯分佈
        dual=False,
        tol=1e-4,
        C=0.2, #越小的數值表示越強的正則化，正則化是用來防止模型過擬合的過程 1.0/0.5/0.3/0.2/0.1
        fit_intercept=False, #指定是否應該向決策函數添加常量(即偏差或截距)
        intercept_scaling=1, #：僅在solver為”liblinear”，且fit_intercept設置為True時有用
        class_weight={0:0.4,1:0.6}, #比如對於0,1的二元模型，我們可以定義class_weight={0:0.9,1:0.1}，這樣類型0的權重為90%，而類型1的權重為10%:None/{0:0.9,1:0.1}/balanced
        random_state=None,
        solver='liblinear', #liblinear適用於小數據集，而sag和saga適用於大數據集因為速度更快:liblinear/saga/newton-cg/lbfgs/sag
        max_iter=100,#僅在正則化優化算法為newton-cg, sag和lbfgs才有用，算法收斂的最大迭代次數。
        multi_class= 'ovr', #ovr/multinomial:多分類
        verbose=0, #印出模型
        warm_start=False, #如果為True，則下一次訓練是以追加樹的形式進行（重新使用上一次的調用作為初始化）
        n_jobs=1 #1的時候，用CPU的一個內核運行程序，2的時候，用CPU的2個內核運行程序。為-1的時候，用所有CPU的內核運行程序
        )


log.fit(x_train,y_train)



#result
prediction = log.predict(test)
prediction = pd.DataFrame(prediction)
prediction.index += 250
prediction.columns = ['target']
prediction.to_csv('C:/Users/user/Desktop/class/maching learning/project/group/Don-t_Overfit_PolyU/result/LG(RobustScaler)(probing).csv', index_label='id', index=True)
 


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




