import os
import pandas as pd

data_dir= 'C:/Users/user/Desktop/class/maching learning/project/group/Don-t_Overfit_PolyU/dataset'

#observe train data
filenameTrain=os.path.join(data_dir,'train.csv')
train=pd.read_csv(filenameTrain)

correlation = train.corr(method='pearson')


correlation=correlation.iloc[0]#每一行,列(除了target)
print(correlation)