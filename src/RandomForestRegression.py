#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 13:42:05 2018

@author: mhci430
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 16:02:28 2018
@author: mhci430
"""
import math
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedKFold # Add important libs
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

Submission_filename="/Users/mhci430/Documents/R06922088/MachineLearningTechniques/FinalProject/data/book_ratings_test.csv"
Raw_Book_filename="/Users/mhci430/Documents/R06922088/MachineLearningTechniques/FinalProject/data/books.csv"
User_filename="/Users/mhci430/Documents/R06922088/MachineLearningTechniques/FinalProject/data/users_pre.csv"
Rate_filename="/Users/mhci430/Documents/R06922088/MachineLearningTechniques/FinalProject/data/book_ratings_train.csv"
Book_filename="/Users/mhci430/Documents/R06922088/MachineLearningTechniques/FinalProject/data/BookClassificationResult.csv"

#


BookData=pd.read_csv(Book_filename)





RateData=pd.read_csv(Rate_filename)

UserData=pd.read_csv(User_filename)


Raw_book=pd.read_csv(Raw_Book_filename)



towmerge=pd.merge(RateData, UserData, how='left', on=['User-ID'])
threedatamerge=pd.merge(towmerge, BookData, how='left', on=['ISBN'])
threedatamerge.drop(columns=['BookTitle', 'BookAuthor','BookPublisher'],inplace=True)
alldatamerge=pd.merge(threedatamerge,Raw_book, how='left', on=['ISBN'])
alldatamerge.drop(columns=['Image-URL-S', 'Image-URL-M','Image-URL-L','Book-Description'],inplace=True)
  

##label and one hot

lebal_Location=LabelEncoder()
alldatamerge['Location']=lebal_Location.fit_transform(alldatamerge['Location'])

lebal_Age=LabelEncoder()
alldatamerge['Age']=lebal_Age.fit_transform(alldatamerge['Age'])

OneHotAge=np.zeros((len(alldatamerge),len(np.unique(alldatamerge['Age']))))
OneHotLocation=np.zeros((len(alldatamerge),len(np.unique(alldatamerge['Location']))))


for i in range(len(alldatamerge)):
    
    OneHotAge[i][int(alldatamerge['Age'][i])]=1
    OneHotLocation[i][int(alldatamerge['Location'][i])]=1


OneHotAgeDataFrame=pd.DataFrame(OneHotAge,index=range(len(OneHotAge)))
OneHotLocationDataFrame=pd.DataFrame(OneHotLocation,index=range(len(OneHotLocation)))

FinalData=alldatamerge.drop(columns=['Age','Location','Book-Title', 'Book-Author','Publisher','BookType','Year-Of-Publication'])
FinalData=pd.merge(FinalData,OneHotAgeDataFrame,left_index=True)
FinalData= pd.concat([FinalData,OneHotAgeDataFrame,OneHotLocationDataFrame],axis=1, ignore_index=True)

X=FinalData.iloc[:,3:].values
y=FinalData.iloc[:,2].values



from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression



regr = RandomForestRegressor(max_depth=2,n_estimators=100,criterion='mse',random_state=1,n_jobs=-1)
regr.fit(X, y)


####test

TestData=pd.read_csv(Submission_filename)


towmergetest=pd.merge(TestData, UserData, how='left', on=['User-ID'])


towmergetest['Location']=lebal_Location.fit_transform(towmergetest['Location'])


towmergetest['Age']=lebal_Age.fit_transform(towmergetest['Age'])

OneHotAgeTest=np.zeros((len(towmergetest),len(np.unique(towmergetest['Age']))))
OneHotLocationTest=np.zeros((len(towmergetest),len(np.unique(towmergetest['Location']))))


for i in range(len(alldatamerge)):
    
    OneHotAgeTest[i][int(towmergetest['Age'][i])]=1
    OneHotLocationTest[i][int(towmergetest['Location'][i])]=1


OneHotAgeDataFrameTest=pd.DataFrame(OneHotAgeTest,index=range(len(OneHotAgeTest)))
OneHotLocationDataFrameTest=pd.DataFrame(OneHotLocationTest,index=range(len(OneHotLocationTest)))

FinalDataTest= pd.concat([OneHotAgeDataFrameTest,OneHotLocationDataFrameTest],axis=1, ignore_index=True)

Predict=regr.predict(FinalDataTest.values)
predict=pd.DataFrame(Predict)
predict.to_csv("/Users/mhci430/Documents/R06922088/MachineLearningTechniques/FinalProject/Output.csv")
