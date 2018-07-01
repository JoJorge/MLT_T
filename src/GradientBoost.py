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
#
#from time import clock
#start = clock() 
#
#
#import tracemalloc
#import sys
 
# Simulate memory allocate by creating bytes

 
# Start tracing
#tracemalloc.start()
 
# Your program run here

 
# Take memory snapshot

import time

start = time.time()

import tracemalloc

tracemalloc.start()

# ... run your application ...




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

Submission_filename="data/book_ratings_test.csv"
Raw_Book_filename="data/books_pre_most.csv"
User_filename="data/users_DT.csv"
Rate_filename="data/book_ratings_train.csv"
Book_filename="data/BookClassificationResult_onlyTitle.csv"

#


BookData=pd.read_csv(Book_filename)





RateData=pd.read_csv(Rate_filename)

UserData=pd.read_csv(User_filename)


Raw_book=pd.read_csv(Raw_Book_filename)


towmerge=pd.merge(RateData, UserData, how='left', on=['User-ID'])
threedatamerge=pd.merge(towmerge, BookData, how='left', on=['ISBN'])
threedatamerge.drop(columns=['BookTitle', 'BookAuthor','BookPublisher'],inplace=True)
alldatamerge=pd.merge(threedatamerge,Raw_book, how='left', on=['ISBN'])
#alldatamerge.drop(columns=['Image-URL-S', 'Image-URL-M','Image-URL-L','Book-Description'],inplace=True)

###Check none or nan
alldatamerge['BookType'].fillna(value=36,inplace=True)
alldatamerge['Book-Author'].fillna(value='unknown',inplace=True)
alldatamerge['Publisher'].fillna(value='unknown',inplace=True)
alldatamerge['Year-Of-Publication'].fillna(value=-1,inplace=True)



#        



##label and one hot

lebal_Location=LabelEncoder()
alldatamerge['Location']=lebal_Location.fit_transform(alldatamerge['Location'])

lebal_Author=LabelEncoder()
alldatamerge['Book-Author']=lebal_Author.fit_transform(alldatamerge['Book-Author'])

lebal_Publisher=LabelEncoder()
alldatamerge['Publisher']=lebal_Publisher.fit_transform(alldatamerge['Publisher'])


#OneHotAge=np.zeros((len(alldatamerge),len(np.unique(alldatamerge['Age']))))
OneHotLocation=np.zeros((len(alldatamerge),len(np.unique(alldatamerge['Location']))))
OneHotBookType=np.zeros((len(alldatamerge),len(np.unique(alldatamerge['BookType']))))
OneHotAuthor=np.zeros((len(alldatamerge),len(np.unique(alldatamerge['Book-Author']))))
OneHotPublisher=np.zeros((len(alldatamerge),len(np.unique(alldatamerge['Publisher']))))



for i in range(len(alldatamerge)):
    
    #OneHotAge[i][int(alldatamerge['Age'][i])]=1
    OneHotLocation[i][int(alldatamerge['Location'][i])]=1
    OneHotBookType[i][int(alldatamerge['BookType'][i])]=1
    OneHotAuthor[i][int(alldatamerge['Book-Author'][i])]=1
    OneHotPublisher[i][int(alldatamerge['Publisher'][i])]=1

    
    

#OneHotAgeDataFrame=pd.DataFrame(OneHotAge,index=range(len(OneHotAge)))
OneHotLocationDataFrame=pd.DataFrame(OneHotLocation,index=range(len(OneHotLocation)))
OneHotBookTypeDataFrame=pd.DataFrame(OneHotBookType,index=range(len(OneHotBookType)))
OneHotAuthorDataFrame=pd.DataFrame(OneHotAuthor,index=range(len(OneHotAuthor)))
OneHotPublisherDataFrame=pd.DataFrame(OneHotPublisher,index=range(len(OneHotPublisher)))




FinalData=alldatamerge.drop(columns=['Location', 'Book-Author','Publisher','BookType'])

FinalData= pd.concat([FinalData,OneHotLocationDataFrame,OneHotBookTypeDataFrame,OneHotAuthorDataFrame,OneHotPublisherDataFrame],axis=1, ignore_index=True)

X=FinalData.iloc[:,3:].values
y=FinalData.iloc[:,2].values



from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import xgboost as xgb

xgdmat=xgb.DMatrix(X,y)
our_params={'eta':0.1,'seed':0,'subsample':0.8,'colsample_bytree':0.8,'objective':'reg:linear','max_depth':3,'min_child_weight':1}
final_gb=xgb.train(our_params,xgdmat)


#regr = RandomForestRegressor(max_depth=2,n_estimators=100,criterion='mse',random_state=1,n_jobs=-1)
#regr.fit(X, y)


####test

TestData=pd.read_csv(Submission_filename)


towmergetest=pd.merge(TestData, UserData, how='left', on=['User-ID'])
threedatamergetest=pd.merge(towmergetest, BookData, how='left', on=['ISBN'])
threedatamergetest.drop(columns=['BookTitle', 'BookAuthor','BookPublisher'],inplace=True)
alldatamergetest=pd.merge(threedatamergetest,Raw_book, how='left', on=['ISBN'])


###Check none or nan
alldatamergetest['BookType'].fillna(value=36,inplace=True)
alldatamergetest['Book-Author'].fillna(value='unknown',inplace=True)
alldatamergetest['Publisher'].fillna(value='unknown',inplace=True)
alldatamergetest['Year-Of-Publication'].fillna(value=-1,inplace=True)



#        



##label and one hot

lebal_Location_test=LabelEncoder()
alldatamergetest['Location']=lebal_Location_test.fit_transform(alldatamergetest['Location'])

lebal_Author_test=LabelEncoder()
alldatamergetest['Book-Author']=lebal_Author_test.fit_transform(alldatamergetest['Book-Author'])

lebal_Publisher_test=LabelEncoder()
alldatamergetest['Publisher']=lebal_Publisher_test.fit_transform(alldatamergetest['Publisher'])


#OneHotAge=np.zeros((len(alldatamerge),len(np.unique(alldatamerge['Age']))))
OneHotLocationtest=np.zeros((len(alldatamergetest),len(np.unique(alldatamergetest['Location']))))
OneHotBookTypetest=np.zeros((len(alldatamergetest),37))
OneHotAuthortest=np.zeros((len(alldatamergetest),len(np.unique(alldatamergetest['Book-Author']))))
OneHotPublishertest=np.zeros((len(alldatamergetest),len(np.unique(alldatamergetest['Publisher']))))



for i in range(len(alldatamergetest)):
    
    #OneHotAge[i][int(alldatamerge['Age'][i])]=1
    OneHotLocationtest[i][int(alldatamergetest['Location'][i])]=1
    OneHotBookTypetest[i][int(alldatamergetest['BookType'][i])]=1
    OneHotAuthortest[i][int(alldatamergetest['Book-Author'][i])]=1
    OneHotPublishertest[i][int(alldatamergetest['Publisher'][i])]=1

    
    

#OneHotAgeDataFrame=pd.DataFrame(OneHotAge,index=range(len(OneHotAge)))
OneHotLocationDataFrametest=pd.DataFrame(OneHotLocationtest,index=range(len(OneHotLocationtest)))
OneHotBookTypeDataFrametest=pd.DataFrame(OneHotBookTypetest,index=range(len(OneHotBookTypetest)))
OneHotAuthorDataFrametest=pd.DataFrame(OneHotAuthortest,index=range(len(OneHotAuthortest)))
OneHotPublisherDataFrametest=pd.DataFrame(OneHotPublishertest,index=range(len(OneHotPublishertest)))




FinalDatatest=alldatamergetest.drop(columns=['User-ID','ISBN','Location', 'Book-Author','Publisher','BookType'])

FinalDatatest= pd.concat([FinalDatatest,OneHotLocationDataFrametest,OneHotBookTypeDataFrametest,OneHotAuthorDataFrametest,OneHotPublisherDataFrametest],axis=1, ignore_index=True)


tesdmat=xgb.DMatrix(FinalDatatest.values)
y_pred=final_gb.predict(tesdmat)

Mydata=y_pred
for i in range(len(y_pred)):
    Mydata[i]=int(y_pred[i])
np.savetxt('GradientBoost_intOutput.csv', Mydata, delimiter = ',')
np.savetxt('GradientBoost_floatOutput.csv', Predict, delimiter = ',')

#pd.DataFrame(y_pred).to_csv("Output_2.csv")




snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

print("[ Top 10 ]")
for stat in top_stats[:10]:
    print(stat)

end = time.time()
elapsed = end - start
print ("Time taken: ", elapsed, "seconds.")





#regr = RandomForestRegressor(max_depth=2,n_estimators=100,criterion='mse',random_state=1,n_jobs=-1)
#regr.fit(X, y)


