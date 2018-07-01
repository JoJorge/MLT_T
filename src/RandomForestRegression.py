import time
import tracemalloc

import math
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import csv as csv
import argparse

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

data_path = '../data/'
Submission_filename=data_path + "book_ratings_test.csv"
Raw_Book_filename=data_path + "books_pre_most.csv"
User_filename=data_path + "users_DT.csv"
Rate_filename=data_path + "book_ratings_train.csv"
Book_filename=data_path + "BookClassificationResult_onlyTitle.csv"

parser = argparse.ArgumentParser()
parser.add_argument('--t2', action="store_true")
args = parser.parse_args()

tracemalloc.start()
start = time.time()
print('Reading....')
BookData=pd.read_csv(Book_filename)
RateData=pd.read_csv(Rate_filename)
UserData=pd.read_csv(User_filename)
Raw_book=pd.read_csv(Raw_Book_filename)

print('Preprocessing....')
towmerge=pd.merge(RateData, UserData, how='left', on=['User-ID'])
threedatamerge=pd.merge(towmerge, BookData, how='left', on=['ISBN'])
threedatamerge.drop(columns=['BookTitle', 'BookAuthor','BookPublisher'],inplace=True)
alldatamerge=pd.merge(threedatamerge,Raw_book, how='left', on=['ISBN'])

###Check none or nan
alldatamerge['Book-Author'].fillna(value='unknown',inplace=True)
alldatamerge['Publisher'].fillna(value='unknown',inplace=True)
alldatamerge['Year-Of-Publication'].fillna(value=-1,inplace=True)


##label and one hot

lebal_Location=LabelEncoder()
alldatamerge['Location']=lebal_Location.fit_transform(alldatamerge['Location'])

lebal_Author=LabelEncoder()
alldatamerge['Book-Author']=lebal_Author.fit_transform(alldatamerge['Book-Author'])

lebal_Publisher=LabelEncoder()
alldatamerge['Publisher']=lebal_Publisher.fit_transform(alldatamerge['Publisher'])


OneHotLocation=np.zeros((len(alldatamerge),len(np.unique(alldatamerge['Location']))))
OneHotAuthor=np.zeros((len(alldatamerge),len(np.unique(alldatamerge['Book-Author']))))
OneHotPublisher=np.zeros((len(alldatamerge),len(np.unique(alldatamerge['Publisher']))))


for i in range(len(alldatamerge)):
    
    OneHotLocation[i][int(alldatamerge['Location'][i])]=1
    OneHotAuthor[i][int(alldatamerge['Book-Author'][i])]=1
    OneHotPublisher[i][int(alldatamerge['Publisher'][i])]=1
    

OneHotLocationDataFrame=pd.DataFrame(OneHotLocation,index=range(len(OneHotLocation)))
OneHotAuthorDataFrame=pd.DataFrame(OneHotAuthor,index=range(len(OneHotAuthor)))
OneHotPublisherDataFrame=pd.DataFrame(OneHotPublisher,index=range(len(OneHotPublisher)))


FinalData=alldatamerge.drop(columns=['Location', 'Book-Author','Publisher','BookType'])

FinalData= pd.concat([FinalData,OneHotLocationDataFrame,OneHotAuthorDataFrame,OneHotPublisherDataFrame],axis=1, ignore_index=True)

X=FinalData.iloc[:,3:].values
y=FinalData.iloc[:,2].values


print('Training....')
regr = RandomForestRegressor(n_estimators=10, criterion='mae', max_depth=2, random_state=1, n_jobs=-1)
if args.t2:
    print('T2')
    sample_weights = []
    for i, row in RateData.iterrows():
        sample_weights.append(1.0 / row['Book-Rating'])
    regr.fit(x, y, sample_weights)
else:
    print('T1')
    regr.fit(X, y)


####test
print('Test data preprocessing....')
TestData=pd.read_csv(Submission_filename)


towmergetest=pd.merge(TestData, UserData, how='left', on=['User-ID'])
threedatamergetest=pd.merge(towmergetest, BookData, how='left', on=['ISBN'])
threedatamergetest.drop(columns=['BookTitle', 'BookAuthor','BookPublisher'],inplace=True)
alldatamergetest=pd.merge(threedatamergetest,Raw_book, how='left', on=['ISBN'])


###Check none or nan
alldatamergetest['Book-Author'].fillna(value='unknown',inplace=True)
alldatamergetest['Publisher'].fillna(value='unknown',inplace=True)
alldatamergetest['Year-Of-Publication'].fillna(value=-1,inplace=True)


##label and one hot

lebal_Location_test=LabelEncoder()
alldatamergetest['Location']=lebal_Location_test.fit_transform(alldatamergetest['Location'])

lebal_Author_test=LabelEncoder()
alldatamergetest['Book-Author']=lebal_Author_test.fit_transform(alldatamergetest['Book-Author'])

lebal_Publisher_test=LabelEncoder()
alldatamergetest['Publisher']=lebal_Publisher_test.fit_transform(alldatamergetest['Publisher'])


OneHotLocationtest=np.zeros((len(alldatamergetest),len(np.unique(alldatamergetest['Location']))))
OneHotAuthortest=np.zeros((len(alldatamergetest),len(np.unique(alldatamergetest['Book-Author']))))
OneHotPublishertest=np.zeros((len(alldatamergetest),len(np.unique(alldatamergetest['Publisher']))))



for i in range(len(alldatamergetest)):
    
    OneHotLocationtest[i][int(alldatamergetest['Location'][i])]=1
    OneHotAuthortest[i][int(alldatamergetest['Book-Author'][i])]=1
    OneHotPublishertest[i][int(alldatamergetest['Publisher'][i])]=1

    
    

OneHotLocationDataFrametest=pd.DataFrame(OneHotLocationtest,index=range(len(OneHotLocationtest)))
OneHotAuthorDataFrametest=pd.DataFrame(OneHotAuthortest,index=range(len(OneHotAuthortest)))
OneHotPublisherDataFrametest=pd.DataFrame(OneHotPublishertest,index=range(len(OneHotPublishertest)))




FinalDatatest=alldatamergetest.drop(columns=['User-ID','ISBN','Location', 'Book-Author','Publisher','BookType'])

FinalDatatest= pd.concat([FinalDatatest,OneHotLocationDataFrametest,OneHotAuthorDataFrametest,OneHotPublisherDataFrametest],axis=1, ignore_index=True)


print('Predicting....')
Predict=regr.predict(FinalDatatest.values)
Mydata=Predict
for i in range(len(Predict)):
    Mydata[i]=int(round(Predict[i]))
np.savetxt(data_path + 'RandomForestRegression_intOutput.csv', Mydata, fmt = '%d', delimiter = ',')
np.savetxt(data_path + 'RandomForestRegression_floatOutput.csv', Predict, delimiter = ',')

#predict=pd.DataFrame(Predict)
#predict.to_csv("Output4.csv")



snapshot = tracemalloc.take_snapshot()

top_stats = snapshot.statistics('lineno')

print("[ Top 10 ]")
for stat in top_stats[:10]:
    print(stat)

end = time.time()
elapsed = end - start
print ("Time taken: ", elapsed, "seconds.")

