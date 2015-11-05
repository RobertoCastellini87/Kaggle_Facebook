# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 21:08:23 2015

@author: roberto
"""

# Problem introduction

# I wrote the following program to give a solution (which is onl partial)
# to a Kaggle competition problem.

# A company offers the possibility to partecipate to online auctions. There 
 # are also some bots who are parteciping to this auctions, so that the
# human users do not have a satisfying experience. Because of the frustation
# cause by the robots presence, many of them stop partecipating at the #
# auctions. The company wants then to have an algorithm capable of recognizing robots. 



import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

#We import the data to two different files, train_file and bids_file. 
#I do not load the test file, because I can not really use it. In fact,
# the outcomes are not present, and it is too late for submission.

train_file = pd.read_csv('/home/roberto/Kaggle_Facebook/train.csv')
bids_file = pd.read_csv('/home/roberto/Kaggle_Facebook/bids.csv')

#I look at the percentage of the robots in the file. They are around the 5%.
print 'The percentage of the robots in the file is:', (float(len(train_file[train_file['outcome'] == 1.0]))/len(train_file))*100, '%.'


#I defin a dataframe for aggregated bid data and I sort it 
#by auctions and time.

bids = pd.DataFrame(data = train_file['bidder_id'], columns = ['bidder_id'])
bids_sorted = bids_file.sort(['auction', 'time'], ascending = [True, True])

#I initialize a new file bids_aggr, in which I will consider the bids
# and the aggregated data.

bids_aggr = pd.DataFrame(data = bids_sorted['bidder_id'].unique(), columns = ['bidder_id'],
                    index = bids_sorted['bidder_id'].unique())
                    
# I consider a new variable time, such that the minmum time is 0. I calculate
#then the differences between different times.

bids_sorted['time'] = bids_sorted['time'] - bids_sorted['time'].min()
time_difference = bids_sorted.groupby('bidder_id')['time'].diff()

#I count the number of different bids for user and I add it at the data 
#aggregator.

counts = bids_sorted.groupby('bidder_id')['bidder_id'].agg('count')
bids_aggr['bid_count'] = counts

#I consider now the number of auctions for each user.

auctions_count = bids_sorted.groupby('bidder_id')['auction'].unique()
auctions_count_num = []
for i in range(0,len(auctions_count)):
    auctions_count_num.append(float(auctions_count[i].count()))
bids_aggr['auctions_count'] = auctions_count_num

#I also consider the number of different countries.
countries_count = bids_sorted.groupby('bidder_id')['country'].unique()
countries_count_num = []
for i in range(0,len(countries_count)):
    countries_count_num.append(countries_count[i].count())
bids_aggr['countries_count'] = countries_count_num
#I now compute the time differences, on the file that we sorted by time,
# and I add it as a new column.

time_difference = bids_sorted.groupby('bidder_id')['time'].diff()
bids_sorted['time_difference'] = time_difference

#I use this information now to compute, for every bidders, what is 
# the maximal difference in time between the biddings, the minimum difference
# and the mean difference. The idea is that a robot and a human have a different
# behaviour with rapport with the time. 

max_diff = bids_sorted.groupby('bidder_id')['time_difference'].max()
bids_aggr['max_diff'] = max_diff

min_diff = bids_sorted.groupby('bidder_id')['time_difference'].min()
bids_aggr['min_diff'] = min_diff

median_diff = bids_sorted.groupby('bidder_id')['time_difference'].median()
bids_aggr['median_diff'] = median_diff

#I now aggregate the file obtained with the train data. We are going to lose
#informations, because the users are splitted in the train and test file.

bids_data = train_file.merge(bids_aggr, left_on ='bidder_id', right_on = 'bidder_id')

#I decide to eliminate the data of all users doing only one bid. This is 
#because for the data I chose to analyse, a single bid is not significative.
#Moreover, I am not sure that it is possible to spot a bot just by looking at 
#a single bid. I am more likely going to eliminate some noise and focus on the
#important things.

bids_data_most = bids_data
bids_data_most = bids_data_most[bids_data_most.bid_count > 1]

#I split my train test in two and I apply the RandomForest method to the train
#file. 

train, test = train_test_split(bids_data_most, test_size = 0.2)

train_outcome = train['outcome']
test_outcome = test['outcome'].values

print 'The percentage of robots in the train test is:', ((len(train[train['outcome'] == 1.0])/ float(len(train)))*100), '%'
#I compute, for control, what percentage of robots are in the test. Then I drop,
# all the data  I do not need for the machine learning method.

train_dropped = train.drop(['outcome', 'bidder_id', 'address', 'payment_account'], axis = 1) 
 
#I start the machine learning method. 
 
clf = RandomForestClassifier(n_estimators=25)

clf.fit(train_dropped, train_outcome)
clf.score(train_dropped,train_outcome)
#I prepare the test data for the prediction.
test_dropped = test.drop(['outcome', 'bidder_id', 'address', 'payment_account'], axis = 1) 

prediction =clf.predict(test_dropped)

# I compute the percentage of good predictions.

j = 0
for i in range(0, len(prediction)):
    if prediction[i] == test_outcome[i]:
        j=j+1

print 'The percentage of good predictions is: ', (float(j)/len(prediction))*100, '%.'

#We obtain more than 93% on the test. This can appear good, but it is not. 
#In fact, the number of users that are robots is 6%. So that an algorithm that
# chooses its member by chance would be only slightly worst 
# than the one we built.


