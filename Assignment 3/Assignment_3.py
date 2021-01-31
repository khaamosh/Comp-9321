#!/usr/bin/env python
# coding: utf-8

# In[1978]:


#iteration 4 of the code for Assignment 3

import pandas as pd
import numpy as np
import ast
import operator
import datetime
import json
import operator
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn import linear_model

# In[1979]:


def encode_column(dataframe, col_index,field_name,limit,new_col):
    
    DF = dataframe.iloc[:, :].values
    #print(DF)
    
    #print("in")
    value_dict = {}

    for row in DF:
            #pre-cautionary check for json 
                for json_val in json.loads(row[col_index]):
                    val = json_val[field_name]

                    if val not in value_dict:
                        value_dict[val] =1
                    else:
                        value_dict[val]+=1

        #print(value_dict)

        #now sorting the dict with most frequent values

    encoded = sorted(value_dict.items(),key=operator.itemgetter(1),reverse=True)

        #getting the top most frequent ones from the list
    encoded = encoded[:limit]

    encoded_dict = dict(encoded)

        #print(encode_dict)
    #now since we have the list of the most frequent ones, time to encode them
    
    encoded_list = []
    
    #again working on series
    col_score = []
    
    for row in DF:
        
        encoded_flag = 0
        row_count = 0
        score = 0
        val_count = 0
        
        for json_val in json.loads(row[col_index]):
                val = json_val[field_name]
                
                if val in encoded_dict:
                    score+=1
                    #encoded_flag = 1
                val_count+=1
        #print("Score is :" + str(score))
        
          #count+=1
        col_score.append(score)
        row_count+=1
    
    #print(dataframe.iloc[0])
    #print(col_score[10])
    
    #print(len(col_score))
    
    return dataframe,col_score              
    


# In[1980]:


def count(dataframe,col):
    cnt = []
    
    for values in dataframe[col]:
        if not(pd.isnull(values)):
            cnt.append(len(eval(values)))
        else:
            cnt.append(0)
    
    return cnt 


# In[1981]:


if __name__ == '__main__':
    
    csv_file = 'training.csv'
    validation_file = 'validation.csv'
    
    training_frame = pd.read_csv(csv_file)
    validation_frame = pd.read_csv(validation_file)
    
    training_frame['budget']=training_frame['budget'].replace(0,training_frame['budget'].mean())
    
    validation_frame['budget']=validation_frame['budget'].replace(0,validation_frame['budget'].mean())


# In[1982]:


#we have a lot of null values for homepage
#Converting homepage as binary

training_frame['has_homepage'] = 0
training_frame.loc[training_frame['homepage'].isnull() == False, 'has_homepage'] = 1
validation_frame['has_homepage'] = 0
#validation_frame.loc[validation_frame['homepage'].isnull() == False, 'has_homepage'] = 1

#Homepage v/s Revenue
#sns.catplot(x='has_homepage', y='revenue', data=training_frame);
#plt.title('Revenue being effecte by homepage');


# In[1983]:


#print(training_frame.columns)

#print(len(training_frame.columns))


# In[1984]:


#print(validation_frame.columns)

#print(len(validation_frame.columns))


# In[1985]:


#training_frame.iloc[0]


# In[1986]:


#cast_score for the data

training_frame_copy,cast_score= encode_column(training_frame, 1,"name",1000,"cast_score")
validation_frame_copy,cast_score_v = encode_column(validation_frame, 1,"name",1000,"cast_score")


# In[1987]:


#training_frame.iloc[1]


# In[1988]:


training_frame_copy["cast_score"]  = cast_score


# In[1989]:


validation_frame_copy["cast_score"]  = cast_score_v


# In[1990]:


#training_frame.iloc[1]


# In[1991]:


#encoding the crew frequent 100 ones of the lot
training_frame_copy,crew_t  = encode_column(training_frame_copy, 2,"name",1000,"crew_score")
validation_frame_copy,crew_v = encode_column(validation_frame_copy, 2,"name",1000,"crew_score")

training_frame_copy["crew_score"] = crew_t
validation_frame_copy["crew_score"] = crew_v


# In[1992]:


#encoding the genre of the list
training_frame_copy,genre_t = encode_column(training_frame_copy, 4,"name",100,"genre_score")
validation_frame_copy,genre_v = encode_column(validation_frame_copy, 4,"name",100,"genre_score")

#now encoding the genre
training_frame_copy["genre_score"] = genre_t
validation_frame_copy["genre_score"] = genre_v


# In[1993]:


#genre count is here
#training_frame_copy["genre_score"] = count(training_frame_copy,"genres")
#validation_frame_copy["genre_score"] = count(validation_frame_copy,"genres")


# In[1994]:


#encoding the keywords for movies
training_frame_copy,key_t = encode_column(training_frame_copy, 6,"name",100,"key_score")
validation_frame_copy,key_v = encode_column(validation_frame_copy, 6,"name",100,"key_score")

training_frame_copy["key_score"] = key_t
validation_frame_copy["key_score"] = key_v


# In[1995]:


#keywords is here
#training_frame_copy["key_score"] = count(training_frame_copy,"keywords")
#validation_frame_copy["key_score"] = count(validation_frame_copy,"keywords")


# In[1996]:


#encoding the production companies
training_frame_copy,prod_t = encode_column(training_frame, 10,"name",100,"prod_score")
validation_frame_copy,prod_v = encode_column(validation_frame, 10,"name",100,"prod_score")

training_frame_copy["prod_score"] = prod_t
validation_frame_copy["prod_score"] = prod_v


# In[1997]:


#prod company is here
#training_frame_copy["prod_score"] = count(training_frame_copy,"production_companies")
#validation_frame_copy["prod_score"] = count(validation_frame_copy,"production_companies")


#training_frame_copy,comp_score = count(validation_frame_copy,"production_companies")
#validation_frame_copy,val_score = count(validation_frame_copy,"production_companies")


# In[1998]:


#prod company is here
#training_frame["prod_country"] = count(training_frame,"production_countries")
#validation_frame["prod_country"] = count(validation_frame,"production_countries")


# In[1999]:


#prod company is here
#training_frame["spoken_languages"] = count(training_frame,"spoken_languages")
#validation_frame["spoken_languages"] = count(validation_frame,"spoken_languages")


# In[2000]:


#encoding the spoken languages
training_frame_copy,sp_t = encode_column(training_frame, 15,"name",100,"valid_score")
validation_frame_copy,sp_v = encode_column(validation_frame, 15,"name",100,"valid_score")

training_frame_copy["valid_score"] = sp_t
validation_frame_copy["valid_score"] = sp_v


# In[ ]:





# In[2001]:


#encoding the country for production
training_frame_copy,sp_t = encode_column(training_frame, 11,"name",100,"country_score")
validation_frame_copy,sp_v = encode_column(validation_frame, 11,"name",100,"country_score")

training_frame_copy["country_score"] = sp_t
validation_frame_copy["country_score"] = sp_v


# In[2002]:


#print(len(training_frame_copy.columns))
#print(len(validation_frame_copy.columns))


# In[2003]:


#training_frame_copy.iloc[0]


# In[2004]:


#validation_frame_copy.iloc[0]


# In[2005]:


#setting the log
training_frame_copy['budget']  = np.log1p(training_frame_copy['budget'])
training_frame_copy['revenue']  = np.log1p(training_frame_copy['revenue'])


validation_frame_copy['budget']  = np.log1p(validation_frame_copy['budget'])
validation_frame_copy['revenue']  = np.log1p(validation_frame_copy['revenue'])


# In[2006]:


#Genres v/s revenue
#sns.catplot(x='genre_score', y='revenue', data=training_frame_copy);
#plt.title('Revenue based on Genre');


# In[2007]:


#training_frame_copy['release_date']=pd.to_datetime(training_frame_copy['release_date'])
#validation_frame_copy['release_date']=pd.to_datetime(validation_frame_copy['release_date'])


# In[2008]:


#training_frame_copy['release_date']= training_frame_copy['release_date'].dt.dayofweek 

#validation_frame_copy['release_date']= validation_frame_copy['release_date'].dt.dayofweek 


# In[2009]:


#training_frame_copy['release_date']=training_frame_copy['release_date'].fillna(0)


# In[2010]:


#validation_frame_copy['release_date']=validation_frame_copy['release_date'].fillna(0)


# In[2011]:


#now encode homepage to 0 or 1
training_frame_copy["encode_homepage"] = 0
#training_frame.loc[training_frame["homepage"].isnull() == False,"encode_homepage" = 1]
training_frame_copy.loc[training_frame_copy['homepage'].isnull() == False, 'encode_homepage'] = 1


# In[2012]:


#now encode homepage to 0 or 1
validation_frame_copy["encode_homepage"] = 0
#training_frame.loc[training_frame["homepage"].isnull() == False,"encode_homepage" = 1]
validation_frame_copy.loc[validation_frame_copy['homepage'].isnull() == False, 'encode_homepage'] = 1


# In[2013]:


#now encode homepage to 0 or 1
training_frame_copy["encode_tag"] = 0
#training_frame.loc[training_frame["homepage"].isnull() == False,"encode_homepage" = 1]
training_frame_copy.loc[training_frame_copy['tagline'].isnull() == False, 'encode_tag'] = 1


# In[2014]:


#now encode homepage to 0 or 1
validation_frame_copy["encode_tag"] = 0
#training_frame.loc[training_frame["homepage"].isnull() == False,"encode_homepage" = 1]
validation_frame_copy.loc[validation_frame_copy['tagline'].isnull() == False, 'encode_tag'] = 1


# In[2015]:


#movie revenue depends on day
training_frame_copy['release_date']=   pd.to_datetime(training_frame_copy['release_date'])
validation_frame_copy['release_date']= pd.to_datetime(validation_frame_copy['release_date'])


# In[2016]:


#encoding days to numbers
training_frame['release_day']=training_frame['release_date'].dt.dayofweek 
validation_frame['release_day']=validation_frame['release_date'].dt.dayofweek 


# In[2017]:


training_frame['release_day']=training_frame['release_day'].fillna(0)
validation_frame['release_day']=validation_frame['release_day'].fillna(0)


# In[2018]:


################################################


# In[2019]:


#movie revenue depends on day
#training_frame['release_month']=training_frame['release_date'].dt.month 
#validation_frame['release_month']=validation_frame['release_date'].dt.month 


# In[2020]:


#training_frame['release_month']=training_frame['release_month'].fillna(0)
#validation_frame['release_month']=validation_frame['release_month'].fillna(0)


# In[2021]:


#number of prod companies vs revenue
#sns.catplot(x='prod_score', y='revenue', data=training_frame_copy);
#plt.title('Revenue for different number of production companies in the film');


# In[2022]:


#number of prod countries vs revenue
#sns.catplot(x='prod_country', y='revenue', data=training_frame_copy);
#plt.title('Revenue for different number of production countries in the film');


# In[2023]:


#validation_frame_copy.iloc[0]


# In[2024]:


#removing the columns from this dataset

TFrame = training_frame_copy.drop(["movie_id","cast","crew","genres","homepage","keywords","original_language","original_title","overview","production_companies","production_countries","release_date","spoken_languages","status","tagline","rating"],axis = 1)
VFrame = validation_frame_copy.drop(["movie_id","cast","crew","genres","homepage","keywords","original_language","original_title","overview","production_companies","production_countries","release_date","spoken_languages","status","tagline","rating"],axis = 1)


# In[2025]:


#print(len(training_frame_copy.columns))
#print(len(validation_frame_copy.columns))


# In[2026]:


#training_frame_copy.iloc[0]


# In[2027]:


#validation_frame_copy.iloc[0]


# In[2028]:


#training_frame_copy = training_frame_copy.replace([np.inf, -np.inf], np.nan)
#validation_frame_copy = validation_frame_copy.replace([np.inf, -np.inf], np.nan)

#validation_frame_copy = validation_frame_copy.dropna()
#training_frame_copy = training_frame_copy.dropna()
TFrame['runtime']=TFrame['runtime'].fillna(TFrame['runtime'].mean())
VFrame['runtime']=VFrame['runtime'].fillna(VFrame['runtime'].mean())


# In[2029]:



model = linear_model.LinearRegression()


# In[2030]:



X = TFrame.drop('revenue',axis = 1).values
Y = TFrame['revenue'].values
  
    
    
model.fit(X,Y)

X_valid = VFrame.drop('revenue',axis = 1).values
Y_valid = VFrame['revenue'].values


# In[2031]:


#y_pred = model.predict(X)


# In[2032]:


#from scipy.stats import pearsonr
#pearsonr(Y,y_pred)


# In[2033]:


y_pred = model.predict(X_valid)


# In[2034]:


from scipy.stats import pearsonr
pr = pearsonr(Y_valid,y_pred)

print(pr[0])

# In[2035]:


from sklearn.metrics import mean_squared_error
MSR = mean_squared_error(Y_valid, y_pred)

print(MSR)

y_pred=np.expm1(y_pred)
pd.DataFrame({'movie_id': validation_frame.movie_id, 'predicted_revenue': y_pred}).to_csv('z5269665.Part1.output.csv', index=False)

pd.DataFrame({'zid': 'z5269665', 'MSR': np.expm1(MSR),'corelation':pr}).to_csv('z5269665.Part1.summary.csv', index=False)

# In[2036]:


#training_frame_copy.iloc[0]


# In[2037]:


#from sklearn.metrics import r2_score
#score = r2_score(Y_valid, y_pred)


# In[2038]:


#print(score)


# In[2039]:


#from sklearn.ensemble import GradientBoostingRegressor

#model_gboost = GradientBoostingRegressor()

#model_gboost.fit(X,Y)

#y_pred = model_gboost.predict(X_valid)


# In[2040]:


#y_pred=np.expm1(y_pred)

#from scipy.stats import pearsonr
#pearsonr(Y_valid,y_pred)

#pd.DataFrame({'id': test.id, 'revenue': y_pred}).to_csv('submission_GradientBoosting.csv', index=False)


# In[2041]:


#from sklearn.model_selection import cross_val_score
#from sklearn.ensemble import RandomForestRegressor
#regr = RandomForestRegressor(max_depth=10, min_samples_split=5, random_state=0,
                    #n_estimators=500)
#scores = cross_val_score(regr, X, Y, scoring="neg_mean_squared_error", cv=10)
#rmse_scores = np.sqrt(-scores)
#print(rmse_scores.mean())


# In[2042]:


#regr = RandomForestRegressor()
#regr.fit(X,Y)

#y_pred = regr.predict(X_valid)

#pearsonr(Y_valid,y_pred)

#y_pred = regr.predict(X_valid)


# In[2043]:


#from scipy.stats import pearsonr
#pearsonr(Y_valid,y_pred)


# In[2044]:


#for question 2 of the problem - Rating Prediction of the Problem

TFrame_q2 = training_frame_copy.drop(["movie_id","cast","crew","genres","homepage","keywords","original_language","original_title","overview","production_companies","production_countries","release_date","spoken_languages","status","tagline","revenue"],axis = 1)
VFrame_q2 = validation_frame_copy.drop(["movie_id","cast","crew","genres","homepage","keywords","original_language","original_title","overview","production_companies","production_countries","release_date","spoken_languages","status","tagline","revenue"],axis = 1)


# In[2045]:



X = TFrame_q2.drop('rating',axis = 1).values
Y = TFrame_q2['rating'].values
  
#from sklearn.neighbors import KNeighborsClassifier

#knn = KNeighborsClassifier()
#knn.fit(X, Y)

#X_valid = VFrame_q2.drop('rating',axis = 1).values
#Y_valid = VFrame_q2['rating'].values


# In[2046]:


#from sklearn.metrics import precision_score, accuracy_score, recall_score

#print("confusion_matrix:\n", confusion_matrix(Y_valid, predictions))
#print("precision:\t", precision_score(Y_valid, predictions, average='macro'))
#print("recall:\t\t", recall_score(Y_valid, predictions, average='macro'))
#print("accuracy:\t", accuracy_score(Y_valid, predictions))


# In[2047]:


X = TFrame_q2.drop('rating',axis = 1).values
Y = TFrame_q2['rating'].values
  
from sklearn.ensemble import GradientBoostingClassifier

knn = GradientBoostingClassifier()
knn.fit(X, Y)

X_valid = VFrame_q2.drop('rating',axis = 1).values
Y_valid = VFrame_q2['rating'].values


# In[2048]:



predictions = knn.predict(X_valid)

pd.DataFrame({'movie_id': validation_frame.movie_id, 'predicted_rating': predictions}).to_csv('z5269665.Part2.output.csv', index=False)
#print("accuracy:\t", accuracy_score(Y_valid, predictions))


# In[2049]:


#from sklearn.metrics import precision_score, accuracy_score, recall_score

#print("confusion_matrix:\n", confusion_matrix(Y_valid, predictions))
#print("precision:\t", precision_score(Y_valid, predictions, average='macro'))
#print("recall:\t\t", recall_score(Y_valid, predictions, average='macro'))
#print("accuracy:\t", accuracy_score(Y_valid, predictions))


ps =  precision_score(Y_valid, predictions, average='macro')
rs = recall_score(Y_valid, predictions, average='macro')
ac = accuracy_score(Y_valid, predictions)

ps = np.round(ps)
rs = np.round(rs)
ac = np.round(ac)

#print(type(ps))
pd.DataFrame({'zid': 'z5269665', 'average_precision': ps,'average_recall':rs,'accuracy':ac},index=[0]).to_csv('z5269665.Part2.summary.csv', index=False)

# In[2050]:


#predictions


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




