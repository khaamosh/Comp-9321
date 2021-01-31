import ast
import json
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import numpy as np

studentid = os.path.basename(sys.modules[__name__].__file__)


#################################################
# Your personal methods can be here ...
#################################################

def reading_csv(data):
    return pd.read_csv(data)


def printing_csv(data, col = True, row = True):
    
    if col:
        print(",".join([columns for columns in data]))
        
    if row:
        for index, row in data.iterrows():
            print(",".join([str(row[cols]) for cols in data]))




def log(question, output_df, other):
    print("--------------- {}----------------".format(question))
    if other is not None:
        print(question, other)
    if output_df is not None:
        print(output_df.head(5).to_string())


def question_1(movies, credits):
    """
    :param movies: the path for the movie.csv file
    :param credits: the path for the credits.csv file
    :return: df1
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    
    movies_df = reading_csv(movies)
    credits_df = reading_csv(credits)
    
    df1 = pd.merge(movies_df,credits_df, how = 'left' , left_on=['id'] , right_on=['id'])
    
    #print(df1.col)
    #################################################

    log("QUESTION 1", output_df=df1, other=df1.shape)
    return df1


def question_2(df1):
    """
    :param df1: the dataframe created in question 1
    :return: df2
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    
    df2 = df1[['id', 'title', 'popularity', 'cast', 'crew', 'budget', 'genres', 'original_language', 'production_companies', 'production_countries', 'release_date', 'revenue', 'runtime', 'spoken_languages', 'vote_average', 'vote_count' ]]
     
    #print(df2.columns)
    #################################################

    log("QUESTION 2", output_df=df2, other=(len(df2.columns), sorted(df2.columns)))
    return df2


def question_3(df2):
    """
    :param df2: the dataframe created in question 2
    :return: df3
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    df3 = df2.set_index('id')
    
    #################################################

    log("QUESTION 3", output_df=df3, other=df3.index.name)
    return df3


def question_4(df3):
    """
    :param df3: the dataframe created in question 3
    :return: df4
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    
    df4 = df3[df3['budget']!=0]
    #print(df4['budget'])
    #################################################

    log("QUESTION 4", output_df=df4, other=(df4['budget'].min(), df4['budget'].max(), df4['budget'].mean()))
    return df4


def question_5(df4):
    """
    :param df4: the dataframe created in question 4
    :return: df5
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    
    df4['success_impact'] = df4.apply(lambda row : ((row.revenue - row.budget)/row.budget), axis = 1) 
    
    df5 = pd.DataFrame(df4)
    #################################################

    log("QUESTION 5", output_df=df5,
        other=(df5['success_impact'].min(), df5['success_impact'].max(), df5['success_impact'].mean()))
    return df5


def question_6(df5):
    """
    :param df5: the dataframe created in question 5
    :return: df6
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    
    df5['popularity'] =   ((df5['popularity'] - df5['popularity'].min() ) /  (df5['popularity'].max() - df5['popularity'].min() ))*100
    
    df6 = pd.DataFrame(df5)
    #################################################

    log("QUESTION 6", output_df=df6, other=(df6['popularity'].min(), df6['popularity'].max(), df6['popularity'].mean()))
    return df6


def question_7(df6):
    """
    :param df6: the dataframe created in question 6
    :return: df7
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    df7 = df6.astype({'popularity' : 'int16'})
    #################################################

    log("QUESTION 7", output_df=df7, other=df7['popularity'].dtype)
    return df7


def conversion(x):
    if x!='nan':
        return ast.literal_eval(x)
    

def val(x):
    list_of_characters = []
    
    if isinstance(x,list):
        for i in range(0,len(x)):
            data = x[i]
            
            list_of_characters.append(data['character'])
    
    return ",".join(sorted(list_of_characters))
    

def question_8(df7):
    """
    :param df7: the dataframe created in question 7
    :return: df8
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    df8 = pd.DataFrame(df7)
    
    df8['cast'] = df8['cast'].apply(str)
    
    df8['cast'] = df8['cast'].apply(conversion)
    
    df8['cast'] = df8['cast'].apply(val)

    #df8['cast'] = df8['cast'].apply(str)
    
    
    #################################################

    log("QUESTION 8", output_df=df8, other=df8["cast"].head(10).values)
    return df8



def count_char(x):
    
    te = x.split(',')
    return len(te)


def list_of_movies(x):
    temp = []
    
    for i in range(0,10):
        temp.append(x)
    
    #print(temp)
    return temp
        
def question_9(df8):
    """
    :param df9: the dataframe created in question 8
    :return: movies
            Data Type: List of strings (movie titles)
            Please read the assignment specs to know how to create the output
    """

    #################################################
    # Your code goes here ...
    
    df9 = pd.DataFrame(df8)
    
    #df9['sum_char'] = df9['cast'].apply(lambda length : len(x.split(',')) for x in df8['cast'] )
    df9['sum_char'] = df9['cast'].apply(count_char)
    #print(df9['sum_char'])
    
    res = df9.sort_values('sum_char',ascending=False)
    
    temp = res['title'].head(10)
    
    re = (",").join(temp.values)
    
    movies = re.split(",")
    
    #################################################

    log("QUESTION 9", output_df=None, other=movies)
    return movies


def question_10(df8):
    """
    :param df8: the dataframe created in question 8
    :return: df10
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    
    df8['Date']=pd.to_datetime(df8['release_date'])
    
    df10 = df8.sort_values('Date', ascending = True)
    #################################################

    log("QUESTION 10", output_df=df10, other=df10["release_date"].head(5).to_string().replace("\n", " "))
    return df10


def conversion_2(x):
        return ast.literal_eval(x)

def genre(x):
    list_of_genres = []
    
    if isinstance(x,list):
        for i in range(0,len(x)):
            data = x[i]
            
            list_of_genres.append(data['name'])
    
    return ",".join(list_of_genres)

def question_11(df10):
    """
    :param df10: the dataframe created in question 10
    :return: nothing, but saves the figure on the disk
    """

    #################################################
    # Your code goes here ...
    
    df11 = pd.DataFrame(df10)
    
    df11['genres'] = df11['genres'].apply(str)
    
    df11['genres'] = df11['genres'].apply(conversion_2)
    
    df11['genres'] = df11['genres'].apply(genre)
    
    #df11.genres.str.get_dummies().sum().plot.pie(label='Genre', autopct='%1.0f%%')
    #df11.genres.str.split(',', expand=True).stack().value_counts().plot(kind='pie', label='Genre', autopct = '%1.0f%%')
    values = df11.genres.str.replace('TV Movie' , 'Other Genre')
    values = values.str.replace('Documentary', 'Other Genre')
    values = values.str.replace('Western', 'Other Genre')
    
    values = values.str.split(',' ,expand = True).stack().value_counts()
    #values = df11.genres.str.split(',', expand=True).stack().value_counts()
    
    #values = df11.genres.str.split(',', expand=True).stack()
    values.plot.pie(subplots=True,autopct='%1.0f%%', pctdistance = 0.9,wedgeprops = {'linewidth' : 20})
    #plt.tight_layout()
    
    plt.rcParams["figure.figsize"] = (15,15)
    plt.rcParams["font.size"] = 20
    
    plt.title("Genre")
    plt.ylabel("")
    #################################################

    plt.savefig("{}-Q11.png".format(studentid))
    
    plt.clf()
    
def countries(x):
    countries = []
    
    if isinstance(x,list):
        for i in range(0,len(x)):
            data = x[i]
            #print(data)
            countries.append(data['name'])
            #print(countries)
            
    return countries

def question_12(df10):
    """
    :param df10: the dataframe created in question 10
    :return: nothing, but saves the figure on the disk
    """

    #################################################
    # Your code goes here ...
    
    df12 = pd.DataFrame(df10)
    
    df12['production_countries'] = df12['production_countries'].apply(conversion_2)
    
    df12['production_countries'] = df12['production_countries'].apply(countries)
    
    
    #values = df12.production_countries.str.split(',' , expand =True).stack().value_counts().sort_index()
    
    values = pd.DataFrame(df12["production_countries"].tolist()).stack().reset_index(drop=True)
    #values.set_index('index')
    
    data = values.value_counts().sort_index()
    #key = values.keys().tolist()
#    key = ",".join(key)
#    
#    key = key.split(",")
#    
#    key = sorted(key)
#    
    data.plot.bar(figsize=(15,15) , title = 'production_country')
    #values.plot.bar(values.keys(),values.tolist(),tick_label=values.keys(),figsize=(15,10))
    
    #values.plot.bar(values.keys(),values.tolist(),tick_label=values.keys(),figsize=(15,10))
    #################################################
    
    
    #plt.xticks(rotation=90)
    #plt.xticks(fontsize=11)
    plt.tight_layout()
    #plt.rcParams["figure.figsize"] = (15,15)
    
    plt.savefig("{}-Q12.png".format(studentid))
    

#
#def langs(x):
#    
#    lang = []
#    
#    if isinstance(x,str):
#        temp = x.split(",")
#        if len(temp) > 1:
#            lang.append(temp[0])

def question_13(df10):
    """
    :param df10: the dataframe created in question 10
    :return: nothing, but saves the figure on the disk
    """

    #################################################
    # Your code goes here ...
    
    df13 = pd.DataFrame(df10)
    
    #en_df = df13.query('original_language == "en"')
    
    #df13.original_language.apply(lambda x : ast.literal_eval(x))
    list_of_language = list(set(df13.original_language.tolist()))
    
    rng = np.random.RandomState(0)
    
    red   = np.random.rand(len(list_of_language)+1)
    blue  = np.random.rand(len(list_of_language)+1)
    green = np.random.rand(len(list_of_language)+1)   

    
    res = 0
    temp = 0.1
    
#    for lang in list_of_language:
#        #colors = np.random.randint(low=1, high = 99)/100
#        query = 'original_language == ' + '"' + lang + '"' 
#        lang_df = df13.query(query)
#            
#        if res == 0:
#            ax = lang_df.plot.scatter(x = "vote_average" , y = "success_impact", c=(red[res],blue[0],green[res])  )
#            res+=1
#        #temp = temp + 0.01
#        ax = lang_df.plot.scatter(x = "vote_average" , y = "success_impact", c= (red[res],blue[res],green[0]), ax = ax )
#        res+=1
        
    for l in list_of_language:
        #colors = np.random.randint(low=1, high = 99)/100
        query1 = 'original_language == ' + '"' + l + '"'
        #query2 = 'original_language == ' + '"' + list_of_language[l+1] + '"'
        
        lang_df_1 = df13.query(query1)
        #lang_df_2 = df13.query(query2)
        
        if res == 0:
            ax = lang_df_1.plot.scatter(x = "vote_average" , y = "success_impact", c= np.array( [ red[res] , blue[res] , green[res] ] )  )
            res+=1
        #temp = temp + 0.01
    
        ax = lang_df_1.plot.scatter(x = "vote_average" , y = "success_impact", c=  np.array( [red[res] , blue[res] , green[res]]   ), ax = ax )
        res+=1    



    
#    for l in range(0,len(list_of_language),2):
#        #colors = np.random.randint(low=1, high = 99)/100
#        query1 = 'original_language == ' + '"' + list_of_language[l] + '"'
#        query2 = 'original_language == ' + '"' + list_of_language[l+1] + '"'
#        
#        lang_df_1 = df13.query(query1)
#        lang_df_2 = df13.query(query2)
#        
#        if res == 0:
#            ax = lang_df_1.plot.scatter(x = "vote_average" , y = "success_impact", c= np.array( [red[res], blue[res] , green[res]] )  )
#            res+=1
#        #temp = temp + 0.01
#    
#        ax = lang_df_2.plot.scatter(x = "vote_average" , y = "success_impact", c= np.array( [red[res], blue[res] , green[res] ] ), ax = ax )
#        res+=1
    #df13.plot.scatter(x = "vote_average" , y = "success_impact")
    
    
    #################################################
    plt.legend(list_of_language)
    plt.savefig("{}-Q13.png".format(studentid))


if __name__ == "__main__":
    df1 = question_1("movies.csv", "credits.csv")
    df2 = question_2(df1)
    df3 = question_3(df2)
    df4 = question_4(df3)
    df5 = question_5(df4)
    df6 = question_6(df5)
    df7 = question_7(df6)
    df8 = question_8(df7)
    movies = question_9(df8)
    df10 = question_10(df8)
    question_11(df10)
    question_12(df10)
    question_13(df10)
