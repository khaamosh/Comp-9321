# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 23:25:44 2020

@author: Uttkarsh Sharma
"""

COLLECTION = 'collection'

import pandas as pd

import sqlite3
from sqlite3 import Error

from flask import Flask
from flask_restplus import Resource, Api
from flask_restplus import fields
from flask import request
import datetime
import requests
import json
from flask_restplus import reqparse
from flask_restplus import inputs

app = Flask(__name__)
api = Api(app)

#this method call is not required

#indicator_model for the api
#indicator_model = api.model('indicators', {
#    'indicator_id': fields.String(Required=True)
#})


#parser for the indicators
parser = reqparse.RequestParser()
parser.add_argument('indicator_id')

#parser for order by values
parser_get = reqparse.RequestParser()
parser_get.add_argument('order_by',action='split')

#parser for the q parameter
parser_top_bottom = reqparse.RequestParser()
parser_top_bottom.add_argument('q')

# sample working URL is as follows :
# http://api.worldbank.org/v2/countries/all/indicators/ NY.GDP.MKTP.CD ?date=2012:2017&format=json&per_page=1000 

#this method will construct the URL and then return it 

def URL_construction(indicator_val, date ="2012:2017",page=1):
    
    URL = "http://api.worldbank.org/v2/countries/all/indicators/" + str(indicator_val) + "?date=" +str(date) + "&format=json&per_page=" +str(page)
    
    return URL


def URL_query(value_indicator,page=1000,pagination=[],max_pages=2):
    
    #print("inside-query")
    if not value_indicator:
        return 'None value indicator'
        
    url = URL_construction(indicator_val=value_indicator,page=page)
    #print(url)
    req = requests.get(url).json()
    #print("request is here")
    #print(req)
    
    if( len(req)<=1 and req[0]['message'][0]['key'] == 'Invalid value'):
        return 'Invalid value indicator'
    
    if req[0]['page']>=max_pages or req[0]['page'] == req[0]['pages']:
        return pagination+req[1]
    
    return URL_query(value_indicator=value_indicator, page=page+1000,pagination=pagination+req[1],max_pages=max_pages,)
        
    
    
#this function is responsible for parsing of the data in the required format
    
def data_parser(data):
    data_set = {
            'country':data['country']['value'],
            'date':data['date'],
            'value':data['value'],
            }
    
    return data_set
    

#for inserting the data in the table
    
def insert_row(conn, values):
    
    query = ''' INSERT INTO indicators(indicator,value,time,entries)
              VALUES(?,?,?,?) '''
     
    try:
         cur = conn.cursor()
         cur.execute(query, values)
         conn.commit()
    except Exception as e:
        
        print("error-inside_insert")
        print(e)
    return cur.lastrowid


@api.route(f'/{COLLECTION}', endpoint=COLLECTION)
#@api.route('/indicators/')
class get_collection_from_url(Resource):
    
   
    #@api.expect(indicator_model)
    @api.doc(description='Question 1: Imporing collection from world bank api')
    
    @api.response(200, 'Successfully retrieved collection.')
    @api.response(201, 'Successfully created collection.')
    @api.response(400, 'Bad Request sent to the backend')
    
    
    
    @api.expect(parser)
    
    def post(self):
        
        args = parser.parse_args()
        #payload = request.json
        
        #print(payload)
        #checking for the payload of the data and making sure that id is not empty
        if not args.get('indicator_id'):
            return {'message' : 'Request you to provide value for indicator id - Cannot be empty'},400
        
        
        
        try:
            #df = pd.read_sql_query("Select id,indicator from indicators where indicator=" + str(payload['indicator_id']),db_conn)
            db_conn = recur_connection("Collections.db")
            
            conversion = args.get('indicator_id')
            conversion = conversion.replace('.','_').strip()
            
            #print(conversion)
            df = pd.read_sql_query("Select id,indicator,time from indicators where indicator='" + conversion + "'",db_conn)
            #print("Select id,indicator,time from indicators where indicator='" + conversion + "'")
            
            #df = pd.read_sql_query(r"Select id,indicator from indicators where indicator=NY'.GDP'.MKTP'.CD" ,db_conn)
            #df = pd.read_sql_query("Select * from indicators",db_conn)
            #print(df)
            
            if not df.empty :
                #print(df)
                
                id_val = df['id'].to_string(header=False,index=False)
                indicator_id_val = df['indicator'].to_string(header=False,index=False)
                
                indicator_id_val = indicator_id_val.replace('_', '.')
                
                return {
                        'uri':f'/{COLLECTION}/{id_val}',               
                        'id': id_val.strip(),
                        'creation_time': df['time'].to_string(header=False,index=False).strip(),
                        'indicator_id': indicator_id_val.strip(),
                        },201
                        
                db_conn.close()
            
            else:
                
                data = URL_query(value_indicator = args.get('indicator_id'),page=1000)
            
                #print("data is fetched")
                #now we will check the status of the data
                if(data == 'None value indicator'):
                    return {'message' : 'Request you to provide value for indicator id - Cannot be empty'},400
                elif(data == 'Invalid value indicator' ):
                    return {'message' : 'Invalid indicator Id was given'},400
                
                #parse the data in the required format
                data_values = [data_parser(d) for d in data]
                
                data_val_json = json.dumps(data_values)
                #print(data_val_json)
                
                #print("data is done")
                #print(data_values)
                
                #the data to be entered inside the table
                
                indi = data[0]['indicator']['id'].replace('.','_')
                
                val = (indi,
                       data[0]['indicator']['value'],
                       str(datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")),
                        data_val_json)
                
                #print(val)
                
                result = insert_row(db_conn,val) 
                
                if result:
                    
                    #print(result)
                    #print("YAY!! added the data")
                    
                    response_df = pd.read_sql_query("Select id,indicator,time from indicators where id=" + str(result),db_conn)
                    #print(response_df)
                    
                    resp_id_val = response_df['id'].to_string(header=False,index=False)
                    #print(resp_id_val)
                    
                    resp_indicator_id_val = response_df['indicator'].to_string(header=False,index=False)
                    #print(resp_indicator_id_val)
                    
                    indicator_id_val = resp_indicator_id_val.replace('_', '.')
                
                    return {
                            'uri':f'/{COLLECTION}/{resp_id_val}',               
                            'id': resp_id_val.strip(),
                            'creation_time': response_df['time'].to_string(header=False,index=False).strip(),
                            'indicator_id': resp_indicator_id_val.strip(),
                        },201
                db_conn.close()
                 
            
        except :
            #print(error)
            # if we are here then we will create a URL and send for getting the information.
            #print(error)
            #print("handling the error")
            return {'message' : 'There were invalid parameters provided'},400
            
        #cursor.execute("Select * from information where indicator=",payload['indicator_id'])
    

    @api.response(200, 'Successfully retreieved collections.')
    @api.response(400, 'Bad request sent to the backend.')    
    
    @api.doc(description='Question 3 - Fetching the  list of available collections' )
    
    
    @api.expect(parser_get)    
    
    def get(self):
        
        try:
            
            args = parser_get.parse_args()
        
            arg_set = {"+id","+creation_time","+indicator","-id","-creation_time","-indicator"}
            db_conn = recur_connection("Collections.db")
            
            #check that arguments are not empty
            #print(args['order_by'])
            #print(len(args))
            
            #checking if the order given is not empty or can be given as required,need to check
            if len(args['order_by'])==1:   
                try:
                    len(args['order_by'])
                except:
                    return{ 'message' : 'You did not supply any order man!!'},400
            
            #check that correct arguments were sent to the API
            for arg in args['order_by']:
                if arg not in arg_set:
                    return{ 'message' : 'Amigo! you sent an invalid order type'},400
            
            orderby_list = []
            
            #print(args['order_by'])
            for arg in args['order_by']:
                #print(arg)
                order = arg[0:1]
                #print(order)
                column = arg[1:]
                #print(column)
                
                column = column.replace('creation_time','time')
                
                asc_desc = 'ASC' if order == '+' else 'DESC'
                temp_string = column + " " + asc_desc
                orderby_list.append(temp_string)
                
                #temp_result = pd.read_sql_query("Select id,time,indicator from indicators order by )
            #print("The string is as follows\n")
            #print(",".join(orderby_list))
            
            order = ",".join(orderby_list)
            
            temp_result = pd.read_sql_query("Select id,time,indicator from indicators order by " + order,db_conn )
            
            
            change = pd.Series(temp_result['id'])
            change = change.apply(lambda x : '/collections/' + str(x))
            
            temp_result.insert(loc=0,column='uri',value=change)        
            resp = temp_result.to_dict('record')
    #        print(args)
            
            return resp,200
        
        except:
            return {'message' : 'There were invalid parameters provided'},400
        
########################################################################################
#here is the code for the second question , deletino the data collection
@api.route(f'/{COLLECTION}/<int:collection_id>')
class deletion_of_collection_using_id(Resource):
    
    @api.response(200, 'Successfully removed collection.')
    @api.response(404, 'Unable to find collection.')
    @api.response(400, 'Bad request sent to the service.')
    
    
    
    @api.doc(description='Question 2 : Deletion of the collection from the provided id.')
    
    
    def delete(self,collection_id):
        
        #this check is redundant
        if not collection_id:
            return {'message' : 'Request you to provide value for collection id - Cannot be empty'},400
        
        
        
        #print("inside-check")
        
        #now we willcheck if the id exists in the database or not
        #val = db_conn.execute( 'Select indicator from indicators where id=' + str(collection_id))
        #print(val.fetchone())
        
        
        
        #now we have checked id is present hence delete it
        try:
            
            #getting the connection to the data base
            db_conn = recur_connection("Collections.db")
            
            if not db_conn.execute( 'Select indicator from indicators where id=' + str(collection_id)).fetchone():
                return {'message' : 'Id not present in collection'},404
            
            db_conn.execute('Delete from indicators where id =' + str(collection_id))
            db_conn.commit()
            
            db_conn.close()
            
            return {'message' : f'The collection {collection_id} was removed from the database!'},200
        except:
            
            db_conn.close()
            return {'message': 'Apologies! collection could not be removed'},400
            
    
    @api.response(200, 'Successfully retreived collection.')
    @api.response(404, 'Bad request sent to the backend service.')
    
    
    
    @api.doc(description='Question 4 - Fetching the collection from collection id.')
   
    
    def get(self,collection_id):
        
        #this check is redundant
        #print("inside")
        if not collection_id:
            return {'output' : 'Request you to provide value for collection id - Cannot be empty'},400
        
        #getting the connection to the data base
        db_conn = recur_connection("Collections.db")
        
        temp_df = pd.read_sql_query("Select * from indicators where id=" + str(collection_id),db_conn)
         
        if not temp_df.empty:
             resp_id_val = temp_df['id'].to_string(header=False,index=False)
             #print(resp_id_val)
                    
             resp1_indicator_id_val = temp_df['indicator'].to_string(header=False,index=False)
             #print(resp1_indicator_id_val)
                    
             indicator_id_val = resp1_indicator_id_val.replace('_', '.')
             
             test = temp_df['entries'].apply(json.loads).iloc[0]
             #print(test)
             
             
             db_conn.close()   
             return {
                            'uri':f'/{COLLECTION}/{resp_id_val}',               
                            'id': indicator_id_val,
                            'creation_time': temp_df['time'].to_string(header=False,index=False),
                            'indicator_id': resp1_indicator_id_val,
                            'entries':temp_df['entries'].apply(json.loads).iloc[0],
                    },200
            
        else:
             return {'output' : 'Id not present in collection'},400
        
        
#################################################################################
            
#now we will work on the fifth question



###############################################################################
@api.route(f'/{COLLECTION}/<int:collection_id>/<int:year>/<string:country>',endpoint="COLLECTION_by_couuntry")
class selection_of_collection_using_id_year_country(Resource):
    
    @api.response(200,'Successfull retrieval of data.')
    @api.response(400,'Bad request sent to the data service.')
    @api.response(404,'Collection not found.')
    
    @api.doc(description='Question 5 - Retrieving indicator for given id,year,country.')
    def get(self,collection_id,year,country):
        
        try:
            
            db_conn = recur_connection("Collections.db")
        
            temp_df = pd.read_sql_query("Select * from indicators where id=" + str(collection_id),db_conn)
            #print(temp_df.head(2))
            
            
            if year < 2012 or year > 2017:
                return {'message':'The given year is not within 2012 and 2017'},400
            
            resp1_indicator_id_val = temp_df['indicator'].to_string(header=False,index=False)
            #print(resp1_indicator_id_val)
                        
            indicator_id_val = resp1_indicator_id_val.replace('_', '.')
            
            print("error")
            
            if not temp_df.empty:
               
               test = temp_df['entries'].apply(json.loads).iloc[0]
               #print(test.head(2))
               
               filtered = [val for val in test if val['country'] == country and val['date'] == str(year)]
              
                     
               if len(filtered) == 0:
                   #print("###")
                   return {'message' : f'Could not locate the data based on given country - {country} and year - {year}'},400
               
               return {
                       'id':collection_id,
                       'indicator':indicator_id_val.strip(),
                       'country':country.strip(),
                       'year':year,
                       'value':filtered[0]['value'],
                       },200
               
            else:
                return{'message': 'The values-inside-else given are incorrect'},400
        except Exception as e:
            print(e)
            return{'message': 'The values given are incorrect'},400
           
             
###############################################################################


#now for the last question
@api.route(f'/{COLLECTION}/<int:collection_id>/<int:year>')
class country_top_bottom(Resource):
    
    @api.response(200,'Successfully retrieved top/bottom economic indicator.')
    @api.response(404,'Unable to find collection')
    
        
    @api.doc(description='Question 6 - Retrieve top/bottom economic indicator values.')
    
    
    @api.expect(parser_top_bottom)
    
    
    def get(self,collection_id,year):
        
        order = request.args.get('q')
        
        try:
            db_conn = recur_connection("Collections.db")
        
            temp_df = pd.read_sql_query("Select * from indicators where id=" + str(collection_id),db_conn)
            
            resp1_indicator_id_val = temp_df['indicator'].to_string(header=False,index=False)
            #print(resp1_indicator_id_val)
            
            resp1_val = temp_df['value'].to_string(header=False,index=False)
            #print(resp1_val)
                        
            indicator_id_val = resp1_indicator_id_val.replace('_', '.')
            
            if not temp_df.empty:
                
                test = temp_df['entries'].apply(json.loads).iloc[0]
                filtered = [val for val in test if val['date'] == str(year)]
                
                none_filtered = [val for val in filtered if val['value']!=None]
                
                res = pd.DataFrame(none_filtered)
                res.drop_duplicates(inplace=True)
                
            
            #print(none_filtered)
            
            #checking if the query parameter was empty or not, if empty return the values as is
            
            if not order:
                
                none_filtered = res.to_dict('record')
                return {
                        'indicator':indicator_id_val.strip(),
                        'value':resp1_val.strip(),
                        'entries':none_filtered[0:100],
                        }, 200
            
        
            
            #length = len(order)
            if order[0] == '+':
                if(  int(order[1:]) >=1 and int(order[1:]) <=100):
                    res.sort_values(by='value',ascending=False,inplace=True)
                    out = res.head(int(order[1:]))
                    
                
            elif order[0] == '-':
                if(  int(order[1:]) >=1 and int(order[1:]) <=100):
                    res.sort_values(by='value',ascending=True,inplace=True)
                    out = res.tail(int(order[1:]))
            else:
                if(  int(order) >=1 and int(order) <=100):
                    res.sort_values(by='value',ascending=False,inplace=True)
                    out = res.head(int(order))
            
            none_filtered = out.to_dict('record')
            
            return{
                    'indicator':indicator_id_val.strip(),
                    'value':resp1_val.strip(),
                    'entries': none_filtered, 
                },200
        
        except:
            return{'message':'Unable to find the collection'},404
            
        
        
        #now we would be given the query parameter, and hence making the descision to filter
    
##############################################################################


            
        
##############################################################################
def create_connection(db_file):
    
    """ create a database connection to a SQLite database """
    conn = None
    
    sql_create_indicator_table = """
    
    CREATE TABLE IF NOT EXISTS indicators (
    
    id integer PRIMARY KEY AUTOINCREMENT,
    indicator text UNIQUE,
    value text NOT NULL,           
    time text NOT NULL,
    entries text NOT NULL
    );

    """
    try:
        conn = sqlite3.connect(db_file)
        #print(sqlite3.version)
        
        c = conn.cursor()
        c.execute(sql_create_indicator_table)
        
    except Error as e:
        print(e)
    finally:
        if conn:
            #create_table(conn,sql_create_indicator_table)
            conn.close()

def recur_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    
    return conn
    
if __name__ == '__main__':
    
    #initialize the database and set the schema of the database if database is not created.
    create_connection("Collections.db")
    app.run(debug=True)