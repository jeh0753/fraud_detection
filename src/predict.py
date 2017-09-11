import pandas as pd
import cPickle as pickle
import numpy as np
from model import *
from pymongo import MongoClient
import json

class Model(object):
    '''Uses our model created in model.py, use this to predict on a single data point. Can handle both pandas dataframe and json formatted inputs with their respective load functions. First run model.load_json_data, then use model.predict to get predicted value or insert_to_mongo to both predict and insert into a mongo db
    '''
    def __init__(self):
        '''
        Loads model from model.pkl.
        ***You need to run model.py before using this or there won't be a model.pkl.
        '''
        with open('model.pkl') as f:
            self.model = pickle.load(f)
    def load_json_data(self,path):
        '''Use to predicting on json formatted inputs.
        Args:
            path (str): file path of json input
        Returns:
            None
        '''
        self.df = pd.read_json(path)

    def load_pandas_data(self,df):
        '''Use when predicting on pandas formatted inputs.
        Args:
            df (pandas df): df input
        Returns:
            None
        '''
        self.df = df

    def predict(self):
        '''Get prediction on previously loaded data
        Args:
            None
        Returns:
            bool: True if predicted fraud, False otherwise
        '''
        return self.model.predict(self.df)


    def insert_to_mongo(self, db_name='predictions', table_name='preds'):
        '''Compute prediction, add that as a column to the input data and insert the result into mongo db
        Args:
            db_name (str): database name to insert into
            table_name (str): database name to insert into
        Returns:
            None
        '''
        client = MongoClient()
        db = client[db_name]
        tab = db[table_name]
        self.df['prediction'] = self.predict()
        j = self.df.to_json()
        parsed = json.loads(j)
        tab.insert_one(parsed)
        client.close()



if __name__ == '__main__':
    model = Model()
    path = 'files/example.json'
    model.load_json_data(path)
    prediction = model.predict()
    model.insert_to_mongo()
