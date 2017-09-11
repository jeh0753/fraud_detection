from bs4 import BeautifulSoup
from predict import Model
from model import *
from flask import Flask, request, redirect, url_for, render_template
from flask import request
from predict import Model
import json
import urllib2
import pandas as pd
import requests
import os
import random


UPLOAD_FOLDER = os.getcwd()+"/uploads"
ALLOWED_EXTENSIONS = set(['json'])
model = Model()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    '''Checks to see if file ends in .json
    Args:
        filename (str): name of file
    Returns:
        bool: True if ends in .json, False otherwise
    '''
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def get_facts():
    return random.choice(open('static/FrogFacts.txt').readlines())

@app.route('/', methods=['GET','POST'])
def api_root():
    '''
    main page
    '''
    return render_template('index.html', frog='Frog', facts=get_facts())

@app.route('/upload', methods=['GET','POST'])
def upload_file():
    '''Inserts file with prediction to mongo
    Args:
        None
    Returns:
        str: string describing result of prediction, fraud or not
    '''
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('upload_file',
                                    filename=filename))
    return render_template('upload.html', facts=get_facts())

@app.route('/scatter')
def api_scatter():
    return render_template('scatter.html', facts=get_facts())



@app.route('/get_and_score')
def api_score():
    '''Goes to prespecified website, makes prediction on data found there and inserts data+prediciton into mongo db
    Args:
        None
    Returns:
        str: String confirming insertion into mongo
    '''
    url = "URL of live data feed"
    data = urllib2.urlopen(url).read()
    data = json.loads(data)
    data = pd.DataFrame(list(zip(*data.items())[1]), index=list(zip(*data.items())[0]))
    data = data.transpose()
    name2 = data['name'][0]
    country = data['country'][0]
    name = data['org_name']
    model = Model()
    model.load_pandas_data(data)
    model.insert_to_mongo(table_name='website_predictions')
    prediction = model.predict()
    return render_template('get_and_score.html', prediction = str(prediction[0]), name = str(name[0]),\
                            name2 = name2, country=country, facts=get_facts())



if __name__ == '__main__':

    app.run()
