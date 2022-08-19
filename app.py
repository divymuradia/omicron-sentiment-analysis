import numpy as np
from flask import Flask, request, jsonify, render_template

import pickle
from PIL import Image

import matplotlib.pyplot as plt
import pandas as pd


app = Flask(__name__)



model = pickle.load(open('NLP_model_naivebased.pkl','rb'))
model1 = pickle.load(open('linearregression.pkl','rb'))
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
print(cv)
corpus=pd.read_csv('corpus_dataset.csv')
corpus1=corpus['tweets'].tolist()
X = cv.fit_transform(corpus1).toarray()


@app.route('/')
def check():
    return render_template("check.html")

@app.route('/index')
def home():
    return render_template("index.html")

@app.route('/aboutusnew')
def aboutusnew():
    return render_template('aboutusnew.html')

@app.route('/new')
def new():
    return render_template("new.html")

  
@app.route('/predict',methods=['GET'])
def predict():
    
    
    '''
    For rendering results on HTML GUI
    '''
    tweets = request.args.get('tweets')
    
    tweets=[tweets]

    input_data = cv.transform(tweets).toarray()

    #prediction = model.predict(input_data)

    #input_pred = input_pred.astype(int)
    Model = (request.args.get('Model'))

    if Model=="Naive Bayes Classifier":
      prediction = model.predict(input_data)

    elif Model=="Decision Tree Classifer":
      prediction = model.predict(input_data)
      
    elif Model=="KNN Classifer":
      prediction = model1.predict(input_data)

    elif Model=="SVM Classifer":
      prediction = model.predict(input_data)

    elif Model=="Kernel SVM CLassifer":
      prediction = model1.predict(input_data)

    elif Model=="RANdom Forest Classifer":
      prediction = model1.predict(input_data)

    elif Model=="Linear Regression":
      prediction = model1.predict(input_data)




    
    if prediction[0]==2:
      return render_template('index.html', prediction_text='Tweets is Positive')
      
    elif prediction[0]==1:    
      return render_template('index.html', prediction_text='Tweets is Negative')
    else:
      return render_template('index.html', prediction_text='Tweets is Netural')


if __name__ == "__main__":
    app.run(debug=True)