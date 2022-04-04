from flask import Flask, redirect, url_for, render_template, request
import os

app = Flask(__name__)

picFolder = os.path.join('/static')

app.config['UPLOAD_FOLDER'] = picFolder

from models import KNN
import pandas as pd 

final = pd.read_csv("data/realestatedata_final.csv")
features = ['ZipCode', 'Unemp Rate']
knn_m, knn_pred, knn_conf, knn_score = KNN.KNN(final, features)
    

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        feature1 = request.form.get("Zip Code")
        feature2 = request.form.get("Unemployment Rate")
        predict = knn_m[1].predict([[feature1, feature2]])
        response = str(predict[0])  
    else:
        response = ''

    return render_template("index.html", response=response)


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html')
 
 
@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html')


if __name__ == "__main__":
    app.run()
