from distutils.command.upload import upload
from flask import Flask, render_template, request
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('stroke.pkl','rb'))

app = Flask(__name__, template_folder='templates')


@app.route('/')
def man():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g']
    data8 = request.form['h']
    data9 = request.form['i']
    data10 = request.form['j']

    
    arr = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10]

    c = np.array(arr)
    dat = MinMaxScaler().fit_transform(c[:,np.newaxis])
    dat = pd.DataFrame(dat)
    dat = dat.T
        
    pred = model.predict(dat)
    
        
    return render_template('pred.html', value=pred)

    

if __name__ == "__main__":
    app.run(debug=True, port = 8000)