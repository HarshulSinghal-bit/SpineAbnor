from typing import final
from flask import Flask,render_template,request,url_for,redirect
import pickle
import numpy as np
app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
@app.route('/Index.html')
@app.route('/')
def hello_world():
    return render_template('Index.html')
@app.route('/form.html')
def form_return():
    return render_template('form.html')
@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    print(prediction)
    if prediction == 'Abnormal':
        return render_template('form.html',pred = 0)
    if prediction == 'Normal':
        return render_template('form.html',pred = 1)

if __name__=='__main__':
    app.run(debug=True)