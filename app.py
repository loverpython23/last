import pickle
import json
import numpy as np

__data_columns = None
__model = None
def load_saved_artifacts():
    print("loading saved artifacts...start")
    global  __data_columns
    with open("columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
    global __model
    if __model is None:
        # with open('static/artifacts/Heart_disease_model.pickle', 'rb') as f:
        __model = pickle.load(open('Heart_disease_model.pickle','rb'))
    print("loading saved artifacts...done")
def get_data_columns():
    return __data_columns
def disease_or_not(age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal):
    global __model
    x = np.zeros(len(__data_columns))
    # age [0]
    x[0]=int(age)
    # sex [1]
    try:
        x[1]=int(sex)
    except:
        if sex.lower()=='male':
            x[1]=1
        elif sex.lower()=='female':
            x[1]=0
    print(x[0],x[1]) 
    # cp [2]
    """  chest pain type
    Value 0: typical angina
    Value 1: atypical angina
    Value 2: non-anginal pain
    Value 3: asymptomatic"""
    try:
        x[2]=int(cp)
    except:
        if cp.lower()=='typical angina' or cp.lower()=='typicalangina':
            x[2]=0
        elif cp.lower()=='atypical angina' or cp.lower()=='atypicalanginaa':
            x[2]=1
        elif cp.lower()=='non-anginal pain' or cp.lower()=='non-anginalpain':
            x[2]=2
        elif cp.lower()=='asymptomatic':
            x[2]=3
    # trestbps [3]
    #resting blood pressure (in mm Hg on admission to the hospital)
    x[3]=trestbps
    # chol [4]
    # serum cholestoral in mg/dl
    x[4]=chol
    # fbs [5]
    """(fasting blood sugar &gt; 120 mg/dl) (1 = true; 0 = false)"""
    try:
        x[5]=int(fbs)
    except:
        if fbs.lower()=='yes' or fbs.lower()=='true':
            x[5]=1
        elif fbs.lower()=='no' or fbs.lower()=='false':
            x[5]=0
    # restecg [6]
    """restecg: resting electrocardiographic results
    -- Value 0: normal
    -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
    -- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria"""
    try:
        x[6]=int(restecg)
    except:
        if restecg.lower()=="normal":
            x[6]=0
        elif restecg.lower()=="st-t wave abnormality" or restecg.lower()=="st t wave abnormality":
            x[6]=1
        elif restecg.lower()=="left ventricular hypertrophy":
            x[6]=2
    # thalach [7]
    """maximum heart rate achieved"""
    x[7]=int(thalach)
    # exang [8]
    """exercise induced angina (1 = yes; 0 = no)"""
    try:
        x[8]=int(exang)
    except:
        if fbs.lower()=='yes' or fbs.lower()=='true':
            x[8]=1
        elif fbs.lower()=='no' or fbs.lower()=='false':
            x[8]=0
    # oldpeak [9]
    """ST depression induced by exercise relative to rest"""
    x[9]=oldpeak
    # slope [10]
    """the slope of the peak exercise ST segment
    -- Value 0: upsloping
    -- Value 1: flat
    -- Value 2: downsloping"""
    try:
        x[10]=int(slope)
    except:
        if slope.lower()=="upsloping":
            x[10]=0
        elif slope.lower()=="flat":
            x[10]=1
        elif slope.lower()=="downsloping":
            x[10]=2
    # ca [11]
    """ca: number of major vessels (0-3) colored by flourosop"""
    x[11]=ca
    # thal
    """1 = normal; 2 = fixed defect; 3 = reversable defect"""
    try:
        x[12]=int(thal)
    except:
        if thal.lower()=="normal":
            x[12]=1
        elif thal.lower()=="fixed defect":
            x[12]=2
        elif thal.lower()=="reversable defect":
            x[12]=3
    predict=__model.predict([x])[0]
    if predict==1:
        one='YES! You have a heart Disease'
    elif predict==0:
        one="NO! You don't have a heart Disease"
    return one
from flask import Flask, request, jsonify,render_template
app = Flask(__name__,template_folder='template',static_folder='static')
@app.route('/')
def home():
    return render_template('app.html')
@app.route('/predict_heart_disease', methods=['POST','GET'])
def predict_heart_disease():
    age = int(request.form['age'])
    sex = request.form['sex']
    cp = request.form['cp']
    trestbps = request.form['trestbps']
    chol = request.form['chol']
    fbs = request.form['fbs']
    restecg = request.form['restecg']
    thalach = request.form['thalach']
    exang = request.form['exang']
    oldpeak = request.form['oldpeak']
    slope = request.form['slope']
    ca = request.form['ca']
    thal = request.form['thal']
    result=disease_or_not(age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal)
    print(result)
    return render_template("app.html",result=result)
if __name__ == "__main__":
    print("Starting Python Flask Server For Heart Disease Prediction...")
    load_saved_artifacts()
    app.run(debug=True)