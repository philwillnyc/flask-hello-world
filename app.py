from flask import Flask, render_template, request, session
from flask_session import Session

from model import predict

#Create the flask app.

app = Flask(__name__)

#Set up individual sessions so multiple users don't conflict. 

app.secret_key = '123456789'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
server_session = (Session(app))

#Computational functions.

def compute_bmi(height,weight):
    """Height in inches, weight in lbs."""
    return 703*(weight/(height**2))

def check_prediabetes(a1c,glucose):
    return int(glucose > 100 or a1c > 5.6)

#Homepage.

@app.route('/', methods=['GET'])
def home():
    session['prediction'] = 'Enter information above to predict risk of gestational diabetes.'
    return render_template(
                'predict.html', 
                prediction = session['prediction'],
                    )

#Calculate risk. 

@app.route('/', methods=['POST'])
def compute():
    r = request.form
    try:
        dia_bp = int(r['diastolic'])
        sys_bp = int(r['systolic'])
        hdl = int(r['hdl'])
        bmi = compute_bmi(int(r['height']),int(r['weight']))
        age = int(r['age'])
        prediabetes = check_prediabetes(float(r['a1c']),int(r['glucose']))
        no_prediabetes = "" if prediabetes else "no"
        percent = f'{(int(100*round(predict(dia_bp,sys_bp,hdl,bmi,age,prediabetes),2)))}%'
        session['prediction'] = f"""
        A {age} year old female with a BMI of {round(bmi,1)}, a serum HDL cholesterol of {hdl}, blood pressure of {sys_bp} 
        over {dia_bp}, and {no_prediabetes} prediabetes has a {percent} chance of developing gestational diabetes in her next pregnancy. 
        """
    except:
        session['prediction'] = 'Error, please check your inputs and try again.'
    return render_template(
                'predict.html', 
                prediction = session['prediction'],
                    )

app.run()
