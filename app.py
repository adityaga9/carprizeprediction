import joblib
from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


app = Flask(__name__)


@app.route('/', methods=['GET'])
def Home():
    return render_template('home.html')


@app.route("/predict", methods=['POST','GET'])
def predict():
    a = []
    #df = pd.read_csv("automobile_final3.csv")
    #X = df.iloc[:,:-1].values
    #y = df.iloc[:,-1].values
    #X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    if request.method == 'POST':
        #car_company = int(request.form['car'])
        symboling = int(request.form['symb'])
        fueltype = request.form['Fuel_Ty']
        if fueltype == 'gas':
            fueltype = 0
        else:
            fueltype = 1
        doornumber = request.form['door']
        if doornumber == 2:
            doornumber = 2
        else:
            doornumber = 4
        carwidth = float(request.form['width'])
        cylindernumber = int(request.form['cylinder'])
        horsepower = int(request.form['hp'])
        citympg = int(request.form['city'])
        highwaympg = int(request.form['highway'])
        a.extend([symboling, fueltype, doornumber, carwidth, cylindernumber, horsepower, citympg, highwaympg])
        #lr = LinearRegression()
        #lr.fit(X_train,y_train)
        #y_pred = lr.predict([a])

        model = joblib.load('linear_regression_model7.pkl')
        prediction = model.predict([a])

        return render_template('prediction.html', msg="done", op=prediction)

    return render_template('prediction.html')


if __name__ == "__main__":
    app.secret_key = "hi"
    app.run(debug=True)
