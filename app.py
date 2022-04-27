from flask import Flask, request, render_template
import pandas as pd
import joblib


# Declare a Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        GBR = joblib.load("GBR.pkl")
        
        # Get values through input bars
        Height = request.form.get("Height")
        SlopeAngle = request.form.get("SlopeAngle")
        Cohesion = request.form.get("Cohesion")
        FrictionAngle = request.form.get("FrictionAngle")
        
        # Put inputs to dataframe
        X = pd.DataFrame([[Height,SlopeAngle,Cohesion,FrictionAngle]], columns = ["Height", "SlopeAngle","Cohesion", "FrictionAngle"])
        
        # Get prediction
        prediction = GBR.predict(X)[0]
        
    else:
        prediction = ""
        
    return render_template("website.html", output = prediction)

# Running the app
if __name__ == '__main__':
	app.run(debug = True)

