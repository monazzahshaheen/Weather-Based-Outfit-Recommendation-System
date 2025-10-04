from flask import Flask, render_template, request
import pandas as pd
import joblib
import webbrowser
import threading

app = Flask(__name__)

# Load your historical dataset
df = pd.read_csv("weather_data+.csv") 

# Load the trained machine learning models and label encoder
rf_model = joblib.load('outfit_model.pkl')
le = joblib.load('label_encoder.pkl')

# Function to predict outfit based on temperature using the trained model
def predict_outfit(input_temp):
    # Use the model to predict outfit based on the temperature
    pred = rf_model.predict([[input_temp]])
    outfit = le.inverse_transform(pred)[0]
    return outfit

@app.route('/', methods=['GET', 'POST'])
def index():
    outfit = None
    if request.method == 'POST':
        try:
            input_temp = float(request.form['temperature'])
            
            # Predict outfit based on the input temperature
            outfit = predict_outfit(input_temp)
        except Exception as e:
            outfit = "Invalid input or no match found."
    
    return render_template('index.html', outfit=outfit)

# Auto-open the browser
def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000')

if __name__ == '__main__':
    threading.Timer(1.25, open_browser).start()
    app.run(debug=True, use_reloader=False)

