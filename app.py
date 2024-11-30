from flask import Flask, request, render_template
import numpy as np
import pickle
from scratch_linear_regression import ScratchLinearRegression
import os

app = Flask(__name__)

model_path = os.path.join('models', 'model.pkl')
scaler_path = os.path.join('models', 'scaler.pkl')

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('make_dict.pkl', 'rb') as file:
    make_dict = pickle.load(file)
with open('model_dict.pkl', 'rb') as file:
    model_dict = pickle.load(file)
with open('trim_dict.pkl', 'rb') as file:
    trim_dict = pickle.load(file)
with open('body_dict.pkl', 'rb') as file:
    body_dict = pickle.load(file)
with open('transmission_dict.pkl', 'rb') as file:
    transmission_dict = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the user input
        year = int(request.form['year'])
        make = request.form['make'].upper()
        model_input = request.form['model'].upper()
        trim = request.form['trim'].upper()
        body = request.form['body'].lower()
        transmission = request.form['transmission'].lower()
        condition = int(request.form['condition'])
        odometer = int(request.form['odometer'])
        mmr = int(request.form['mmr'])

        print(f"Year: {year}, Make: {make}, Model: {model_input}, Trim: {trim}, Body: {body}")
        print(f"Transmission: {transmission}, Condition: {condition}, Odometer: {odometer}, MMR: {mmr}")

        # Convert input to encoded values
        make_encoded = make_dict.get(make, None)
        model_encoded = model_dict.get(model_input, None)
        trim_encoded = trim_dict.get(trim, None)
        body_encoded = body_dict.get(body, None)
        transmission_encoded = transmission_dict.get(transmission, None)

        print(f"Encoded Values -> Make: {make_encoded}, Model: {model_encoded}, Trim: {trim_encoded}, Body: {body_encoded}, Transmission: {transmission_encoded}")

        # Check for any None values indicating invalid inputs
        if None in [make_encoded, model_encoded, trim_encoded, body_encoded, transmission_encoded]:
            available_makes = ', '.join(make.capitalize() for make in make_dict.keys())
            available_trims = ', '.join(trim.upper() for trim in trim_dict.keys())
            return render_template('index.html', result=f"Error: Invalid car details. Available makes: {available_makes}, Avaible trims: {available_trims}")

        # Prepare the input features for the model
        features = np.array([[year, make_encoded, model_encoded, trim_encoded, body_encoded, transmission_encoded, condition, odometer, mmr]])
        features_scaled = scaler.transform(features)

        print(f"Features Scaled: {features_scaled}")

        # Predict using the model
        predicted_price = model.predict(features_scaled)[0]

        # Return the result to the HTML template
        return render_template('index.html', result=f"Predicted Selling Price: ${predicted_price:.2f}")
    except Exception as e:
        print(f"Error: {str(e)}")
        return render_template('index.html', result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
