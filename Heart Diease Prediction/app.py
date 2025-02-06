from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize the app
app = Flask(__name__)

# Load the trained model
model = joblib.load('heart_disease_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        age = int(request.form['age'])
        cp = int(request.form['cp'])
        thalach = int(request.form['thalach'])

        # Create input array for prediction
        data = np.array([[age, cp, thalach]])
        prediction = model.predict(data)

        # Interpret prediction result
        result = "Heart Disease Present" if prediction[0] == 1 else "No Heart Disease"
        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)