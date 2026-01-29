"""
Flask web application for Titanic survival prediction.
"""
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler on startup
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Store feedback in memory (in production, use a database)
feedback_list = []


@app.route('/')
def index():
    """Display the input form."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Process form and return prediction."""
    try:
        # Get form data
        pclass = int(request.form['pclass'])
        sex = int(request.form['sex'])  # 0=male, 1=female
        age = float(request.form['age'])
        fare = float(request.form['fare'])
        marital_status = int(request.form['marital_status'])  # 0=single, 1=married, 2=widowed
        height = float(request.form['height'])

        # Validate inputs
        if age < 0 or age > 120:
            return render_template('index.html', error="Age must be between 0 and 120")
        if fare < 0:
            return render_template('index.html', error="Fare must be non-negative")
        if height < 50 or height > 250:
            return render_template('index.html', error="Height must be between 50 and 250 cm")

        # Prepare features (order: pclass, sex, age, fare, marital_status, height)
        features = np.array([[pclass, sex, age, fare, marital_status, height]])

        # Scale features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]

        # Prepare result
        survived = bool(prediction)
        survival_prob = probability[1] * 100  # Probability of survival

        # Map marital status to display text
        marital_map = {0: 'Single/Bachelor', 1: 'Married', 2: 'Widowed'}

        return render_template('result.html',
                               survived=survived,
                               probability=survival_prob,
                               pclass=pclass,
                               sex='Female' if sex == 1 else 'Male',
                               age=age,
                               fare=fare,
                               marital_status=marital_map.get(marital_status, 'Unknown'),
                               height=height)

    except ValueError as e:
        return render_template('index.html', error="Please enter valid numeric values")
    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {str(e)}")


@app.route('/feedback', methods=['POST'])
def feedback():
    """Store user feedback."""
    try:
        data = request.get_json()
        rating = data.get('rating')
        feedback_list.append(rating)
        return jsonify({'status': 'success', 'message': 'Thank you for your feedback!'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
