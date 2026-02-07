"""
Flask web application for Titanic survival prediction.
"""
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load model and scaler on startup
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Store predictions and feedback in memory
predictions_history = []
feedback_list = []


@app.route('/')
def index():
    """Display the input form."""
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    """Display the dashboard with statistics."""
    # Calculate statistics
    total_predictions = len(predictions_history)
    survived_count = sum(1 for p in predictions_history if p['survived'])
    not_survived_count = total_predictions - survived_count

    survival_rate = (survived_count / total_predictions * 100) if total_predictions > 0 else 0

    # Calculate average probability
    avg_probability = sum(p['probability'] for p in predictions_history) / total_predictions if total_predictions > 0 else 0

    # Class distribution
    class_counts = {1: 0, 2: 0, 3: 0}
    for p in predictions_history:
        class_counts[p['pclass']] = class_counts.get(p['pclass'], 0) + 1

    # Gender distribution
    gender_counts = {'Male': 0, 'Female': 0}
    for p in predictions_history:
        gender_counts[p['sex']] = gender_counts.get(p['sex'], 0) + 1

    # Feedback stats
    feedback_counts = {'excellent': 0, 'good': 0, 'okay': 0, 'bad': 0}
    for f in feedback_list:
        feedback_counts[f] = feedback_counts.get(f, 0) + 1

    # Recent predictions (last 10)
    recent_predictions = predictions_history[-10:][::-1]

    return render_template('dashboard.html',
                           total_predictions=total_predictions,
                           survived_count=survived_count,
                           not_survived_count=not_survived_count,
                           survival_rate=survival_rate,
                           avg_probability=avg_probability,
                           class_counts=class_counts,
                           gender_counts=gender_counts,
                           feedback_counts=feedback_counts,
                           recent_predictions=recent_predictions)


@app.route('/predict', methods=['POST'])
def predict():
    """Process form and return prediction."""
    try:
        # Get form data
        pclass = int(request.form['pclass'])
        sex = int(request.form['sex'])  # 0=male, 1=female
        age = float(request.form['age'])
        fare = float(request.form['fare'])
        marital_status = int(request.form['marital_status'])
        height = float(request.form['height'])

        # Validate inputs
        if age < 0 or age > 120:
            return render_template('index.html', error="Age must be between 0 and 120")
        if fare < 0:
            return render_template('index.html', error="Fare must be non-negative")
        if height < 50 or height > 250:
            return render_template('index.html', error="Height must be between 50 and 250 cm")

        # Prepare features
        features = np.array([[pclass, sex, age, fare, marital_status, height]])
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]

        survived = bool(prediction)
        survival_prob = probability[1] * 100

        # Map values for display
        marital_map = {0: 'Single/Bachelor', 1: 'Married', 2: 'Widowed'}
        sex_display = 'Female' if sex == 1 else 'Male'

        # Store prediction in history
        predictions_history.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'pclass': pclass,
            'sex': sex_display,
            'age': age,
            'fare': fare,
            'marital_status': marital_map.get(marital_status, 'Unknown'),
            'height': height,
            'survived': survived,
            'probability': survival_prob
        })

        return render_template('result.html',
                               survived=survived,
                               probability=survival_prob,
                               pclass=pclass,
                               sex=sex_display,
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


@app.route('/api/stats')
def api_stats():
    """API endpoint for dashboard stats (for real-time updates)."""
    total = len(predictions_history)
    survived = sum(1 for p in predictions_history if p['survived'])
    return jsonify({
        'total': total,
        'survived': survived,
        'not_survived': total - survived,
        'survival_rate': (survived / total * 100) if total > 0 else 0
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
