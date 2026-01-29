# Titanic Survival Predictor

A Flask web application that predicts Titanic passenger survival using Logistic Regression.

## Features

- **Survival Prediction**: Enter passenger details to get survival probability
- **Bilingual Support**: English and Arabic (RTL) languages
- **Dark/Light Mode**: Toggle between themes
- **Feedback System**: Rate predictions with emoji feedback

## Input Features

| Feature | Description |
|---------|-------------|
| Passenger Class | 1st, 2nd, or 3rd class |
| Sex | Male or Female |
| Marital Status | Single, Married, or Widowed |
| Age | Passenger age |
| Height | Height in cm |
| Fare | Ticket fare in pounds |

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/titanic-survival-predictor.git
   cd titanic-survival-predictor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model:
   ```bash
   python train_model.py
   ```

4. Run the application:
   ```bash
   python app.py
   ```

5. Open http://localhost:5000 in your browser

## Quick Start (Windows)

Double-click `start_app.bat` to launch the application.

## Tech Stack

- **Backend**: Flask (Python)
- **ML Model**: Scikit-learn Logistic Regression
- **Frontend**: Bootstrap 5
- **Dataset**: Seaborn Titanic Dataset

## Screenshots

### Light Mode
![Light Mode](screenshots/light-mode.png)

### Dark Mode
![Dark Mode](screenshots/dark-mode.png)

### Arabic Version
![Arabic](screenshots/arabic.png)

## License

MIT License
