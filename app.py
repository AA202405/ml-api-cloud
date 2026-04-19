from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Load model (works in AWS + local)
model_path = os.path.join(os.path.dirname(__file__), 'sentiment_model.joblib')
model = joblib.load(model_path)

@app.route('/')
def home():
    return "Sentiment API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': 'Please provide text in JSON format'}), 400

    text = data['text']
    prediction = model.predict([text])[0]

    return jsonify({
        'input_text': text,
        'sentiment_prediction': prediction
    })

# Required for AWS Elastic Beanstalk
application = app

# Only for local run
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)