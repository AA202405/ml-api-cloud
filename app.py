from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
application = app


# Load model
model = joblib.load('sentiment_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided. Please send a JSON with a "text" key.'}), 400

    prediction = model.predict([text])[0]

    return jsonify({
        'input_text': text,
        'sentiment_prediction': prediction
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)