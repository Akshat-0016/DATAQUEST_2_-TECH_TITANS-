from flask import Flask, request, jsonify
from sensitive_detector import SensitiveDetector  # your class

app = Flask(__name__)
detector = SensitiveDetector()
detector.load('model_path/model')  # load your saved model

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get("text", "")
    is_sensitive, prob, details = detector.is_sensitive(text)
    return jsonify({
        "is_sensitive": is_sensitive,
        "probability": prob,
        "details": details
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
