from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# โหลดโมเดล
model = joblib.load("model/stroke_prediction_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = np.array([data['features']]).reshape(1, -1)
        prediction = model.predict(features)[0]
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)