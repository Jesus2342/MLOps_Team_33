from flask import Flask, request, jsonify
import pickle

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Model API is running!"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()  # Assuming input is JSON
    prediction = model.predict([data['input']])
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
