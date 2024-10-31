from flask import Flask, request, jsonify
import pickle


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Student Perfomance Classifier - MLP Team 33 is running"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()  
    prediction = model.predict([data['input']])
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
