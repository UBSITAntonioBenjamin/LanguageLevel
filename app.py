import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))


@app.route("/",  methods=["GET", "POST"])
def home():
    data = {}
    prediction = ""
    if request.method == "POST":
        for x, value in request.form.items():
            data[x] = float(value)
        prediction = model.predict([list(data.values())])[0]

    return render_template("index.html", prediction=prediction, data=data)


if __name__ == "__main__":
    app.run(debug=True)
