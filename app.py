from flask import Flask, request, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("best_model.joblib")

# Feature names (must match training order)
COLS = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]

# Simple HTML template
HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <title>Diabetes Prediction</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 40px; }
    h1 { color: #2c3e50; }
    label { display: block; margin-top: 10px; }
    input { margin-top: 5px; padding: 5px; width: 200px; }
    button { margin-top: 15px; padding: 10px 15px; }
    .result { margin-top: 20px; font-size: 18px; color: #e74c3c; }
  </style>
</head>
<body>
  <h1>Diabetes Prediction</h1>
  <form method="POST" action="/predict_form">
    {% for c in cols %}
      <label>{{c}}:
        <input type="number" step="any" name="{{c}}" required>
      </label>
    {% endfor %}
    <button type="submit">Predict</button>
  </form>

  {% if result is defined %}
    <div class="result">
      <h2>Prediction: {{ result }}</h2>
      {% if probability is not none %}
        <p>Probability of Diabetes: {{ probability|round(3) }}</p>
      {% endif %}
    </div>
  {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE, cols=COLS)

@app.route("/predict_form", methods=["POST"])
def predict_form():
    data = request.form.to_dict()
    try:
        values = [float(data[c]) for c in COLS]
        X = np.array(values).reshape(1, -1)
        pred = int(model.predict(X)[0])
        prob = None
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X)[0,1])
        label = "Diabetic" if pred == 1 else "Not Diabetic"
        return render_template_string(HTML_TEMPLATE, cols=COLS, result=label, probability=prob)
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, cols=COLS, result="Error", probability=None)

if __name__ == "__main__":
    app.run(debug=True)

