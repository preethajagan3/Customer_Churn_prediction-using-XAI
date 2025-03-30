from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import shap


app = Flask(__name__)


# Load the trained model and scaler
model = pickle.load(open("model .pkl", "rb"))
scaler = pickle.load(open("scaler .pkl", "rb"))


# Define the 19 input feature names
feature_names = [
    'tenure', 'Contract', 'TechSupport', 'TotalCharges', 'InternetService',
    'MultipleLines', 'StreamingTV', 'MonthlyCharges', 'PhoneService',
    'PaperlessBilling', 'PaymentMethod', 'SeniorCitizen', 'Dependents',
    'Partner', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'StreamingMovies', 'UnlimitedData'
]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values for all 19 features
        input_data = [float(request.form[feature]) for feature in feature_names]
        input_array = np.array([input_data])


        # Apply feature scaling
        input_scaled = scaler.transform(input_array)


        # Make prediction
        probability = model.predict_proba(input_scaled)[:, 1][0] * 100
        prediction = "Yes (Churn)" if probability > 50 else "No (Not Churn)"


        # Explain prediction using SHAP
        explainer = shap.Explainer(model)
        shap_values = explainer(input_scaled)
        top_features = shap_values.values[0]
        sorted_indices = np.argsort(np.abs(top_features))[::-1]


        # Extract top 5 most influencing features
        explanation = [
            {"feature": feature_names[idx], "impact": round(top_features[idx], 2)}
            for idx in sorted_indices[:5]
        ]


        return render_template("result.html", prediction=prediction, probability=round(probability, 2), explanation=explanation)


    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)


