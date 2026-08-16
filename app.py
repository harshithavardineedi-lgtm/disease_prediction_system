from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)


# ==========================================
# LOAD DATASET
# ==========================================

data = pd.read_csv("dataset.csv")

X = data.drop("Disease", axis=1)
y = data["Disease"]


# ==========================================
# TRAIN MODEL
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

model = DecisionTreeClassifier(
    random_state=42
)

model.fit(X_train, y_train)


# ==========================================
# MODEL ACCURACY
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

accuracy_percentage = round(
    accuracy * 100,
    2
)


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template(
        "home.html"
    )


# ==========================================
# PREDICTION PAGE
# ==========================================

@app.route("/prediction")
def prediction_page():

    return render_template(
        "index.html",
        accuracy=accuracy_percentage
    )


# ==========================================
# PREDICT DISEASE
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        selected_symptoms = request.form.getlist(
            "symptoms"
        )


        # Check input

        if len(selected_symptoms) == 0:

            return render_template(
                "index.html",
                accuracy=accuracy_percentage,
                error="Please select at least one symptom."
            )


        # Convert symptoms to model format

        input_data = []

        for symptom in X.columns:

            if symptom in selected_symptoms:

                input_data.append(1)

            else:

                input_data.append(0)


        # Create DataFrame

        user_input = pd.DataFrame(
            [input_data],
            columns=X.columns
        )


        # Prediction

        prediction = model.predict(
            user_input
        )[0]


        # Return result

        return render_template(
            "index.html",
            accuracy=accuracy_percentage,
            prediction=prediction
        )


    except Exception as e:

        return render_template(
            "index.html",
            accuracy=accuracy_percentage,
            error=str(e)
        )


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )