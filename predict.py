import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("dataset.csv")

# Separate features and target
X = data.drop("Disease", axis=1)
y = data["Disease"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Create Decision Tree model
model = DecisionTreeClassifier(random_state=42)

# Train model
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display title
print("\n" + "=" * 60)
print("        AI-BASED DISEASE PREDICTION SYSTEM")
print("=" * 60)

# Display model performance
print("\nMODEL PERFORMANCE")
print("-" * 60)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("\nPlease answer the following questions.")
print("Enter 'yes' or 'no' for each symptom.\n")

# Store user symptoms
symptoms = []

# Ask about each symptom
for symptom in X.columns:
    while True:
        answer = input(
            f"Do you have {symptom.replace('_', ' ')}? (yes/no): "
        ).strip().lower()

        if answer == "yes":
            symptoms.append(1)
            break

        elif answer == "no":
            symptoms.append(0)
            break

        else:
            print("Invalid input. Please enter only 'yes' or 'no'.")

# Convert input into DataFrame
user_input = pd.DataFrame([symptoms], columns=X.columns)

# Predict disease
prediction = model.predict(user_input)[0]

# Display input summary
print("\n" + "=" * 60)
print("                  INPUT SUMMARY")
print("=" * 60)

for symptom, value in zip(X.columns, symptoms):
    status = "Yes" if value == 1 else "No"
    print(f"{symptom.replace('_', ' '):<20}: {status}")

# Display prediction
print("\n" + "=" * 60)
print("                PREDICTION RESULT")
print("=" * 60)
print(f"Predicted Disease: {prediction}")
print("=" * 60)

# Disclaimer
print("\nDisclaimer:")
print("This project is developed for educational purposes only.")
print("It is not a medical diagnosis or a substitute for professional")
print("medical advice.")