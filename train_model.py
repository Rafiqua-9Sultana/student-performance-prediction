
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Create project folders
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "student_data.csv"

# Sample dataset
data = pd.DataFrame({
    "study_hours": [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 11
    ],
    "score": [
        35, 40, 50, 55, 65, 70, 75, 85, 90, 95,
        38, 45, 48, 62, 67, 72, 82, 86, 92, 98
    ]
})

# Always rewrite the CSV to fix any empty or corrupted file
data.to_csv(DATA_FILE, index=False)
print("Sample dataset created successfully.")

# Load the dataset
df = pd.read_csv(DATA_FILE)

print("\n=== Student Performance Prediction ===")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# Features and target
X = df[["study_hours"]]
y = df["score"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

print("\n=== Model Evaluation ===")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred) ** 0.5:.2f}")
print(f"R-squared: {r2_score(y_test, y_pred):.3f}")

# Display test predictions
results = pd.DataFrame({
    "Study Hours": X_test["study_hours"].values,
    "Actual Score": y_test.values,
    "Predicted Score": y_pred.round(2)
})

print("\n=== Test Predictions ===")
print(results.to_string(index=False))
results.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

# Save graph
plt.figure(figsize=(8, 5))
plt.scatter(df["study_hours"], df["score"], label="Sample data")

line = pd.DataFrame({
    "study_hours": [df["study_hours"].min(), df["study_hours"].max()]
})
plt.plot(line["study_hours"], model.predict(line), label="Regression line")

plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.title("Student Performance Prediction")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "performance_plot.png", dpi=150)
plt.close()

print("\nGraph saved to outputs/performance_plot.png")

# Predict a score for new study hours
try:
    hours = float(input("\nEnter study hours (0 to 12): "))

    if 0 <= hours <= 12:
        new_data = pd.DataFrame({"study_hours": [hours]})
        prediction = model.predict(new_data)[0]
        prediction = max(0, min(100, prediction))
        print(f"Estimated score: {prediction:.2f} / 100")
        print("Note: This is a sample educational model.")
    else:
        print("Please enter a number between 0 and 12.")

except ValueError:
    print("Invalid input. Please enter a number.")

print("\nProject completed successfully!")
