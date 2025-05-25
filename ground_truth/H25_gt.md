```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- 1. Load the dataset ---
try:
    df = pd.read_csv('iris.csv')
except FileNotFoundError:
    print("Error: 'iris.csv' not found. Please ensure the file is in the current directory.")
    exit()

# --- 2. Define features X and target y ---
# Features are SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = df[features]

# Target is Species
y = df['Species']

# --- 3. Encode the Species target variable ---
# Initialize LabelEncoder
label_encoder = LabelEncoder()
# Fit and transform the 'Species' column
y_encoded = label_encoder.fit_transform(y)

# --- 4. Split data into training (80%) and testing (20%) sets ---
# Use random_state for reproducibility and stratify to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# --- 5. Train three different classification models ---

# Initialize a dictionary to store models and their performance
models = {
    'Logistic Regression': LogisticRegression(max_iter=200),  # Increased max_iter for convergence
    'Support Vector Machine (SVC)': SVC(kernel='linear', random_state=42),
    'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Dictionary to store performance metrics for comparison
performance_metrics = {}

print("--- Model Performance Comparison ---")

for name, model in models.items():
    print(f"\n--- Training and Evaluating: {name} ---")

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # --- 6. Report performance metrics for each classifier ---
    # Calculate Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # Calculate Precision (macro average)
    precision = precision_score(y_test, y_pred, average='macro')
    # Calculate Recall (macro average)
    recall = recall_score(y_test, y_pred, average='macro')
    # Calculate F1-score (macro average)
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    print(f"F1-Score (Macro): {f1:.4f}")

    # Store F1-score for comparison
    performance_metrics[name] = f1

# --- 7. Print which model performed best based on the highest F1-score (macro average) ---
if performance_metrics:
    best_model_name = max(performance_metrics, key=performance_metrics.get)
    best_f1_score = performance_metrics[best_model_name]

    print(f"\n--- Best Performing Model ---")
    print(f"The best model based on F1-Score (Macro Average) is: {best_model_name}")
    print(f"With an F1-Score of: {best_f1_score:.4f}")
else:
    print("\nNo model performance data available for comparison.")
```