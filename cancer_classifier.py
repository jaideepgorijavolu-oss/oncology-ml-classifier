import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_and_evaluate_model():
    print("Loading Breast Cancer Wisconsin dataset...\n")
    
    # 1. Load the dataset built into scikit-learn
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names) # The features (cell measurements)
    y = data.target # The answers (0 = malignant, 1 = benign)
    
    # 2. Split the data
    # 80% of data is used to train the AI, 20% is hidden away to test it later
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training AI on {len(X_train)} samples, testing on {len(X_test)} samples...\n")
    
    # 3. Initialize and Train the Model (Random Forest)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Make Predictions on the hidden test set
    predictions = model.predict(X_test)
    
    # 5. Evaluate the Model
    accuracy = accuracy_score(y_test, predictions)
    print("--- Model Performance ---")
    print(f"Accuracy: {accuracy * 100:.2f}%\n")
    
    print("Classification Report:")
    print(classification_report(y_test, predictions, target_names=data.target_names))
    
    # 6. Generate a Confusion Matrix Visualization
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=data.target_names, 
                yticklabels=data.target_names)
    plt.title('Machine Learning Confusion Matrix')
    plt.xlabel('Predicted Diagnosis')
    plt.ylabel('Actual Diagnosis')
    plt.tight_layout()
    
    # Save the output image
    plt.savefig('confusion_matrix.png', dpi=300)
    print("Success! Evaluation matrix saved locally as 'confusion_matrix.png'.")

if __name__ == "__main__":
    train_and_evaluate_model()
