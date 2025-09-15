from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Make sure the "model" folder exists
os.makedirs("model", exist_ok=True)

iris = load_iris()
X, y = iris.data, iris.target

model = LogisticRegression(max_iter=200)
model.fit(X, y)

joblib.dump(model, "model/Iris_classifier.pkl")
print("✅ Model saved to model/Iris_classifier.pkl")
