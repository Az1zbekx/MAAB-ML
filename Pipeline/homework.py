from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


data = load_breast_cancer()
X = data.data
y = data.target



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)
baseline_accuracy = accuracy_score(y_test, y_pred)

print("Baseline Accuracy (manual scaling):", baseline_accuracy)


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])


pipeline.fit(X_train, y_train)

y_pred_pipeline = pipeline.predict(X_test)
pipeline_accuracy = accuracy_score(y_test, y_pred_pipeline)

print("Pipeline Accuracy:", pipeline_accuracy)

print("\nComparison:")
print("Baseline:", baseline_accuracy)
print("Pipeline:", pipeline_accuracy)

if baseline_accuracy == pipeline_accuracy:
    print("Natija bir xil.")
else:
    print("Natija biroz farq qiladi.")

print("\nWhy Pipeline is better?")
print("- Kod toza va qisqa bo‘ladi")
print("- Data leakage oldini oladi")
print("- Cross-validation bilan ishlash oson")
print("- Productionga chiqarish qulay")