import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

file_path = 'healthcare-dataset-stroke-data.csv'
df = pd.read_csv(file_path)

df.dropna(inplace=True)

df['gender'] = df['gender'].map({'Male': 1, 'Female': 0, 'Other': 2})

df['ever_married'] = df['ever_married'].map({'Yes': 1, 'No': 0})
df['work_type'] = df['work_type'].astype('category').cat.codes
df['Residence_type'] = df['Residence_type'].map({'Urban': 1, 'Rural': 0})
df['smoking_status'] = df['smoking_status'].astype('category').cat.codes

X = df.drop(columns=['id', 'stroke']) 
y = df['stroke']  

# Train-Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ปรับค่าข้อมูล StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(),
    "kNN": KNeighborsClassifier(n_neighbors=3),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier()
}

# Train
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.2f}")

# Ensemble Learning (Voting Classifier)
ensemble_model = VotingClassifier(estimators=[
    ('lr', models['Logistic Regression']),
    ('knn', models['kNN']),
    ('dt', models['Decision Tree']),
    ('rf', models['Random Forest']),
    ('gb', models['Gradient Boosting']),
    ('ada', models['AdaBoost'])
], voting='soft')

ensemble_model.fit(X_train, y_train)
y_pred_ensemble = ensemble_model.predict(X_test)
print("Ensemble Model Accuracy:", accuracy_score(y_test, y_pred_ensemble))

joblib.dump(ensemble_model, 'model/stroke_prediction_model.pkl')