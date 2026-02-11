import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

df = pd.read_csv("data/healthcare_data.csv")

# Encode categorical columns
le_dept = LabelEncoder()
le_ins = LabelEncoder()

df["Department"] = le_dept.fit_transform(df["Department"])
df["Insurance_Type"] = le_ins.fit_transform(df["Insurance_Type"])

X = df[["Department", "Insurance_Type", "Claim_Amount", "Payment_Delay_Days"]]
y = df["Claim_Denied"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(classification_report(y_test, predictions))

joblib.dump(model, "model.pkl")
joblib.dump(le_dept, "le_dept.pkl")
joblib.dump(le_ins, "le_ins.pkl")

print("Model Saved Successfully")
