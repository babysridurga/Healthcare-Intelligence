import pandas as pd
import numpy as np

np.random.seed(42)
n = 5000

departments = ["Cardiology", "Orthopedics", "Neurology", "General"]
insurance = ["Private", "Government", "Self-Pay"]

data = pd.DataFrame({
    "Patient_ID": range(1, n+1),
    "Department": np.random.choice(departments, n),
    "Insurance_Type": np.random.choice(insurance, n),
    "Claim_Amount": np.random.randint(5000, 50000, n),
    "Payment_Delay_Days": np.random.randint(0, 90, n)
})

# Paid amount logic
data["Paid_Amount"] = data["Claim_Amount"] * np.random.uniform(0.4, 1.0, n)

# Revenue leakage
data["Revenue_Leakage"] = data["Claim_Amount"] - data["Paid_Amount"]

# Claim denial
data["Claim_Denied"] = np.where(
    data["Paid_Amount"] < data["Claim_Amount"]*0.6, 1, 0)

# Delay category
data["Delay_Category"] = pd.cut(
    data["Payment_Delay_Days"],
    bins=[-1, 15, 45, 90],
    labels=["Low", "Medium", "High"]
)

data.to_csv("data/healthcare_data.csv", index=False)

print("Dataset Generated Successfully")
