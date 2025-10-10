import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# 1 Load dataset
df = pd.read_csv("Data.csv")

# 2 Convert target column ("Churn") to numeric
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# 3 Drop ID column
df = df.drop(columns=["customerID"])

# 4 Encode categorical columns
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# 5️ Scale numeric columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop("Churn")
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# 6️ Separate features and target
X = df.drop(columns=["Churn"])
y = df["Churn"]

# 7️ Choose your model here:
# model = LogisticRegression(max_iter=2000)
# model = RandomForestClassifier(n_estimators=200, random_state=42)
model = LogisticRegression(max_iter=2000)


# 8️ K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Fold {fold} Accuracy: {acc:.3f}")
    scores.append(acc)

print(f"\nAverage Accuracy: {sum(scores)/len(scores):.3f}")
