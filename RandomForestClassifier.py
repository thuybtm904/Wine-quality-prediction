import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("DATA/WINE DATASET.csv")
dummies = pd.get_dummies(df['Type'], drop_first=True).astype(int)

df = df.drop(columns=['Type'])
df = pd.concat([df, dummies], axis=1)

df.rename(columns={'White Wine': 'Type'}, inplace=True)
X = df.drop(columns=["quality"])
y = df["quality"]

kfold = KFold(n_splits=10, shuffle=True, random_state=42)

accuracy_list = []

fold_idx = 1
for train_idx, val_idx in kfold.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    accuracy_list.append(acc)
    print(f"Fold {fold_idx} - Accuracy: {acc:.4f}")
    fold_idx += 1

print("\nKết quả trung bình sau 10-Fold:")
print(f"Accuracy trung bình: {np.mean(accuracy_list):.4f}")