import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score

df = pd.read_csv("wine_cleaned.csv")
X = df.drop(columns=["quality"])
y = df["quality"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = np.array(X_scaled)
y = np.array(y)

kfold = KFold(n_splits=10, shuffle=True, random_state=42)

r2_list = []
mae_list = []
mse_list = []

def get_model(input_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))  # khai báo input rõ ràng
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # output hồi quy
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mae'])
    return model

fold_idx = 1
for train_index, val_index in kfold.split(X):
    print(f"\nBắt đầu Fold {fold_idx}")

    model = get_model(X.shape[1])
    model.fit(X[train_index], y[train_index],
              epochs=100,
              batch_size=32,
              verbose=0)
    # Dự đoán trên tập validation
    y_pred = model.predict(X[val_index]).flatten()
    # Tính metrics thủ công
    r2 = r2_score(y[val_index], y_pred)
    mae = np.mean(np.abs(y[val_index] - y_pred))
    mse = np.mean((y[val_index] - y_pred) ** 2)

    print(f"Fold {fold_idx} - R²: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}")

    r2_list.append(r2)
    mae_list.append(mae)
    mse_list.append(mse)

    fold_idx += 1

print("\nKết quả trung bình sau K-Fold:")
print(f"R² trung bình: {np.mean(r2_list):.4f}")
print(f"MAE trung bình: {np.mean(mae_list):.4f}")
print(f"MSE trung bình: {np.mean(mse_list):.4f}")