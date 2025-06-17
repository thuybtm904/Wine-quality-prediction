import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

def run_linear_regression_cv(X, y):
    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Xây dựng mô hình
    model = LinearRegression()

    # K-Fold Cross-Validation trên tập train
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    r2_scores = []
    rmse_scores = []

    fold = 1
    for train_index, val_index in cv.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_val_fold)

        r2 = metrics.r2_score(y_val_fold, y_pred_fold)
        rmse = np.sqrt(metrics.mean_squared_error(y_val_fold, y_pred_fold))

        r2_scores.append(r2)
        rmse_scores.append(rmse)

        print(f"Fold {fold}: R² = {round(r2, 4)} | RMSE = {round(rmse, 4)}")
        fold += 1

    print("\nR² trung bình (train CV):", round(np.mean(r2_scores), 4))
    print("RMSE trung bình (train CV):", round(np.mean(rmse_scores), 4))

    # Huấn luyện trên toàn bộ train
    model.fit(X_train, y_train)

    # Dự đoán và đánh giá trên tập test
    y_pred_test = model.predict(X_test)
    r2_test = metrics.r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(metrics.mean_squared_error(y_test, y_pred_test))

    print("\n=== Đánh giá trên tập TEST ===")
    print("R² trên tập test:", round(r2_test, 4))
    print("RMSE trên tập test:", round(rmse_test, 4))

    # Biểu đồ dự đoán vs thực tế
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_test, color='blue', alpha=0.6, label='Dự đoán vs Thực tế (test)')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Đường hoàn hảo (y = x)')
    plt.xlabel("Thực tế (quality)")
    plt.ylabel("Dự đoán")
    plt.title("Linear Regression - Tất cả đặc trưng (Không chuẩn hóa)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


df = pd.read_csv("wine_cleaned.csv")
X = df.drop("quality", axis=1)
y = df["quality"]
run_linear_regression_cv(X, y)
