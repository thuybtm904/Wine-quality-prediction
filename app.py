# from sklearn.ensemble import RandomForestClassifier
# import pandas as pd
#
# df1 = pd.read_csv('DATA/WINE DATASET.csv')
# df = df1.drop(columns=['s.no'])
#
# dummies = pd.get_dummies(df['Type'], drop_first=True).astype(int)
#
# df = df.drop(columns=['Type'])
# df = pd.concat([df, dummies], axis=1)
#
# df.rename(columns={'White Wine': 'Type'}, inplace=True)
#
#
# X = df.drop('quality', axis=1)
# y = df['quality']
#
# model = RandomForestClassifier(
#     n_estimators=50,
#     max_depth=15,
#     min_samples_split=5,
#     random_state=42
# )
# model.fit(X, y)
#
# import joblib
# joblib.dump(model, 'modelclass.pkl')


# from sklearn.linear_model import LinearRegression
# import pandas as pd
#
# df = pd.read_csv('wine_cleaned.csv')
#
# X = df.drop(['quality', 'Unnamed: 0'], axis=1)
# y = df['quality']
#
# model = LinearRegression()
# model.fit(X, y)
#
# import joblib
# joblib.dump(model, 'modellinear.pkl')



import numpy as np
import pandas as pd
import streamlit as st
import joblib

fea_col = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol"]

def apply_log_transform(X, features, offset=1e-6):
    X_df = X.copy()
    for col in features:
        if (X_df[col] >= 0).all():
            if (X_df[col] == 0).any():
                X_df[col] = np.log(X_df[col] + offset)
            else:
                X_df[col] = np.log(X_df[col])
    return X_df

def apply_winsorization(X, features):
    X_df = X.copy()
    for col in features:
        Q1 = X_df[col].quantile(0.25)
        Q3 = X_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        X_df[col] = X_df[col].clip(lower, upper)
    return X_df

st.title("Dự đoán chất lượng rượu vang")

mode = st.radio("Chọn chế độ", ['Dự đoán bằng file csv', 'Nhập tay'])
select_model = st.selectbox('Chọn mô hình', ['Linear Regression', 'Random Forest'])
file_path, processing, rounding = {
    'Linear Regression': ('modellinear.pkl', True, True),
    'Random Forest': ('modelclass.pkl', False, False)
}[select_model]
model = joblib.load(file_path)

if mode == "Dự đoán bằng file csv":
    st.warning('File phải có 12 cột: fixed acidity, volatile acidity, ..., alcohol, Type')
    file = st.file_uploader('Tải lên file csv', type='csv')

    if file is not None:
        df = pd.read_csv(file, delimiter=';')
        cols = fea_col + ['Type']
        if list(df.columns)!=cols:
            st.error("Thứ tự hoặc tên cột không đúng!")
            st.stop()

        df1 = df

        if processing:
            df = apply_log_transform(df, fea_col)
            df = apply_winsorization(df, fea_col)

        pred = model.predict(df)
        df1['quality'] = np.round(pred) if rounding else pred
        st.success('Kết quả dự đoán')
        st.dataframe(df1)

        result = df1.to_csv(index=False)
        st.download_button('Tải file kết quả', result, 'ket_qua_du_doan.csv', 'text/csv')
else:
    col1, col2 = st.columns(2)
    with col1:
        fixed_acidity = st.number_input("Fixed Acidity", 3.8, 15.9, step=0.1)
        volatile_acidity = st.number_input("Volatile Acidity", 0.08, 1.8, step=0.01)
        citric_acid = st.number_input("Citric Acid", 0.0, 1.66, step=0.01)
        residual_sugar = st.number_input("Residual Sugar", 0.6, 65.88, step=0.1)
        free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", 1, 289, step=1)
        alcohol = st.number_input("Alcohol", 8.0, 14.9, step=0.1, format="%.2f")
    with col2:
        chlorides = st.number_input("Chlorides", 0.009, 0.611, step=0.001)
        total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", 6, 440, step=1)
        density = st.number_input("Density", 0.9872, 1.038, format="%.6f")
        pH = st.number_input("pH", 2.72, 4.0, step=0.01)
        sulphates = st.number_input("Sulphates", 0.22, 2.0, step=0.01)
        wine_type_str = st.selectbox("Wine Type", ["Red wine", "White wine"])
        wine_type_num = 0 if wine_type_str == "Red wine" else 1

    input_dict = {
        "fixed acidity": fixed_acidity,
        "volatile acidity": volatile_acidity,
        "citric acid": citric_acid,
        "residual sugar": residual_sugar,
        "chlorides": chlorides,
        "free sulfur dioxide": free_sulfur_dioxide,
        "total sulfur dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol,
        "Type": wine_type_num
    }
    df_input = pd.DataFrame([input_dict])
    df1 = df_input
    if processing:
        df_input = apply_log_transform(df_input, fea_col)
        df_input = apply_winsorization(df_input, fea_col)

    if st.button("Dự đoán"):
        pred = model.predict(df_input)
        predicted_value = np.round(pred[0]) if rounding else pred[0]
        st.success(f"Chất lượng rượu dự đoán: **{int(predicted_value)}** (trên thang 0–10)")

        df1["quality"] = pred
        result = df1.to_csv(index=False)
        st.download_button("Tải kết quả", result, "ket_qua_du_doan.csv", "text/csv")

with st.expander("Xem đánh giá mô hình"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Random Forest ")
        st.markdown("""- **Accuracy:** 0.9966""")
    with col2:
        st.markdown("##### Linear Regression")
        st.markdown("""
        - **R² Score:** 0.3149
        - **RMSE:** 0.7193
        """)

