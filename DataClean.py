# Working with data
import numpy as np
import pandas as pd

# Visualizations
import matplotlib.pyplot as plt
from holoviews.plotting.bokeh.styles import alpha
from matplotlib import rcParams
import seaborn as sns

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('DATA/WINE DATASET.csv')
# df = df1.drop(columns=['s.no'])
print(df.head(37).to_string())
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# print(df.skew(numeric_only=True))
#
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(4, 4))
# counts = df['Type'].value_counts()
# plt.pie (counts, labels = counts.index, autopct = lambda p: '{:.1f}%\n({:.0f})'.format(p, p * sum(counts) / 100), startangle=90)
# plt.title('Type')
#
# plt.show()

#
# dummies = pd.get_dummies(df['Type'], drop_first=True).astype(int)
#
# df = df.drop(columns=['Type'])
# df = pd.concat([df, dummies], axis=1)
#
# df.rename(columns={'White Wine': 'Type'}, inplace=True)
#
#
#
# def apply_log_transform(X, features, offset=1e-6):
#     X_df = X.copy()
#     for col in features:
#         if (X_df[col] >= 0).all():
#             if (X_df[col] == 0).any():
#                 X_df[col] = np.log(X_df[col] + offset)
#             else:
#                 X_df[col] = np.log(X_df[col])  # Không cần offset nếu không có 0
#     return X_df
#
# def apply_winsorization(X, features):
#     X_df = X.copy()
#     for col in features:
#         Q1 = X_df[col].quantile(0.25)
#         Q3 = X_df[col].quantile(0.75)
#         IQR = Q3 - Q1
#         lower = Q1 - 1.5 * IQR
#         upper = Q3 + 1.5 * IQR
#         X_df[col] = X_df[col].clip(lower, upper)
#     return X_df
#
#
# features = df.drop(columns=['quality', 'Type']).columns
# X = df[features]
# y = df['quality']
# z = df['Type']
#
# X_processed = apply_log_transform(X, features)
# X_processed = apply_winsorization(X_processed, features)
#
# df2 = X_processed.copy()
# df2['quality'] = y
# df2['Type'] = z
# df2.to_csv('wine_cleaned.csv')
#
#
#
#

# import matplotlib.pyplot as plt
# import seaborn as sns
#
# numeric_cols = df2.select_dtypes(include=['float64', 'int64']).columns
#
# n_cols = 3
# n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
#
# fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
#
# axes = axes.flatten()
#
# # sns.boxplot(x=df2[col], ax=axes[i])
#
# for i, col in enumerate(numeric_cols):
#     sns.histplot(df2[col], kde=True, color='purple', ax=axes[i], bins=60, edgecolor='gray')
#     axes[i].set_title(f'Boxplot của {col}', fontsize=9)
#     axes[i].tick_params(axis='x', labelsize=7)
#     axes[i].tick_params(axis='y', labelsize=7)
#
# for j in range(i+1, len(axes)):
#     fig.delaxes(axes[j])
#
# plt.tight_layout(pad=3.0)
# plt.show()




#
#
#
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# target_col = 'quality'
# numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop([target_col, 'Unnamed: 0', 'Type'])
#
# n_cols = 3
# n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
#
# fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
# axes = axes.flatten()
#
# for i, col in enumerate(numeric_cols):
#     sns.scatterplot(x=df[col], y=df[target_col], ax=axes[i], color='#1f77b4', alpha=0.3, edgecolor=None, marker='o')
#     axes[i].set_title(f'{col} vs {target_col}', fontsize=9)
#     axes[i].tick_params(axis='x', labelsize=7)
#     axes[i].tick_params(axis='y', labelsize=7)
#
# for j in range(i+1, len(axes)):
#     fig.delaxes(axes[j])
#
# plt.tight_layout(pad=3.0)
# plt.show()
#
#



