import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

file_path = 'C:/Users/jidax/Downloads/Global Electricity Statistics.csv'
start_row = 2  
end_row = 58  
columns_to_read = ['1980', '1981']  
target_column = '1980'  

data = pd.read_csv(file_path, skiprows=range(1, start_row), nrows=end_row - start_row, usecols=columns_to_read + [target_column])

data.replace('--', np.nan, inplace=True)

data = data.apply(pd.to_numeric, errors='coerce')

imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

X = data_imputed[columns_to_read]
y = data_imputed[target_column]

pca = PCA(n_components=2)
pca_result = pca.fit_transform(data_imputed)

print("Explained Variance Ratio:", pca.explained_variance_ratio_)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.scatter(data[columns_to_read[0]], data[columns_to_read[1]])
plt.title('Original Data')


plt.subplot(1, 2, 2)
plt.scatter(pca_result[:, 0], pca_result[:, 1])
plt.title('PCA-transformed Data')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


baseline_prediction = np.mean(y_train)
baseline_predictions_train = np.full_like(y_train, fill_value=baseline_prediction)
baseline_predictions_test = np.full_like(y_test, fill_value=baseline_prediction)

mae_train = mean_absolute_error(y_train, baseline_predictions_train)
mse_train = mean_squared_error(y_train, baseline_predictions_train)

mae_test = mean_absolute_error(y_test, baseline_predictions_test)
mse_test = mean_squared_error(y_test, baseline_predictions_test)

print(f"Training Mean Absolute Error: {mae_train}")
print(f"Training Mean Squared Error: {mse_train}")
print(f"Testing Mean Absolute Error: {mae_test}")
print(f"Testing Mean Squared Error: {mse_test}")
