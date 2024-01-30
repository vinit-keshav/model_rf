import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Read data
data = pd.read_csv(r"C:\Users\vinit\Downloads\nf_all\NF_ALL.csv")
data = data.drop('Series', axis=1)

# Downsample data
data = data.sample(frac=0.1, random_state=42)

# Calculate mean values
mean_trades = data['Trades'].mean()
mean_deliverable_volume = data['Deliverable Volume'].mean()
mean_percent_deliverable = data['%Deliverble'].mean()

# Replace null values with mean values
data['Trades'].fillna(mean_trades, inplace=True)
data['Deliverable Volume'].fillna(mean_deliverable_volume, inplace=True)
data['%Deliverble'].fillna(mean_percent_deliverable, inplace=True)

from sklearn.preprocessing import LabelEncoder

# Assuming 'data' is your DataFrame

# Initialize LabelEncoder
encoder = LabelEncoder()

# Apply LabelEncoder to categorical columns
categorical_columns = ['Symbol']
for column in categorical_columns:
    data[column] = encoder.fit_transform(data[column])

# Display the modified DataFrame
print(data)
plt.figure(figsize=(12, 8))

# Boxplot before outlier removal
for column in ['Prev Close', 'Open', 'High', 'Low', 'Last', 'Close', 'VWAP', 'Volume', 'Turnover', 'Trades', 'Deliverable Volume', '%Deliverble']:
    plt.boxplot(data[column], vert=False)
    plt.title(f'Boxplot for {column} Before Outlier Removal')
    plt.show()

# Function to remove outliers based on IQR
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Drop rows with outliers
    data.drop(data[(data[column] < lower_bound) | (data[column] > upper_bound)].index, inplace=True)

# Remove outliers from specified columns
for column in ['Prev Close', 'Open', 'High', 'Low', 'Last', 'Close', 'VWAP', 'Volume', 'Turnover', 'Trades', 'Deliverable Volume', '%Deliverble']:
    remove_outliers_iqr(data, column)

# Boxplot after outlier removal
plt.figure(figsize=(12, 8))
for column in ['Prev Close', 'Open', 'High', 'Low', 'Last', 'Close', 'VWAP', 'Volume', 'Turnover', 'Trades', 'Deliverable Volume', '%Deliverble']:
    plt.boxplot(data[column], vert=False)
    plt.title(f'Boxplot for {column} After Outlier Removal')
    plt.show()