import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = './data/house_prices/house_prices_train.csv'
df = pd.read_csv(file_path)

# Data Exploration
print(f"Shape of the dataset: {df.shape}")
print("First few rows of the dataset:")
print(df.head())
print("Information about the dataset:")
print(df.info())
print("Descriptive statistics of the dataset:")
print(df.describe())
missing_values = df.isnull().sum()
print("Missing values in each column:")
print(missing_values[missing_values > 0])

# Data Processing
# Step 1: Handle missing values
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Step 2: Remove unnecessary columns
df_cleaned = df.drop(columns=['Id', 'PoolQC', 'Alley', 'MiscFeature', 'Fence', 'FireplaceQu'])

# Step 3: Convert categorical variables to numerical variables using one-hot encoding
df_cleaned = pd.get_dummies(df_cleaned, drop_first=True)

# Step 4: Scale numerical variables
scaler = StandardScaler()
numerical_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
df_cleaned[numerical_cols] = scaler.fit_transform(df_cleaned[numerical_cols])

# Step 5: Separate target variable and features
X = df_cleaned.drop(columns=['SalePrice'])
y = df_cleaned['SalePrice']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
# Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the performance metrics
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Visualize the predictions vs actual values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs Predicted Sale Price')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal line
plt.savefig('actual_vs_predicted.png')
plt.show()