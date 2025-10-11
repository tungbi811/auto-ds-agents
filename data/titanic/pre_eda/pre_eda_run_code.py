import sys
import os
sys.path.extend(['.', '..', '../..', '../../..', '../../../..', 'multi_agents', 'multi_agents/tools', 'multi_agents/prompts'])
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools.ml_tools import *

def generated_code_function():
    import numpy as np
    import pandas as pd
    
    import pandas as pd
    
    # Load the datasets
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    
    # Overview of the training dataset
    print("Training Data Preview:")
    print(train_data.head())
    print("\nTraining Data Info:")
    print(train_data.info())
    print("\nTraining Data Description:")
    print(train_data.describe())
    
    # Overview of the test dataset
    print("\nTest Data Preview:")
    print(test_data.head())
    print("\nTest Data Info:")
    print(test_data.info())
    print("\nTest Data Description:")
    print(test_data.describe())
    
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Checking for missing values in both datasets
    missing_train = train_data.isnull().sum()
    missing_test = test_data.isnull().sum()
    print("\nMissing Values in Train Data:")
    print(missing_train[missing_train > 0])
    print("\nMissing Values in Test Data:")
    print(missing_test[missing_test > 0])
    
    # Visualizing missing values with a heatmap for the training data
    plt.figure(figsize=(10, 6))
    sns.heatmap(train_data.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap - Training Data')
    plt.show()
    
    
    # Univariate Analysis: Categorical Features
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.countplot(x='Sex', data=train_data)
    plt.title('Count of Passengers by Sex')
    
    plt.subplot(1, 3, 2)
    sns.countplot(x='Pclass', data=train_data)
    plt.title('Count of Passengers by Passenger Class')
    
    plt.subplot(1, 3, 3)
    sns.countplot(x='Embarked', data=train_data)
    plt.title('Count of Passengers by Embarkation Port')
    
    plt.tight_layout()
    plt.show()
    
    # Univariate Analysis: Numerical Features
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(train_data['Age'].dropna(), bins=30, kde=True)
    plt.title('Distribution of Age')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x=train_data['Fare'])
    plt.title('Boxplot of Fare')
    
    plt.tight_layout()
    plt.show()
    
    
    # Correlation Analysis
    correlation_matrix = train_data.corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Matrix')
    plt.show()
    


if __name__ == "__main__":
    generated_code_function()