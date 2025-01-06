import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the dataset
dataset_path = (r'C:\\Users\\Aryan\\Downloads\\Student Depression Dataset.csv')
try:
    df = pd.read_csv(dataset_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"File not found at {dataset_path}. Please check the path and try again.")
    raise

# Check the shape and first few rows
print("Dataset shape:", df.shape)
print("First few rows of the dataset:")
print(df.head())

# Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Handle missing values
df.fillna(0, inplace=True)  # Replace NaN with 0
df.drop_duplicates(inplace=True)  # Remove duplicate rows

# Check data types
print("Data types in the dataset:")
print(df.dtypes)

# Example: Convert a column to datetime (replace 'date_column' with the actual column name)
if 'date_column' in df.columns:
    try:
        df['date_column'] = pd.to_datetime(df['date_column'])
        print("Date column converted to datetime format.")
    except Exception as e:
        print(f"Error converting date_column to datetime: {e}")

# Verify the cleaned dataset
print("Dataset information after cleaning:")
print(df.info())

# Summary statistics
print("Summary statistics:")
print(df.describe())

# Group by a column (replace 'category_column' and 'value_column' with actual names)
if 'category_column' in df.columns and 'value_column' in df.columns:
    print("Group by category and calculate mean:")
    print(df.groupby('category_column')['value_column'].mean())

# Histogram (replace 'numeric_column' with an actual column name)
if 'numeric_column' in df.columns:
    sns.histplot(df['numeric_column'], kde=True)
    plt.title('Distribution of Numeric Column')
    plt.show()

# Boxplot (replace 'category_column' and 'value_column' with actual names)
if 'category_column' in df.columns and 'value_column' in df.columns:
    sns.boxplot(x='category_column', y='value_column', data=df)
    plt.title('Boxplot of Values by Category')
    plt.show()

# Heatmap of correlations
numeric_cols = df.select_dtypes(include=[np.number])
if not numeric_cols.empty:
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

# Plotly bar plot (replace 'category_column' and 'value_column' with actual names)
if 'category_column' in df.columns and 'value_column' in df.columns:
    fig = px.bar(df, x='category_column', y='value_column', color='category_column', title="Bar Plot Example")
    fig.show()

# Plotly scatter plot (replace 'numeric_column1' and 'numeric_column2' with actual names)
if 'numeric_column1' in df.columns and 'numeric_column2' in df.columns and 'category_column' in df.columns:
    fig = px.scatter(df, x='numeric_column1', y='numeric_column2', color='category_column', title="Scatter Plot Example")
    fig.show()

# Aggregate data by date (replace 'date_column' and 'value_column' with actual names)
if 'date_column' in df.columns and 'value_column' in df.columns:
    time_series_data = df.groupby('date_column')['value_column'].sum().reset_index()
    fig = px.line(time_series_data, x='date_column', y='value_column', title="Time Series Visualization")
    fig.show()

# Chunk processing for large datasets
large_dataset_path = 'large_dataset.csv'  # Replace with your large dataset path
chunk_size = 100000
try:
    for chunk in pd.read_csv(large_dataset_path, chunksize=chunk_size):
        print(f"Processed chunk shape: {chunk.shape}")
except FileNotFoundError:
    print(f"Large dataset file not found at {large_dataset_path}. Skipping chunk processing.")

# Sampling (10% of the data)
sample_df = df.sample(frac=0.1, random_state=42)
print("Sampled dataset shape:", sample_df.shape)

# Save the cleaned and sampled datasets
df.to_csv('cleaned_dataset.csv', index=False)
sample_df.to_csv('sample_dataset.csv', index=False)
print("Cleaned and sampled datasets saved as 'cleaned_dataset.csv' and 'sample_dataset.csv'.")
