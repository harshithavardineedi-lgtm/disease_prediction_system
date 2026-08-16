import pandas as pd

# Load the dataset
data = pd.read_csv("dataset.csv")

# Display the first 5 rows
print("First 5 rows of the dataset:")
print(data.head())

# Display dataset information
print("\nDataset information:")
print(data.info())

# Display the shape
print("\nDataset shape:")
print(data.shape)

# Display disease categories
print("\nDisease categories:")
print(data["Disease"].unique())