import os
import pandas as pd

# Load the CSV data
df = pd.read_csv('D:/Technologia/ML/C2H4/dataset/valid/_annotations.csv')

# Define a function to map class names to binary labels based on filenames
def relabel_based_on_filename(row):
    # Check if 'plastic' is in the filename
    if 'plastic' in row['filename'].lower():
        return 1  # Plastic = 1
    else:
        return 0  # Non-plastic = 0

# Apply the function to create a new binary class column
df['binary_class'] = df.apply(relabel_based_on_filename, axis=1)

# Save the updated CSV with binary labels
df.to_csv('updated_annotations.csv', index=False)

# Check the updated data
print(df.head())
