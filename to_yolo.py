import pandas as pd
import os

# Load the CSV data
df = pd.read_csv('path_to_annotations.csv')

# Ensure the 'labels' folder exists in the parent directory
labels_path = 'labels'  # Parent directory for labels
if not os.path.exists(labels_path):
    os.makedirs(labels_path)

# Function to convert to YOLO format
def convert_to_yolo(row):
    # Parse the CSV row
    filename = row['Filename']
    width = row['width']
    height = row['height']
    xmin = row['xmin']
    ymin = row['ymin']
    xmax = row['xmax']
    ymax = row['ymax']
    
    # Map class to class ID (plastic = 0, non-plastic = 1)
    class_id = 0 if row['Class'] == 'plastic' else 1
    
    # Normalize the bounding box coordinates
    center_x = (xmin + xmax) / 2.0 / width
    center_y = (ymin + ymax) / 2.0 / height
    bbox_width = (xmax - xmin) / width
    bbox_height = (ymax - ymin) / height
    
    # Save the YOLO formatted annotation to a text file in the 'labels' folder
    label_filename = os.path.splitext(filename)[0] + '.txt'  # Create the text file with same name as image
    label_filepath = os.path.join(labels_path, label_filename)
    
    with open(label_filepath, 'a') as label_file:
        label_file.write(f"{class_id} {center_x} {center_y} {bbox_width} {bbox_height}\n")

# Apply conversion to each row in the CSV
df.apply(convert_to_yolo, axis=1)
