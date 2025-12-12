
import pandas as pd

# Load the CSV data
df = pd.read_csv('path_to_annoD:\Technologia\ML\C2H4\dataset\test\_annotations.csv')

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
    
    # Save the YOLO formatted annotation to a text file
    yolo_annotation = f"{class_id} {center_x} {center_y} {bbox_width} {bbox_height}\n"
    label_filename = filename.replace('.jpg', '.txt')  # Assuming images are .jpg
    with open(f"labels/{label_filename}", "a") as file:
        file.write(yolo_annotation)

# Apply conversion to each row in the CSV
df.apply(convert_to_yolo, axis=1)
