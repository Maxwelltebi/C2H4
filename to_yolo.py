
import pandas as pd
import os

df = pd.read_csv("D:/Technologia/ML/C2H4/dataset/test/_annotations.csv")

os.makedirs("labels", exist_ok=True)

def convert_to_yolo(row):
    filename = row['filename']
    width = row['width']
    height = row['height']
    xmin = row['xmin']
    ymin = row['ymin']
    xmax = row['xmax']
    ymax = row['ymax']

    class_id = 0 if row['class'] == 'plastic' else 1

    center_x = (xmin + xmax) / 2.0 / width
    center_y = (ymin + ymax) / 2.0 / height
    bbox_width = (xmax - xmin) / width
    bbox_height = (ymax - ymin) / height

    label_name = filename.replace('.jpg', '.txt')

    with open(f"labels/{label_name}", "a") as f:
        f.write(f"{class_id} {center_x} {center_y} {bbox_width} {bbox_height}\n")

df.apply(convert_to_yolo, axis=1)
