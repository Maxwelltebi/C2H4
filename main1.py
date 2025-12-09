import cv2
import torch
import tensorflow as tf
import matplotlib.pyplot as plt

print("Libraries installed successfully!")

# Load the trained model
model = torch.load("runs/train/exp/weights/best.pt")

# Initialize video capture
cap = cv2.VideoCapture(0)  # 0 for default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    results = model(frame)  # Run inference on the frame

    # Extract bounding boxes and labels
    boxes = results.xywh[0]  # Get bounding boxes for plastic objects
    for box in boxes:
        # Draw bounding boxes
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show the frame with detected plastics
    cv2.imshow("Plastic Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
