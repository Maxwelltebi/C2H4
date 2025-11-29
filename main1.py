import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input , decode_predictions

model = tf.keras.applications.MobileNetV2(weights='imagenet')

PLASTIC_KEYWORDS = [
    "bottle", "packet", "plastic", "bag", "water bottle",
    "container", "cup"
]

cap = cv2.VideoCapture(0)  # default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize to model input size 224x224
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Make prediction
    preds = model.predict(img)
    decoded = decode_predictions(preds, top=1)[0][0]
    label = decoded[1]  # predicted class name (string)

    # Determine plastic / non-plastic
    if any(keyword in label.lower() for keyword in PLASTIC_KEYWORDS):
        text = f"Plastic ({label})"
        color = (0, 255, 0)
    else:
        text = f"Non-Plastic ({label})"
        color = (0, 0, 255)

    # Display result
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Plastic Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()