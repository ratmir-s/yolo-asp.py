from ultralytics import YOLO
import torch
from ultralytics.nn.tasks import DetectionModel
import cv2
import os

#Add the class to the safe globals

torch.serialization.add_safe_globals([DetectionModel])

model = YOLO('bestnn.pt')  # Path to your custom YOLO weights

# Load an image
img_path = '00385.jpg'  # Replace with your image path
img = cv2.imread(img_path)  # Load the image

confidence_threshold = 0.25

# Run inference
results = model.predict(img,conf=confidence_threshold)  # Use the .predict() method for inference

'''print(results)

for result in results:
    result.plot()  # This will plot the detections on the image

# Optionally save results to a folder
for i, result in enumerate(results):
    result.save(f'output_image_{i}.jpg')  # Save each result as an output image'''


for result in results:
    boxes = result.boxes.xyxy  # Bounding box coordinates
    scores = result.boxes.conf  # Confidence scores
    labels = result.boxes.cls   # Class labels (e.g., traffic light colors)

    # Loop over detections
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        if int(labels[i]) == 0:
            label = "no color"
        elif int(labels[i]) == 1:
            label = "red"
        elif int(labels[i]) == 2:
            label = "yellow"
        elif int(labels[i]) == 3:
            label = "green"

        label = f"{label} {scores[i]:.2f}"

        # Adjust font scale and thickness to avoid overlapping text
        font_scale = 0.5
        thickness = 1

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Example: green box

        # Draw label text above the box with background rectangle
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255, 255, 255), -1)  # White background for text
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

# Show the image with adjusted labels
cv2.imshow('Detections', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the image
cv2.imwrite('output_image.jpg', img)
