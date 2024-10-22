from ultralytics import YOLO
import torch
from ultralytics.nn.tasks import DetectionModel
import cv2
import os

# Add the class to the safe globals
torch.serialization.add_safe_globals([DetectionModel])

# Load the YOLO model
model = YOLO('bestnn.pt')  # Path to your custom YOLO weights

# Set the confidence threshold
confidence_threshold = 0.25

# Directory containing images (replace with your directory)
image_folder = 'test_images'  # Replace with the path to the folder containing your images

# Output directory for saving results
output_folder = 'output_images'
os.makedirs(output_folder, exist_ok=True)

# Loop through each image in the folder
for image_file in os.listdir(image_folder):
    img_path = os.path.join(image_folder, image_file)

    # Load the image
    img = cv2.imread(img_path)

    if img is None:
        print(f"Failed to load {image_file}")
        continue

    # Run inference on the image
    results = model.predict(img, conf=confidence_threshold)

    # Process the results
    for result in results:
        boxes = result.boxes.xyxy  # Bounding box coordinates
        scores = result.boxes.conf  # Confidence scores
        labels = result.boxes.cls  # Class labels (e.g., traffic light colors)

        # Loop over detections
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            label = int(labels[i])
            if label == 0:
                label_text = "no color"
            elif label == 1:
                label_text = "red"
            elif label == 2:
                label_text = "yellow"
            elif label == 3:
                label_text = "green"

            label_text = f"{label_text} {scores[i]:.2f}"

            # Adjust font scale and thickness
            font_scale = 0.5
            thickness = 1

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box

            # Draw label text above the box with background rectangle
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255, 255, 255), -1)  # White background for text
            cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    # Save the processed image
    output_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_path, img)
    print(f"Processed and saved {output_path}")
