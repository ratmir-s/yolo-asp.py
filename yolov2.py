import torch
import cv2
from ultralytics import YOLO
# Add the class to the safe globals
from ultralytics.nn.tasks import DetectionModel

torch.serialization.add_safe_globals([DetectionModel])

model = YOLO('bestnn.pt')  # Path to your custom YOLO weights


def get_yolo_output(img_path='01221.jpg'):
    img = cv2.imread(img_path)  # Load the image
    confidence_threshold = 0.25
    results = model.predict(img, conf=confidence_threshold)

   # Initialize default values
    light_color = "none"
    cars_detected = False
    turn_signal_on = False

    for result in results:
        boxes = result.boxes.xyxy  # Bounding box coordinates
        scores = result.boxes.conf  # Confidence scores
        labels = result.boxes.cls  # Class labels (e.g., traffic light colors)

        # Loop over detections to determine the traffic light color
        for i, box in enumerate(boxes):
            if int(labels[i]) == 1:
                light_color = "red"
            elif int(labels[i]) == 2:
                light_color = "yellow"
            elif int(labels[i]) == 3:
                light_color = "green"
            # Add logic for detecting cars and turn signals if labeled in the model
            # For this example, `cars_detected` and `turn_signal_on` remain False

    return {
        'light_color': light_color,
        'cars_detected': cars_detected,
        'turn_signal_on': turn_signal_on
    }

print(get_yolo_output())
