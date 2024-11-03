import cv2
from ultralytics import YOLO
from casp import run_asp_solver


import cv2
from ultralytics import YOLO
from casp import run_asp_solver

def run(filename):
    

    # Load video capture object using cv2
    cap = cv2.VideoCapture(filename)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        res = pipeline(frame)
        print(res)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def pipeline(frame):
    # Load the YOLOv5 model
    model = YOLO(model=r'runs\detect\train6\weights\best.pt')
    results = model(frame)
    annotated_frame = results[0].plot()
    boxes = results[0].boxes
    use_detections = []
    for box in boxes:
        coords = box.xyxy[0].tolist()

        class_id = int(box.cls[0])
        color = results[0].names[class_id]

        confidence = float(box.conf[0])
        if color!="No color":
            use_detections.append({"bbox": coords, "confidence": confidence, "color": color})
    # Only show predictions if objects are detected
    if len(use_detections) > 0:
        cv2.imshow("Predictions", annotated_frame)
                # Get detection results - ensure indices exist before accessing
        if len(results[0].boxes.cls) != -1: #arbitrary rn
           left_light = get_left_light(use_detections)
           light_color = left_light["color"]  # Traffic light
           print("LIGHT COLOR",light_color)
           if light_color == "no color":
               return "unknown"
           cars_detected = True #bool(results[0].boxes.cls[1])  # Cars
           turn_signal_on = True #bool(results[0].boxes.cls[2])  # Turn signal
           res = run_asp_solver(light_color, cars_detected, turn_signal_on)
           return res

def get_left_light(detections):
    # If no detections, return default
        if not detections:
            return {"bbox": [0,0,0,0], "confidence": 0, "color": "no color"}
        
        # Find leftmost detection by comparing x coordinates of bounding boxes
        leftmost = detections[0]
        leftmost_x = detections[0]["bbox"][0]
        
        for detection in detections:
            current_x = detection["bbox"][0]
            if current_x < leftmost_x:
                leftmost = detection
                leftmost_x = current_x
                
        return leftmost
    
img = cv2.imread(r"data\images\train\00004.jpg")
print(pipeline(img))