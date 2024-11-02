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
    print(results[0].boxes.cls)
    # Only show predictions if objects are detected
    if len(results[0].boxes) > 0:
        cv2.imshow("Predictions", annotated_frame)
                # Get detection results - ensure indices exist before accessing
        if len(results[0].boxes.cls) != -1: #arbitrary rn
           light_color = results[0].names[int(results[0].boxes.cls[0])].lower()  # Traffic light
           print("LIGHT COLOR",light_color)
           if light_color == "no color":
               return "unknown"
           cars_detected = True #bool(results[0].boxes.cls[1])  # Cars
           turn_signal_on = True #bool(results[0].boxes.cls[2])  # Turn signal
           res = run_asp_solver(light_color, cars_detected, turn_signal_on)
           return res

img = cv2.imread(r"data\images\train\00004.jpg")
print(pipeline(img))
