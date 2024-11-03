from ultralytics import YOLO

model = YOLO(model='yolov5n.pt')

model.train(data='config.yaml',epochs=30,patience=10)