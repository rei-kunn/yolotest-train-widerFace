from ultralytics import YOLO

# model = YOLO('best.pt')
# model.export(format='saved_model')

model = YOLO("yolov8n.pt")
model.export(format="onnx")