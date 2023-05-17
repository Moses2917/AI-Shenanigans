from ultralytics import YOLO
#based upon Ultralytics YOLOv8.0.43

# Load a model
# model = YOLO("yolov8n.pt")  # load an official model
model = YOLO("path/to/best.pt")  # load a custom model
# Predict with the model
results = model("bus.jpg")  # predict on an image
# print(model.predict("bus.jpg",box=True,save=True))

# Count objects of specific class
class_name = 'person'
num_objects = 0

boxes = results[0].boxes
box = boxes[0].cls[0]
print(str(boxes) + "\n" + str(box))

# print('Number of', class_name, 'objects detected:', num_objects)