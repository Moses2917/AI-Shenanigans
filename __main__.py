from ultralytics import YOLO
# import torch
#
# if torch.cuda.is_available():
#     device = torch.device('cuda')

model = YOLO("yolov8s.pt") # load the yolov8n model

# For training a new model
if __name__ == '__main__':
    model.train(data="data.yaml", epochs=1000, patience=50, imgsz=1280, project="houseplan",name="elecStuffBigger", batch=5, val=False,cache= "ram_cache")

#for resuming training
# model= YOLO("houseplan/elecStuffBigger/weights/best.pt")
# if __name__ == '__main__':
#     model.train(data="data.yaml", epochs=350, imgsz=1280, project="houseplan",name="elecStuffBigger", batch=5, val=False, cache= "ram_cache")