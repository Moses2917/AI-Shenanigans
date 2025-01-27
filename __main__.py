from ultralytics import YOLO
from roboflow import Roboflow

def download():
    rf = Roboflow(api_key="LxQNlz0xUnCkdOdNwMRd")
    project = rf.workspace("movses-movsesyan-pnofn").project("elec-stuff")
    dataset = project.version(22).download("yolov8")

# download()

model = YOLO("yolov8s.pt") # load the yolov8n model

# For training a new model
if __name__ == '__main__':
    model.train(data="M:/PyCharm/Project/datasets/elec-stuff-22/data.yaml", epochs=75, patience=5, imgsz=256, project="houseplan",name="elecStuffIndivImgs", batch=-1, val=False) #use error to see if batch size can go up or dowen

#for resuming training
# model= YOLO("houseplan/elecStuffBigger/weights/best.pt")
# if __name__ == '__main__':
#     model.train(data="data.yaml", epochs=350, imgsz=1280, project="houseplan",name="elecStuffBigger", batch=5, val=False, cache= "ram_cache")