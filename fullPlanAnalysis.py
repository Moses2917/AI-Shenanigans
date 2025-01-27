# Import patchify library
from glob import glob
import reportlab
from reportlab.platypus import SimpleDocTemplate, Paragraph
report = SimpleDocTemplate("report.pdf")
report_title = Paragraph("A Complete Inventory of My Fruit")
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie
report_pie = Pie(width=3, height=3)
import split_image
from ultralytics import YOLO
# Load an image as a numpy array
import numpy as np
from PIL import Image
import cv2 as cv2
# webcam = cv2.VideoCapture(0)
from pdf2image import convert_from_path

def idkow():
    image = np.array(Image.open('large_image.png'))

    # Define patch size and step size
    patch_size = (2048, 2048)
    step_size = (1920, 1440)

    # Split the image into patches with patchify
    # patches = patchify(image, patch_size, step=step_size)

    # model = YOLO("houseplan/elecStuffBigger/weights/best.pt")
# model = YOLO("yolov8n-seg.pt")
def idk():

    # Run each patch through your custom YOLO model and get the output
    outputs = []
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patch = patches[i,j,:,:]
            output = model.predict(patch) # Replace this with your model function
            outputs.append(output)
            # Print some attributes of results
            print(output.labels)  # List of lists of class labels for each patch
            print(output.scores)  # List of lists of confidence scores for each patch
            print(output.boxes)  # List of lists of bounding boxes for each patch

    # Merge the outputs back into the original image shape
    merged_output = unpatchify(np.array(outputs), image.shape)

#YOLOv8 webcam
def fun():
    # output = model.predict("M:/new downloads/plan ag..v3i.multiclass/test/large_image_png.rf.ca11051192b1dc365bbe89f274fecfed.jpg", save=True, box=True)
    url = "https://wzmedia.dot.ca.gov/D5/1atMunrasAve.stream/chunklist_w23641507.m3u8"
    # url  ="https://wzmedia.dot.ca.gov/D5/1atMunrasAve.stream/playlist.m3u8"
    # url = "https://embedwistia-a.akamaihd.net/deliveries/e34aff6a88bad833560cba056370a2f714694345.m3u8" ##Some hospital ad :))
    url = "https://youtu.be/1-iS7LArMPA"
    def web():
        while True:
            ret, frame = webcam.read()
            output = model(frame,box = True, show=True)

    output = model(url,box = True, show=True)

    # print(output.labels)  # List of lists of class labels for each patch
    # print(output.scores)  # List of lists of confidence scores for each patch
    # print(output.boxes)  # List of lists of bounding boxes for each patch

trained = "houseplan/elecStuffBigger23/weights/best.pt"
# model = YOLO(trained)
model = YOLO(r"houseplan\elecStuffIndivImgs3\weights\best.pt") #Old Value: "houseplan/Colab/v21/best.pt"

# limg = convert_from_path("M:/PyCharm/Project/valid.pdf", poppler_path ="M:/poppler", output_file="Limg.jpg",single_file=True)

# split_image.split_image("Large_image.png",10,10,should_square=False, should_cleanup=False) # to undo run split-image "Large_image.png" 10 10 -r
split_image.split_image("656 Townsend.png",8,8,should_square=False, should_cleanup=False)
Split_Images = glob("656 Townsend" + "*.png") ## VIA "split_image(image,8,8,should_square=False, should_cleanup=False)"
class_names = ['220 Volt', 'Bathroom Fan', 'Ceiling Fan', 'Ceiling Light', 'Duplex Outlet', 'GFCIOutlet',
               'Smoke Detector', 'Switch', 'Wall Switched Outlet', 'WallLight', 'WaterProof GFCI', 'ress_can_light']
class_counter = []
classes = []

for x in Split_Images:
    results = model(x,box=True,save=True)
    for result in results:
        boxes = result.boxes
        class_indices = boxes.cls.int()

        class_labels = [class_names[i] for i in class_indices]
        for cls in class_labels:
            classes.append(cls)
            # print(cls)

#this is prolly a slow counter, but for now its ok
count = 0
for i in class_names:
    for x in classes:
        if i == x:
            count = count + 1
    print(i + ": " + str(count)) # 220 Volt: 12
    count = 0 #resets count var

# split_image.reverse_split("/runs/detect/predict27",10,10,should_cleanup=False, image_path="/runs/detect/predict27/652 Townsend.png")