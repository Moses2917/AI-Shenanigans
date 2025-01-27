import os

import gradio as gr
from glob import glob
import split_image
import pandas as pd

def calculate_split_number(img_size, split_factor=800):
    # Calculate the optimal split number pair based on the desired output image size
    target_size = img_size / split_factor
    split_width = int(target_size)
    split_height = int(target_size)
    return split_width, split_height

def clean_up(directory_path):
    # for file in glob(r"M:\PyCharm\Project\temp"):
    #     os.remove(file)
    import shutil
    shutil.rmtree(directory_path)

def sort(filePaths):
    from re import findall
    filePathCopy = filePaths[:]
    nums = []
    for imgPath in filePaths: #gets file number, stores in list nums, sorts that list and then reorginizes filePaths in that order
        x = (findall(r'\d+',imgPath))[0]
        nums.append(x)
    nums.sort(key=lambda x: int(x))
    for imgPath in filePaths:
        sortedIndex = nums.index((findall(r'\d+',imgPath))[0])
        filePathCopy[sortedIndex] = imgPath
    return filePathCopy



def predict(image_path):
    from ultralytics import YOLO
    from PIL import Image

    im = Image.open(image_path)
    wid, hgt, = im.size
    im.save("tempFile.png",'PNG')
    image_path = "tempFile.png"
    if wid>hgt: img_size = wid
    else: img_size = hgt
    split_width, split_height = calculate_split_number(img_size)


    split_image.split_image(image_path, split_width, split_height, should_square=False, should_cleanup=False, output_dir='temp')

    split_images = glob(r'M:\PyCharm\Project\temp\*')

    class_names = ['220 Volt', 'Bathroom Fan', 'Ceiling Fan', 'Ceiling Light', 'Duplex Outlet', 'GFCIOutlet',
                   'Smoke Detector', 'Switch', 'Wall Switched Outlet', 'WallLight', 'WaterProof GFCI', 'ress_can_light']

    classes = []

    model = YOLO("houseplan/Colab/v21/best.pt")
    # Process each split image
    for x in split_images:
        results = model(x, box=True, save=True,project="xxx", name="yyy")
        for result in results:
            boxes = result.boxes
            class_indices = boxes.cls.int()

            class_labels = [class_names[i] for i in class_indices]
            for cls in class_labels:
                classes.append(cls)

    # Count occurrences of each class
    class_counter = {class_name: classes.count(class_name) for class_name in class_names}
    df = pd.DataFrame(list(class_counter.items()), columns=['Class', 'Count'])

    # Display the class counts
    for class_name, count in class_counter.items():
        print(f"{class_name}: {count}")

    # Reverse the split (if needed)
    filePaths = glob(r"M:\PyCharm\Project\xxx\yyy\*")
    filePaths = sort(filePaths)
    split_image.reverse_split(filePaths, split_width, split_height, image_path=r"M:\PyCharm\Project\tempFile.png",should_cleanup=False)

    # Return the result (you can modify this based on your needs)
    clean_up(r"M:\PyCharm\Project\temp")
    clean_up(r"M:\PyCharm\Project\xxx\yyy")
    return df,r"M:\PyCharm\Project\tempFile.png"


# Create a Gradio interface
# with gr.Row():
#     plot = gr.BarPlot(
#         x="Item",
#         y="Count",
#         title="AI bid",
#         x_title="Different items",
#         y_title="All nums",
#         height=300,
#         width=400
#     )
im = gr.Image(

    label="Marked Plan",
    interactive=False,
    show_label=True
)
# txt = gr.TextArea(
#     label="Items Total"
# )
pl = gr.BarPlot(
    x="Class",
    y="Count",
    label="Marked Plan",
    interactive=True,
    show_actions_button=True,
)
data = gr.Dataframe(
    label="Items Total"
)

fl = gr.File(
    file_count='single',
    file_types=['image'],
    label="Please upload an image file of your plan"

)
# gr.Image(type='filepath', label='Input Image',sources=['upload', 'clipboard'])
iface = gr.Interface(
    fn=predict,
    inputs=fl,
    outputs=[data,im], # You can modify this based on your model's output
    live=True,
    title="ATM Tech Advanced AI Bidding Software",
    allow_flagging='never'
)

iface.launch()
