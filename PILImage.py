import os
from PIL import Image
import pdf2image

# for roots, dir, files in os.walk("runs/detect/predict25"):
#
#     for file in files:
#         print(roots+file)
#         im = Image.open(roots+"/"+file)
#         im.resize([128,128]).rotate(-90).save(file,"JPEG")
#

# Image.open(r'656_Townsend_pg_19.pdf').save(r"M:/new downloads/656 Townsend pg 19.png","PNG")

images = pdf2image.convert_from_path("M:/new downloads/MTM_PROJ_MASON_Model.pdf", poppler_path="M:/poppler/Library/bin")
# image[0].save("M:/new downloads/656 Townsend pg 20.png")
index = 0
for image in images:
    image.save("M:/new downloads/MTM_PROJ_MASON_Model {}.png".format(index))
    index += 1