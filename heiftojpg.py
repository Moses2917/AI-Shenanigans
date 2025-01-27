from PIL import Image
from pillow_heif import register_heif_opener
from glob import glob
from re import sub
register_heif_opener()

fileNames = glob("PAPI*.HEIC",root_dir=r'M:\new downloads')
for file in fileNames:
    im = Image.open(r'M:\new downloads\{}'.format(file))
    file = sub("HEIC","JPEG",file.upper())
    im.save(r'M:\new downloads\NEW_{}'.format(file),"JPEG")