## Import Libraries
import shutil
import os
import pathlib
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

## Create Constants
NUMS=[str(i)for i in range(10)]
SMALL=[chr(ord('a')+i)for i in range(26)]
CAPS=[chr(ord('A')+i)for i in range(26)]
COMBINED=NUMS+SMALL+CAPS

IMAGE_SHAPE=(32,32)
FONT_SIZE=28

font_paths=list(map(str,list(pathlib.Path('fonts/').glob('*.ttf'))))

## Base Image Creation
base_image=np.ones(IMAGE_SHAPE)*255
cv2.imwrite('base_image.png',base_image)
base_image=Image.open('base_image.png')

## Fonts Dataset Creation
try:
    shutil.rmtree('dataset')
except:
    pass
os.mkdir('dataset')

def center_text(img, font, text, color='black'):
    draw = ImageDraw.Draw(img)
    text_width, text_height = draw.textsize(text, font)
    position = ((IMAGE_SHAPE[1]-text_width)/2,(IMAGE_SHAPE[0]-text_height)/2)
    draw.text(position, text, color, font=font)
    return img

for char in COMBINED:
    count=0
    os.mkdir(f'dataset/{char}')
    for font_path in font_paths:
        font_name=font_path.split('/')[-1][:-4]
        base_image=Image.open('base_image.png')
        font_style = ImageFont.truetype(font_path, FONT_SIZE)
        base_image=center_text(base_image,font_style,char)
        base_image.save(f'dataset/{char}/{font_name}.png')
        count+=1
        

## Adding Augmented Images
if str(input('>>> Augment the images?? (Y/N)   : ')).upper()=='Y':
    image_paths=list(map(str,list(pathlib.Path('dataset').glob('*'))))
    for image_path in image_paths:
        char=image_path.split('/')[-1]
        datagen=ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        )
        generator=datagen.flow_from_directory('dataset',
        classes=[char],
        save_format='png',
        save_to_dir=image_path,
        batch_size=64,
        target_size=IMAGE_SHAPE,)
        generator.next()
        generator.next()
else:
    pass
print("We are Done")