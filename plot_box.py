import os
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont

def draw_bounding_boxes(Im, bbox):
    draw = ImageDraw.Draw(Im)
    # fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 40)
    for ibox in bbox:
        tl_row,tl_col,br_row,br_col,confidence = ibox
        draw.line([(tl_col,tl_row),(tl_col,br_row)],fill=(255,0,255),width=2)
        draw.line([(tl_col,tl_row),(br_col,tl_row)],fill=(255,0,255),width=2)
        draw.line([(br_col,br_row),(br_col,tl_row)],fill=(255,0,255),width=2)
        draw.line([(br_col,br_row),(tl_col,br_row)],fill=(255,0,255),width=2)
        # draw.text((10,10),'%.1f'%confidence,font =fnt, fill = (255,255,255))
def draw_bounding_boxes_gt(Im, bbox):
    draw = ImageDraw.Draw(Im)
    # fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 40)
    for ibox in bbox:
        tl_row,tl_col,br_row,br_col = ibox
        draw.line([(tl_col,tl_row),(tl_col,br_row)],fill=(0,255,255),width=1)
        draw.line([(tl_col,tl_row),(br_col,tl_row)],fill=(0,255,255),width=1)
        draw.line([(br_col,br_row),(br_col,tl_row)],fill=(0,255,255),width=1)
        draw.line([(br_col,br_row),(tl_col,br_row)],fill=(0,255,255),width=1)
        # draw.text((10,10),'%.1f'%confidence,font =fnt, fill = (255,255,255))

data_path = '../RedLights2011_Medium'
gts_path = './hw02_annotations'       


with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts = json.load(f)
# path to json: 
preds_path = './hw02_preds' 
out_path = '/train_results/'
# os.makedirs(out_path,exist_ok=True) # create directory if needed 

# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 

# read in all bounding boxes from json
with open(preds_path+'/preds_train.json', 'r') as f:
    output = json.load(f)

file_names_train = list(output.keys())

for fi in range(len(file_names_train)):
    print('%d/%d'%(fi,len(file_names_train)))
    fname = file_names_train[fi]
    I = Image.open(os.path.join(data_path,fname))
    box = output[fname]
    gtbox = gts[fname]
    draw_bounding_boxes(I, box)
    draw_bounding_boxes_gt(I, gtbox)
    I.save(os.path.join(preds_path+out_path,file_names_train[fi]),"JPEG")



with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
    gts = json.load(f)
# path to json: 
preds_path = './hw02_preds' 
out_path = '/test_results/'
# os.makedirs(out_path,exist_ok=True) # create directory if needed 

# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 

# read in all bounding boxes from json
with open(preds_path+'/preds_test.json', 'r') as f:
    output = json.load(f)

file_names_test = list(output.keys())

for fi in range(len(file_names_test)):
    print('%d/%d'%(fi,len(file_names_test)))
    fname = file_names_test[fi]
    I = Image.open(os.path.join(data_path,fname))
    box = output[fname]
    gtbox = gts[fname]
    draw_bounding_boxes(I, box)
    draw_bounding_boxes_gt(I, gtbox)
    I.save(os.path.join(preds_path+out_path,file_names_test[fi]),"JPEG")
