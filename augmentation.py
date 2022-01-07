from io import StringIO
import PIL
import absl
import torch
import os
import os.path
import torchvision
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as F
import numpy as np
import random
import cv2
import argparse
from PIL import Image, ImageDraw
from IPython.display import display
from absl import flags
from absl import app

parser = argparse.ArgumentParser(description='Skip')
# parser.add_argument('-i', '--input_dir', type=str, metavar='', required=True, help='To choose directory')
parser.add_argument('-mode', '--mode', type=list, metavar='', required=True, help='Mode : 1. Flip, 2. Contrast, 3. Saturation')
args = parser.parse_args()


def init(file_image, file_xml):
    image = Image.open(file_image)
    image = image.convert("RGB")
    objects = parse_annot(file_xml)
    boxes = torch.FloatTensor(objects['boxes'])
    labels = torch.LongTensor(objects['labels'])
    difficulties = torch.ByteTensor(objects['difficulties'])
    # draw_PIL_image(image, boxes, labels)
    return image, boxes, labels
    
def parse_annot(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    boxes = list()
    labels = list()
    difficulties = list()
    
    for object in root.iter("object"):
        difficult = int(object.find("difficult").text == "1")
        label = object.find("name").text.lower().strip()
        if label not in label_map:
            print("{0} not in label map.".format(label))
            assert label in label_map
            
        bbox = object.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)
        
    return {"boxes":boxes, "labels":labels, "difficulties":difficulties}

def draw_PIL_image(image, boxes, labels):
    if type(image) != PIL.Image.Image:
        image = F.to_pil_image(image)
    new_image = image.copy()
    labels = labels.tolist()
    draw = ImageDraw.Draw(new_image)
    boxes =  boxes.tolist()
    for i in range(len(boxes)):
        draw.rectangle(xy=boxes[i], outline=label_color_map[rev_label_map[labels[i]]])
    display(new_image)    
    
#augmentation mode
def Adjust_contrast(image, boxes):
    new_image = F.adjust_contrast(image, 2)
    return new_image, boxes

def Adjust_saturation(image,boxes):
    new_image = F.adjust_saturation(image, 2)
    return new_image, boxes
def Adjust_brightness(image, boxes):
    new_image = F.adjust_brightness(image, 2)
    return new_image, boxes

def random_blur(image, boxes):
    a = random.randrange(5, 20, 2) #start, end, step, can change you like it
    b = random.randrange(5, 20, 2)
    print(a, b)
    new_image = F.gaussian_blur(image, (a,b))
    return new_image, boxes

def h_flip(image, boxes):
    new_image = F.hflip(image)
    
    #flip boxes 
    new_boxes = boxes.clone()
    new_boxes[:, 0] = image.width - boxes[:, 0]
    new_boxes[:, 2] = image.width - boxes[:, 2]
    new_boxes = new_boxes[:, [2, 1, 0, 3]]
    new_boxes = new_boxes.numpy()
    return new_image, new_boxes

def v_flip(image, boxes):
    new_image = F.vflip(image)
    
    #flip boxes 
    new_boxes = boxes.clone()
    new_boxes[:, 1] = image.height - boxes[:, 1]
    new_boxes[:, 3] = image.height - boxes[:, 3]
    new_boxes = new_boxes[:, [2, 1, 0, 3]]
    return new_image, new_boxes


## File Upgrade ##

def save_image(file_image, new_image, id_mode):
    new_file = file_image[:-4]
    if (id_mode == '1'):
        mode = "hflip"
    if (id_mode == '2'):
        mode = "vflip"
    if (id_mode == '3'):
        mode = "contrast"
    if (id_mode == '4'):
        mode = "blur"
    if (id_mode == '5'):
        mode = "brightness"
    if (id_mode == '6'):
        mode = "saturation"
    if (id_mode == '7'):
        mode = "h_flip_blur"
    if (id_mode == '8'):
        mode = "v_flip_blur"
    if (id_mode == '9'):
        mode = "h_v_flip_blur"
    new_file = new_file + "_" + mode + ".jpg"
    new_image.save(new_file)
    print(new_file + "telah tersimpan")
    
    
def save_xml(file_xml, boxes, id_mode):
    new_file = file_xml[:-4]
    if (id_mode == '1'):
        mode = "hflip"
    if (id_mode == '2'):
        mode = "vflip"
    if (id_mode == '3'):
        mode = "contrast"
    if (id_mode == '4'):
        mode = "blur"
    if (id_mode == '5'):
        mode = "brightness"
    if (id_mode == '6'):
        mode = "saturation"
    if (id_mode == '7'):
        mode = "h_flip_blur"
    if (id_mode == '8'):
        mode = "v_flip_blur"
    if (id_mode == '9'):
        mode = "h_v_flip_blur"
    new_file = new_file + "_" + mode + ".xml"
    box = np.array(boxes)
    i = 0
    tree = ET.parse(file_xml)
    root = tree.getroot()
    new_filename = os.path.basename(file_xml).split(".")[0] + "_" + mode + ".jpg"
    new_folder = os.path.basename(cwd)
    root.find("filename").text = str(new_filename)
    root.find("path").text = str(new_file)
    root.find("folder").text = str(new_folder)
    for object in root.iter("object"):
        bbox = object.find("bndbox")
        bbox.find("xmin").text = str(int(box[i][0]))
        bbox.find("ymin").text = str(int(box[i][1]))
        bbox.find("xmax").text = str(int(box[i][2]))
        bbox.find("ymax").text = str(int(box[i][3]))
        i+=1
    tree.write(new_file)
    print(new_file + "telah tersimpan")

if __name__ == "__main__":
    #setup variables
    voc_labels = ('ball', 'post', 'line_x', 'line_t', 'obstacle', 'white_ball')
    label_map = {k:v+1 for v, k in enumerate(voc_labels)}
    rev_label_map = {v:k for k, v in label_map.items()}
    classes = 6
    distinct_colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                      for i in range(classes)]
    label_color_map = {k:distinct_colors[i] for i, k in enumerate(label_map.keys())}
    mode = args.mode
    print(mode)

    #load image
    list_image = list()
    list_xml = list()
    cwd = os.getcwd()
    for item in os.listdir(cwd):
        if item.endswith(".jpg"):
            if not (os.path.splitext(item)[0] + ".xml") in os.listdir(cwd):
                print("tidak ada file xml")
                break
            else :
                file_image = os.path.join(cwd, item)
                list_image.append(file_image)
                file_xml = os.path.splitext(file_image)[0] + ".xml"
                list_xml.append(file_xml)

    for i in range(len(list_image)):
        image, boxes, labels = init(list_image[i], list_xml[i])
        for j in range(len(mode)):
            if(mode[j] == '1'):
                new_image, new_boxes = h_flip(image, boxes)
            if(mode[j] == '2'):
                new_image, new_boxes = v_flip(image, boxes)            
            if(mode[j] == '3'):
                new_image, new_boxes = Adjust_contrast(image, boxes)
            if(mode[j] == '4'):
                new_image, new_boxes = random_blur(image, boxes)
            if(mode[j] == '5'):
                new_image, new_boxes = Adjust_brightness(image, boxes)
            if(mode[j] == '6'):
                new_image, new_boxes = Adjust_saturation(image, boxes)
            if(mode[j] == '7'):
                new_image, new_boxes = h_flip(image, boxes)
                new_image, new_boxes = random_blur(new_image, new_boxes)
            if(mode[j] == '8'):
                new_image, new_boxes = v_flip(image, boxes)
                new_image, new_boxes = random_blur(new_image, new_boxes)
            # if(mode[j] == '9'):
            #     new_image, new_boxes = h_flip(image, boxes)
            #     new_image, new_boxes = v_flip(new_image, new_boxes)
            #     new_image, new_boxes = random_blur(new_image, new_boxes)
                        
            save_image(list_image[i], new_image, mode[j])
            save_xml(list_xml[i], new_boxes, mode[j])
