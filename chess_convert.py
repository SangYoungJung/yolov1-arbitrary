# Copyright 2023. (lycobs@gmail.com) all rights reserved.

import glob
import cv2
import time
import os
import random
import numpy as np


def chess_piece_imwrite(img, name=None):
    name = name if name else str(round(time.time() * 1000))+'.png'
    cvt = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(name , cvt)


def chess_to_yolo_bbox(bbox, h, w):
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width    = (bbox[2] - bbox[0]) / w
    height   = (bbox[3] - bbox[1]) / h
    return (x_center, y_center, width, height)


def chess_write_yolo_classes(location, classes):
    os.makedirs(os.path.dirname(location), exist_ok=True)
    with open(os.path.dirname(location)+'/classes.txt', "w") as file:
        for class_name in classes:
            file.writelines(class_name + '\n')


def chess_write_yolo_format(location, image, bbox, classes):
    os.makedirs(os.path.dirname(location), exist_ok=True)
    # Create a Image    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(location+'.png', image)
    # Create a annotation as <object-class> <x> <y> <width> <height>
    if bbox is not None:
        with open(location+'.txt', "w") as file:
            for box in bbox:
                msg = '{} {:.5f} {:.5f} {:.5f} {:.5f}\n'.format(classes.index(box[0]), 
                                                                box[1][0], box[1][1], 
                                                                box[1][2], box[1][3])
                file.writelines(msg)
        print(location, len(bbox))


def chess_piece_segmentation(location, rows, cols):
    segmentations = []
    for idx, name in enumerate(sorted(glob.glob(location))):
        png, txt = name, name.replace('.png', '.txt')

        # Read chess piece
        txt_file = open(txt, 'r')
        pieces_name = [entry.strip().split(',') for entry in txt_file.readlines()]
        pieces_name = sum(pieces_name, [])
        
        # Read Images
        img = cv2.imread(png)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Grid 8 by 8
        height = int(img.shape[0] / rows)
        width  = int(img.shape[1] / cols)

        # Segmentation each chess piece
        pieces_img = []
        for row in range(rows):
            for col in range(cols):
                y0 = row * height
                y1 = y0 + height
                x0 = col * width
                x1 = x0 + width
                roi = img[y0:y1, x0:x1]
                pieces_img.append(roi)
        
        # Concatenate lists into one
        segmentations += [ {pieces_name[i] : pieces_img[i]} for i in range(len(pieces_name)) ]
    return segmentations, height, width


def chess_piece_augmentation(segmentations, h, w, rows, cols):
    random.shuffle(segmentations)
    bbox = []
    img = np.zeros((h*rows, w*cols,3), dtype=np.uint8)
    for row in range(rows):
        for col in range(cols):
            y0 = row * h   # top
            y1 = y0  + h   # bottom
            x0 = col * w    # left
            x1 = x0  + w    # right
            id = row*rows + col
            name = str(list(segmentations[id].keys())[0]).strip()
            # print(str.ljust(str(id), 3), str.ljust(name,10), segmentations[id][name].shape)
            img[y0:y1, x0:x1] = segmentations[id][name]
            # if len(name) != 0 : bbox.append({ 'piece':name,'x0':x0,'y0':y0,'x1':x1,'y1':y1 })
            if len(name) != 0 : bbox.append([name,chess_to_yolo_bbox([x0,y0,x1,y1], h*rows, w*cols)])
    return img, bbox


#===============================================================================
# Main
#===============================================================================

if __name__ == "__main__":
    num_data = 512
    rows = 8
    cols = 8
    segmentations, height, width = chess_piece_segmentation('./data/chess/*.png', rows, cols)

    classes = [ list(entry.keys())[0] for entry in segmentations ] # Extract class
    classes = list(set(classes)) # Remove duplicated class
    classes = list(filter(None, classes)) # Remove empty class
    chess_write_yolo_classes('./data/dataset/', classes)

    for idx in range(num_data):
        img, bbox = chess_piece_augmentation(segmentations, height, width, rows, cols)
        chess_write_yolo_format('./data/dataset/i_{:08}'.format(idx), img, bbox, classes) 
