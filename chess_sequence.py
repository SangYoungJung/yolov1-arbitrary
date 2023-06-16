# Copyright 2023. (lycobs@gmail.com) all rights reserved.

import math
import os
import cv2
import numpy as np
import tensorflow as tf

class ChessSequence(tf.keras.utils.Sequence):
    def __init__(self, data_loc, class_loc, batch_size=64, input_shape=(448, 448, 3), shuffle=True):
        
        self.batch_size  = batch_size
        self.input_shape = input_shape[0:2]
        self.shuffle     = shuffle

        self.datasets = []
        with open(data_loc, 'r') as f:
            self.datasets += f.readlines()
        self.indexes = np.arange(len(self.datasets))

        self.classes = []
        with open(class_loc, 'r') as f:
            self.classes += f.readlines()
        self.classes = [ entry.strip() for entry in self.classes]


    def __len__(self):
        return math.ceil(len(self.datasets) / float(self.batch_size))


    def __getitem__(self, index):
        batch_indexs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.datasets[k] for k in batch_indexs]
        X, y = self.data_generation(batch)
        return X, y
    

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.indexes)


    def data_generation(self, batch_datasets):
        images = []
        labels = []

        for dataset in batch_datasets:
            # print(dataset.strip())
            image, label = self.read(dataset.strip())
            images.append(image)
            labels.append(label)

        X = np.array(images)
        y = np.array(labels)

        return X, y
    
    
    def read(self, dataset):
        png, txt = dataset, dataset.replace('.png', '.txt')
        matrix = np.zeros([7, 7, 25])

        # Image Read
        image = cv2.imread(png)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        image_h, image_w = image.shape[0:2]
        image = cv2.resize(image, self.input_shape)
        image = image / 255. # Normalization [0,1]
        input_h, input_w = image.shape[0:2]

        # Annotation Read
        txt_file = open(txt, 'r')
        annotation = [ entry.strip().split(',') for entry in txt_file.readlines()]
        for idx, an in enumerate(annotation):
            label = int(an[0])
            xmin, ymin, xmax, ymax = int(an[1]), int(an[2]), int(an[3]), int(an[4])
            # print(str.ljust(self.classes[label], 10), xmin, ymin, xmax, ymax)

            # Re-scale
            xmin = int(xmin * (input_w / image_w))
            ymin = int(ymin * (input_h / image_h))
            xmax = int(xmax * (input_w / image_w))
            ymax = int(ymax * (input_h / image_h))
            # print(str.ljust('Re-scale', 10), xmin, ymin, xmax, ymax)

            # YOLO coordinate
            x = (xmin + xmax) / 2 / input_w
            y = (ymin + ymax) / 2 / input_h
            w = (xmax - xmin) / input_w
            h = (ymax - ymin) / input_h
            # print(str.ljust('YOLO', 10), x, y, w, h)

            xy = [7 * x, 7 * y]
            xy_i, xy_j = int(xy[1]), int(xy[0])
            y = xy[1] - xy_i
            x = xy[0] - xy_j
            if matrix[xy_i, xy_j, 24] == 0:
                matrix[xy_i, xy_j, label] = 1
                matrix[xy_i, xy_j, 20:24] = [x, y, w, h]
                matrix[xy_i, xy_j, 24] = 1
                print(xy_i, xy_j, label, x, y, w, h)


        return None, None


train_loc = './data/dataset/chess-train.txt'
class_loc = './data/dataset/classes.txt'
seq = ChessSequence(train_loc, class_loc, batch_size=1)
seq.__getitem__(0)




        # for an in annotation:
        #     label = an.split(',')



        # label = 


        # image_path = dataset[0]
        # label = dataset[1:]

        
         
        
        
        

        # label_matrix = np.zeros([7, 7, 25])
        # for l in label:
        #     l = l.split(',')
        #     l = np.array(l, dtype=np.int)
        #     xmin = l[0]
        #     ymin = l[1]
        #     xmax = l[2]
        #     ymax = l[3]
        #     cls = l[4]
        #     x = (xmin + xmax) / 2 / image_w
        #     y = (ymin + ymax) / 2 / image_h
        #     w = (xmax - xmin) / image_w
        #     h = (ymax - ymin) / image_h


        #     loc = [7 * x, 7 * y]
        #     loc_i = int(loc[1])
        #     loc_j = int(loc[0])
        #     y = loc[1] - loc_i
        #     x = loc[0] - loc_j

        #     if label_matrix[loc_i, loc_j, 24] == 0:
        #         label_matrix[loc_i, loc_j, cls] = 1
        #         label_matrix[loc_i, loc_j, 20:24] = [x, y, w, h]
        #         label_matrix[loc_i, loc_j, 24] = 1  # response

        # return image, label_matrix    