
from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, ZeroPadding2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

import csv, cv2, sklearn
import numpy as np
from math import floor
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from keras.models import load_model

IMG_PATH = "./data/IMG/"


def load_center_img(line):
    name = IMG_PATH + line[0].split('/')[-1]
    im = cv2.imread(name)# .astype(np.float32)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR, im)
    #im = ((im - [128.0, 128.0, 128.0]) / 128.0)
    return im

def checkmodel(lines, model_path):
    model = load_model(model_path)

    for i in range(1000,1050):
        line = lines[i]

        im = load_center_img(line)
        #im = ((im - [128.0, 128.0, 128.0]) / 128.0)
        center_angle = float(line[3])
        pred = model.predict(np.asarray(im)[None, :, :, :])
        print(str(i) + " : "+str(center_angle)+ " - pred : "+str(pred))



def load_lines(path):

    lines = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            center_angle = float(line[3])
            keep_me = True
            if (center_angle == 0):
                keep_me = (random.uniform(0, 1) > 0.5)
                print("Keep_me : " +str(keep_me))

            if (center_angle != 0 or keep_me):
                lines.append(line)

    return lines



def crop_img(img):
    img = img[60:140, 0:320]
    return img


data_paths = ["data", "my_data", "training_1"]

lines = []
for path in data_paths:
    i=0
    with open(path+'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if(i >0):
                center_angle = float(line[3])
                keep_me = True
                if (center_angle > -0.15 and center_angle < 0.15):
                    keep_me = (random.uniform(0, 1) > 0.9)
                    print("Keep_me : " + str(keep_me))

                if (keep_me):
                    lines.append(line)
            else:
                i = 1


angles = np.array([line[3] for line in lines]).astype(np.float32)

plt.hist(angles)
plt.show()