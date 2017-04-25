import csv, cv2, sklearn
import numpy as np
from math import floor
from model import Model
import random
data_paths = ["data", "my_data", "training_1"]

lines = []
for path in data_paths:
    i = 0
    with open(path+'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if(i > 0):
                center_angle = float(line[3])
                keep_me = True
                #if (center_angle > -0.15 and center_angle < 0.15):
                #    keep_me = (random.uniform(0, 1) > 0.80)

                if (keep_me):
                    lines.append(line)
            else:
                i = 1

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines[1:], test_size=0.2)


def load_img(file):

    split = file.split('/')[-3:]
    if(len(split) < 3):
        name = "data/IMG/"+split[-1]
    else:
        name = '/'.join(split)


    im = cv2.imread(name).astype(np.float32)

    #im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR, im)
    # crop interesting part of image
    im = im[60:140, 0:320]
    im = cv2.resize(im, (64, 64), interpolation=cv2.INTER_AREA)
    im = ((im / 127.5) - 1.0)
    return im


def blur_image(im):
    k = random.randint(3,7)
    if k % 2 == 0:
        k = k + 1
    return cv2.GaussianBlur(im, (k,k), 0)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:

                    center_im = load_img(batch_sample[0])
                    left_im = load_img(batch_sample[1])
                    right_im = load_img(batch_sample[2])


                    # Center
                    images.append(center_im)
                    angles.append(center_angle)

                    images.append(blur_image(center_im))
                    angles.append(center_angle)
                    # Center flipped
                    im2 = cv2.flip(center_im,1)
                    images.append(im2)
                    angles.append(0-center_angle)

                    images.append(blur_image(im2))
                    angles.append(0 - center_angle)

                    # Left
                    images.append(left_im)
                    angles.append(center_angle + 0.25)

                    images.append(blur_image(left_im))
                    angles.append(center_angle + 0.25)
                    # Left flipped
                    im3 = cv2.flip(left_im, 1)
                    images.append(im3)
                    angles.append(0 - center_angle - 0.25)

                    images.append(blur_image(im3))
                    angles.append(0 - center_angle - 0.25)
                    # Right
                    images.append(right_im)
                    angles.append(center_angle - 0.25)

                    images.append(blur_image(right_im))
                    angles.append(center_angle - 0.25)

                    # Right flipped
                    im4 = cv2.flip(right_im, 1)
                    images.append(im4)
                    angles.append(0 - center_angle + 0.25)

                    images.append(blur_image(im4))
                    angles.append(0 - center_angle + 0.25)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


model = Model()
model.make_simple_2()

nb_epoch = 50

model.fit_generator(train_generator, floor(len(train_samples)), validation_generator, len(validation_samples),nb_epoch=nb_epoch)

model.save("simple_2_30")



