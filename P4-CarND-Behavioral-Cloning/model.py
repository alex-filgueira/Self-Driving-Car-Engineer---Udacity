import csv
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import math
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def add_samples(samples, csv_file):
  with open(csv_file) as driving_log:
    reader = csv.reader(driving_log)
    # skip csv header
    next(reader)
    for line in reader:
      samples.append(line)
    return samples

udacity_samples = []    
# add samples data from udacity
udacity_samples = add_samples(udacity_samples, '/opt/carnd_p3/data/driving_log.csv')

### extract images used for training data

images = []
measurements = []

# extract images from udacity's sample
for line in udacity_samples:
  source_path = line[0]
  filename = source_path.split('/')[-1]
  current_path = '/opt/carnd_p3/data/IMG/' + filename
  image = mpimg.imread(current_path)
  images.append(image)
  measurement = float(line[3])
  measurements.append(measurement)

print("procesed udacity_samples")


augmented_images, augmented_measurements = [], []
### performed data augmentation by flipping the images to enrich data training
for image, measurement in zip(images, measurements):
  augmented_images.append(image)
  augmented_measurements.append(measurement)
  augmented_images.append(cv2.flip(image, 1))
  augmented_measurements.append(measurement * -1.0)

print("procesed flip")


# setup variable for training images and its measurement  
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

print("Total samples for data training: ", len(augmented_images))


from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten 
from keras.layers import Activation
from keras.layers import Lambda
from keras.layers import Dense
from keras.layers import Cropping2D
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator 

from sklearn.model_selection import train_test_split



model = Sequential()
# setup lambda layer
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
# crop image area to make training faster
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))

dropout_p = .2

# setup using NVIDIA architecture
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Dropout(dropout_p))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Dropout(dropout_p))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Dropout(dropout_p))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Dropout(dropout_p))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Dropout(dropout_p))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
fitted_model = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

# construct the training image generator for data augmentation
"""
datagen = ImageDataGenerator(horizontal_flip=True)

X_trainPre = np.array(images)
y_trainPre = np.array(measurements)

X_train, X_test, y_train, y_test = train_test_split(X_trainPre, y_trainPre, test_size=0.2, random_state=7)
print("X_train: ",len(X_train))
print("y_train: ",len(y_train))
print("X_test: ",len(X_test))
print("y_test: ",len(y_test))

datagen.fit(X_train)
datagen.fit(X_test)


EPOCHS = 5
BS = 256
train_generator = datagen.flow(X_train, y_train, batch_size=BS)
valid_generator = datagen.flow(X_test, y_test, batch_size=BS)


# train the network
fitted_model =model.fit_generator(
        train_generator,
        samples_per_epoch = len(X_train), 
        epochs=42,
        validation_data = valid_generator,
        validation_steps = 50)

"""

model.save('modelX.h5')
print("Model saved")

plt.plot(fitted_model.history['loss'])
plt.plot(fitted_model.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

