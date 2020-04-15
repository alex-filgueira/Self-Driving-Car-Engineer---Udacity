import csv
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import math
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


udacity_samples = []
#track1_samples = []
#track2_samples = []


def add_samples(samples, csv_file):
  with open(csv_file) as driving_log:
    reader = csv.reader(driving_log)
    # skip csv header
    next(reader)
    
    for line in reader:
      samples.append(line)
    
    return samples
    
# add samples data from udacity
udacity_samples = add_samples(udacity_samples, '/opt/carnd_p3/data/driving_log.csv')

#Datos propios
#track1_samples = add_samples(track1_samples, 'data_propios/datos_almacenados/driving_log.csv')#Is not included because: "ake sure that your directory /home/workspace/ doesn't include your training images since the Reviews system is limited to 10,000 files and 500MB"
#track2_samples = add_samples(track2_samples, 'data_propios/datos_almacenados_2/driving_log.csv')


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


# extract images from track-1 sample
"""
for line in track1_samples:
  source_path = line[0]
  filename = source_path.split('\\')[-1]
  current_path = 'data_propios/datos_almacenados/IMG/' + filename
  image = mpimg.imread(current_path)
  images.append(image)
  measurement = float(line[3])
  measurements.append(measurement)
 
print("procesed track 1")


# extract images from track-2 sample
for line in track2_samples:
  source_path = line[0]
  filename = source_path.split('\\')[-1]
  current_path = 'data_propios/datos_almacenados_2/IMG/' + filename
  image = mpimg.imread(current_path)
  images.append(image)
  measurement = float(line[3])
  measurements.append(measurement)

print("procesed track 2")
"""


augmented_images, augmented_measurements = [], []
### performed data augmentation by flipping the images to enrich data training
for image, measurement in zip(images, measurements):
  augmented_images.append(image)
  augmented_measurements.append(measurement)
  augmented_images.append(cv2.flip(image, 1))
  augmented_measurements.append(measurement * -1.0)

print("procesed flip")


folders = [
    'data',
]

"""
image_load_data = []
for data_folder in folders:
    print('Gathering csv lines for folder:', data_folder)
    with open('/opt/carnd_p3/{}/driving_log.csv'.format(data_folder)) as f: #/opt/carnd_p3/data/driving_log.csv
        reader = csv.reader(f)
        count = 0
        for csv_line in reader:
            image_load_data.append((csv_line, data_folder))

image_load_data = shuffle(image_load_data)
train_load_data, validation_load_data = train_test_split(image_load_data, test_size=.2, random_state=42)
"""

def data_gen(load_data, batch_size):
    images = []
    steering_angles = []
    while True:
        for offset in range(0, len(load_data), batch_size):
            for line, folder in load_data[offset:offset+batch_size]:
                center_img_path = line[0]
                file_name = center_img_path.split('/')[-1]
                image = cv2.imread('/opt/carnd_p3/{}/IMG/{}'.format(folder, file_name))
                if image is not None:
                    images.append(image)
                    #images.append(np.fliplr(image))

                    steering_angle = float(line[3])
                    steering_angles.append(steering_angle)
                    #steering_angles.append(-steering_angle)

            if len(images) == 0:
                print("Didn't find any good images - continuing")
                continue

            yield shuffle(np.array(images), np.array(steering_angles))
            
            
batch = 512
epochs = 1
activation_type = 'elu'
dropout_p = .2

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


model = Sequential()
# setup lambda layer
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
# crop image area to make training faster
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))

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

#fitted_model = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10); 
"""
fitted_model = model.fit_generator(
    generator=data_gen(train_load_data, batch),
    steps_per_epoch=math.ceil(len(train_load_data)/batch),
    epochs=epochs,
    validation_data=data_gen(validation_load_data, batch),
    validation_steps=math.ceil(len(validation_load_data)/batch)
)
"""

model.save('model.h5')
print("Model saved")

plt.plot(fitted_model.history['loss'])
plt.plot(fitted_model.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

