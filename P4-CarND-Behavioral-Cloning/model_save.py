import csv
import cv2
import numpy as np
import matplotlib.image as mpimg

udacity_samples = []
track1_samples = []
track2_samples = []


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
track1_samples = add_samples(track1_samples, 'data_propios/datos_almacenados/driving_log.csv')#Is not included because: "ake sure that your directory /home/workspace/ doesn't include your training images since the Reviews system is limited to 10,000 files and 500MB"
track2_samples = add_samples(track2_samples, 'data_propios/datos_almacenados_2/driving_log.csv')


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

### performed data augmentation by flipping the images to enrich data training
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
  augmented_images.append(image)
  augmented_measurements.append(measurement)
  augmented_images.append(cv2.flip(image, 1))
  augmented_measurements.append(measurement * -1.0)
print("procesed zip")

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
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
#model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)

model.save('model_uda-d-d2.h5')