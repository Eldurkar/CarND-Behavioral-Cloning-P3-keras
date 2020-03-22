# Import statements
import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import Sequence
from keras.optimizers import Adam
# End of import 

# Start function definitions
def get_lines_from_file(path):
    """Get lines from csv fileData as Python List  
    :param path: path for driving_log.csv  
    :return lines: csv data list
    """
    error_txt = "{} does not exist!!".format(path)
    assert os.path.exists(path), error_txt
    
    lines = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip the headers
        for line in reader:
            lines.append(line)
    return lines

def get_fname(path):
    """Get path name  
    :param path: image path  
    :return path: like "/data/IMG/*.jpg"
    """
    return '/'.join(path.split('/')[-3:])

def get_image_and_steering(lines, target_image_dir='./../data/', \
    is_both_side=True, is_flip=True):
    """Get lines as Python List  
    :param lines: list of lines in driving_log.csv  
    :return arrays X, y: image and steering measurement 
    """    

    n_side = 3 if is_both_side else 1
    images = []
    measurements = []
        
    for line in lines:

        correction = 0.22         
        steering = float(line[3])
            
        for i in range(n_side):    
            if i == 1:
            # if camera is set on left
                steering += correction
            elif i == 2:
            # if camera is set on right
                steering -= correction
               
        image_path = target_image_dir + get_fname(line[0])
        assert os.path.exists(image_path), "image path is wrong: {}".format(image_path)
        image = cv2.imread(image_path)
        images.append(image)
        measurements.append(steering)
        
        if is_flip:
            images.append(cv2.flip(image, 1))
            measurements.append(steering * -1.0)
        
    return np.array(images), np.array(measurements)

def batch_generator(samples, batch_size=32):
    """Get lines as Python List and generate X, y data in
    batch_size proportions 
    :param lines: list of lines in driving_log.csv  
    :return arrays: image and steering measurement 
    """  
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            X_batch, y_batch = get_image_and_steering(batch_samples)
            yield sklearn.utils.shuffle(X_batch, y_batch)

def get_model():
    """Get Keras Model
    :return model: keras model
    """
    model = Sequential()
    # Regression check
    #model.add(Flatten(input_shape=(160, 320, 3)))
    #model.add(Dense(1))
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))

    model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))

    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')

    return model
    #model.compile(loss='mse', optimizer='adam')
    #model.fit(X_train, y_train, batch_size=128, epochs=1, \
        #validation_split=0.2, shuffle=True)
    #model.save('model.h5')

# End of functions

# Main
path = './../data/driving_log.csv'
csv_lines = get_lines_from_file(path)

# Split traiing set
train_samples, validation_samples = train_test_split(csv_lines, test_size=0.2)
# Set batch size
batch_size1=32
# Create batches for the training and validation sets
train_generator = batch_generator(train_samples, batch_size=batch_size1)
validation_generator = batch_generator(validation_samples, batch_size=batch_size1)
# Fit the keras model
model = get_model()
history_object = model.fit_generator(train_generator, \
    steps_per_epoch=np.ceil(len(train_samples)/batch_size1), \
    validation_data=validation_generator, \
    validation_steps=np.ceil(len(validation_samples)/batch_size1), \
    epochs=5, verbose=1)

model.save('model_1.h5')

print('\n')
print('-------------------')
print('\n')
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

"""
X_train, y_train = get_image_and_steering(csv_lines)
regr_model = get_model(X_train, y_train)
"""
# End