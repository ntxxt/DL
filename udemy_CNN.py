#kernal == filter = feature detector; feature map; reduce size; could be many feature maps
#max pooling; have flexiblity, not affected by image spacial distorsion; reduce size/parametert; prevent overfitting
#flattern, fully connected layer
#softmax & cross-entropy

"""
dog cat dog cat
0.9 0.1  1  0
-(log(0.9)+ long(0.4)
l=−(ylog(p)+(1−y)log(1−p)) for classification
"""

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

#transformations on the images to aviod overfitting -> image agurmentation, 
#code snip from keras, rescale -->(0,1); 

train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2, horizontal_flip = True)

training_set = train_datagen.flow_from_directory('dataset/training_set', #path
                                                 target_size = (64, 64), #image size feed to NN
                                                 batch_size = 32,
                                                 class_mode = 'binary')
                                
test_datagen = ImageDataGenerator(rescale = 1./255) #no transformation but fit

test_set = test_datagen.flow_from_directory('dataset/test_set', 
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


##buliding
cnn = tf.keras.models.Sequential()
# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3])) #color image -> RGB
#input only needed at the begining
# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())
# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu')) #unit -> n. of hidden neurons
# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) #binary classification, unit ->1


##compile & training
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

## Training the CNN on the Training set and evaluating it on the Test set
# only feature scaling, no transform 
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

## predict single image
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0) #add extra dimension relates to the batch; the first dimentiosn
#format expected by the prediction 
result = cnn.predict(test_image/255.0)
training_set.class_indices #check class indices of the model 
if result[0][0] > 0.5:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)