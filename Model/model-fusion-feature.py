#i!/usr/bin/env python

#print ('test1')
import numpy as np
import matplotlib.cm as cm

import h5py 
import pickle
import matplotlib.pyplot as plt

#print ('test2')

from tensorflow import keras
from tensorflow.keras.layers import MaxPool2D, GlobalAveragePooling2D, BatchNormalization, Conv2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, Concatenate, Add
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Model 

fileinp = 'training.h5'
fild = h5py.File(fileinp, 'r')
s1 = np.array(fild['sen1'])
print (s1.shape)

s2 = np.array(fild['sen2'])
print (s2.shape)

lab = np.array(fild['label'])
print (lab.shape)


n = 352366
X1 = s1.reshape(n,32,32,8)
X2 = s2.reshape(n,32,32,10)
Y = lab.reshape(n,17)


#n = s1.shape[0]
n_classes = 17


# Define the shape of individual image modalities
input_shape_s1 = (32, 32, 8)
input_shape_s2 = (32, 32, 10)

kernel_size = 5
stride_size = 1
pool_size = 2

# Create separate input layers for each modality
input_layer_s1 = Input(shape=input_shape_s1)
input_layer_s2 = Input(shape=input_shape_s2)


# CNN model for sentinel-1 images 
x_s1 = Conv2D(16, kernel_size, strides=stride_size, padding='same', activation='relu')(input_layer_s1)
x_s1 = MaxPool2D(pool_size=pool_size, strides=(2, 2))(x_s1)
x_s1 = BatchNormalization()(x_s1)

# CNN model for sentinel-2 images 
x_s2 = Conv2D(16, kernel_size, strides=stride_size, padding='same', activation='relu')(input_layer_s2)
x_s2 = MaxPool2D(pool_size=pool_size, strides=(2, 2))(x_s2)
x_s2 = BatchNormalization()(x_s2)

# Feature-level fusion; Concatenate features from both modalities
concatenated_features = Concatenate()([x_s1, x_s2])  # Combine features from both modalities

# Convolutional layers, pooling, ReLU activation and normalization
x = Conv2D(32, kernel_size, strides=stride_size, padding='same', activation='relu')(concatenated_features)
x = MaxPool2D(pool_size=pool_size, strides=(2, 2))(x)
x = BatchNormalization()(x)

# More convolutional layers and pooling
x = Conv2D(64, kernel_size, strides=stride_size, padding='same', activation='relu')(x)
x = MaxPool2D(pool_size=pool_size, strides=(2, 2))(x)
x = BatchNormalization()(x)


# Continue with more convolutional layers, pooling, and other operations 
x = Conv2D(128, kernel_size, strides=stride_size, padding='same', activation='relu')(x)
x = MaxPool2D(pool_size=pool_size, strides=(2, 2))(x)
x = BatchNormalization()(x)

# GlobalAveragePooling2D layer to reduce spatial dimensions
x = GlobalAveragePooling2D()(x)

# Dense layers for further processing
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)

# Softmax layer for classification
output_layer = Dense(n_classes, activation='softmax')(x)

# Final model
model = Model(inputs=[input_layer_s1, input_layer_s2], outputs=output_layer)

# Compile the model
model.compile(optimizer=Adam(lr=0.0002), loss='categorical_crossentropy', metrics=['accuracy'])
#model.compile(optimizer=RMSprop(lr=0.0002), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using both modalities as input
history = model.fit([X1,X2], Y, epochs=100)
model.save("Feat-100.h5")


model = tf.keras.models.load_model("Feat-100.h5")
try:
	data = []
	X1 = np.array(X1)  # Convert X1 to a NumPy array
	X2 = np.array(X2)  # Convert X2 to a NumPy array
	y_pred=(model.predict([X1, X2]))
	for i in range(n):
		y_pred_argmax = np.argmax(y_pred[i])  # Find the index of the maximum value
		y_pred_binary = np.zeros(y_pred.shape[1], dtype=np.int32)  # Initialize with zeros
		y_pred_binary[y_pred_argmax] = 1  # Set the corresponding element to 1
		item = (X1[i], X2[i], y_pred_binary)
	data.append(item)

#                       print (item)
	classes = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9', 'Class 10', 'Class A', 'Class B', 'Class C', 'Class D', 'Class E', 'Class F', 'Class G']
#               for i, item in enumerate(data[:320 // 32]):
	for i, item in enumerate(data):
		X1_image=item[0][:, : , 0]
		X2_image=item[1][:, :, 0]
		item_classes = item[2]
		print(f"Item Classes: {item_classes}")

# Find the predicted class label with the highest probability
		predicted_class = np.argmax(item_classes) + 1  # Add 1 to get the actual class label (1-based index)
#
		classes_labels = [classes[predicted_class - 1]]  # Subtract 1 to get the correct index for classes list
	with h5py.File('train-hyb-100.h5', 'w') as file:
# Create datasets for X1, X2, and y_pred_binary
		X1_dataset = file.create_dataset('sen1', shape=(len(data),) + X1[0].shape, dtype=np.float32)
		X2_dataset = file.create_dataset('sen2', shape=(len(data),) + X2[0].shape, dtype=np.float32)
		y_dataset = file.create_dataset('label', shape=(352366, 17), dtype=np.int32)

# Write data to the datasets
		for i, item in enumerate(data):
			X1_dataset[i] = item[0]
			X2_dataset[i] = item[1]
			y_dataset[i] = item[2]

except IOError:
	print("Error: Failed to write file")



