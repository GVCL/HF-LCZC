#i!/usr/bin/env python

#print ('test1')
import numpy as np
import matplotlib.cm as cm

import h5py 
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf

#print ('test2')

from tensorflow import keras
from tensorflow.keras.layers import MaxPool2D, GlobalAveragePooling2D, BatchNormalization, Conv2D
from tensorflow.keras.layers import Layer, Dense, Dropout, Activation, Flatten, Input, Concatenate, Add, UpSampling2D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Model 
from tensorflow.keras.models import load_model
from scipy.ndimage import gaussian_filter

# Clear any previous session to free up memory
tf.keras.backend.clear_session()


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

#kernel_size = 5
stride_size = 1
pool_size = 2

class GaussianSmoothing(Layer):
	def __init__(self, kernel_size=3, **kwargs):
		super(GaussianSmoothing, self).__init__(**kwargs)
		self.kernel_size = kernel_size

	def build(self, input_shape):
        # Create Gaussian kernel
		self.sigma = self.kernel_size / 2.0
		x = tf.range(-self.kernel_size // 2 + 1, self.kernel_size // 2 + 1)
		x = tf.cast(x, dtype=tf.float32)
		gauss = tf.exp(-x**2 / (2.0 * self.sigma**2))
		gauss /= tf.reduce_sum(gauss)
		gauss_kernel = tf.tensordot(gauss, gauss, axes=0)
		gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
		self.gauss_kernel = tf.tile(gauss_kernel, [1, 1, input_shape[-1], 1])
		self.gauss_kernel = tf.Variable(self.gauss_kernel, trainable=False)

	def call(self, inputs):
		smoothed = tf.nn.depthwise_conv2d(inputs, self.gauss_kernel, strides=[1, 1, 1, 1], padding='SAME')
		return smoothed

	def compute_output_shape(self, input_shape):
		return input_shape

	def get_config(self):
		config = super(GaussianSmoothing, self).get_config()
		config.update({'kernel_size': self.kernel_size})
		return config

# Define Gaussian smoothing kernel sizes
#kernel_sizes = [4, 8, 12, 16]
kernel_sizes = [2, 4, 6, 8]
#kernel_sizes = [3, 5, 7]

# Apply Gaussian smoothing at different kernel sizes
def apply_multiscale_smoothing(input_layer, kernel_sizes):
	smoothed_images = [GaussianSmoothing(kernel_size=size)(input_layer) for size in kernel_sizes]
	combined_image = Concatenate(axis=-1)(smoothed_images)
	return combined_image


# Define input shapes
input_shape_s1 = (32, 32, 8)
input_shape_s2 = (32, 32, 10)
n_classes = 17  # Assuming 17 classes for the final output


# Create separate input layers for each modality
input_layer_s1 = Input(shape=input_shape_s1)
input_layer_s2 = Input(shape=input_shape_s2)

# Apply multiscale smoothing and averaging for s1 modality
smooth_s1 = apply_multiscale_smoothing(input_layer_s1, kernel_sizes)
print(smooth_s1)
# Check shape before passing to Conv2D
print(smooth_s1.shape)

# Apply multiscale smoothing and averaging for s2 modality
smooth_s2 = apply_multiscale_smoothing(input_layer_s2, kernel_sizes)
print (smooth_s2)

#DATA level fusion
concatenated_raw_data = Concatenate()([smooth_s1, smooth_s2])

# Continue with the convolutional layers, pooling, and other operations as needed
pix = Conv2D(32, kernel_size=5, strides=stride_size, padding='same', activation='relu')(concatenated_raw_data)
pix = MaxPool2D(pool_size=pool_size, strides=(2, 2))(pix)
pix = BatchNormalization()(pix)

# More convolutional layers and pooling
pix = Conv2D(64, kernel_size=5, strides=stride_size, padding='same', activation='relu')(pix)
pix = MaxPool2D(pool_size=pool_size, strides=(2, 2))(pix)
pix = BatchNormalization()(pix)

pix = Conv2D(128, kernel_size=5, strides=stride_size, padding='same', activation='relu')(pix)
pix = MaxPool2D(pool_size=pool_size, strides=(2, 2))(pix)
pix = BatchNormalization()(pix)

# CNN model for s1 modality
x_s1 = Conv2D(16, kernel_size=5, strides=stride_size, padding='same', activation='relu')(smooth_s1)
x_s1 = MaxPool2D(pool_size=pool_size, strides=(2, 2))(x_s1)
x_s1 = BatchNormalization()(x_s1)

# CNN model for s2 modality
x_s2 = Conv2D(16, kernel_size=5, strides=stride_size, padding='same', activation='relu')(smooth_s2)
x_s2 = MaxPool2D(pool_size=pool_size, strides=(2, 2))(x_s2)
x_s2 = BatchNormalization()(x_s2)

concatenated_features = Concatenate()([x_s1, x_s2])  # Combine features from both modalities


# Continue with the convolutional layers, pooling, and other operations as needed
x = Conv2D(32, kernel_size=5, strides=stride_size, padding='same', activation='relu')(concatenated_features)
x = MaxPool2D(pool_size=pool_size, strides=(2, 2))(x)
x = BatchNormalization()(x)

# More convolutional layers and pooling
x = Conv2D(64, kernel_size=5, strides=stride_size, padding='same', activation='relu')(x)
x = MaxPool2D(pool_size=pool_size, strides=(2, 2))(x)
x = BatchNormalization()(x)

# Feature-level fusion

# Continue with more convolutional layers, pooling, and other operations as needed
x = Conv2D(128, kernel_size=5, strides=stride_size, padding='same', activation='relu')(x)
x = MaxPool2D(pool_size=pool_size, strides=(2, 2))(x)
x = BatchNormalization()(x)


x = Conv2D(18, kernel_size=5, strides=stride_size, padding='same', activation='relu')(x)
x = MaxPool2D(pool_size=pool_size, strides=(2, 2))(x)
x = BatchNormalization()(x)

x = UpSampling2D(size=(4, 4), data_format=None, interpolation='nearest')(x)
# Combine features from both data-level and feature-level fusion
combined_features = Concatenate()([x, pix])

# Add a GlobalAveragePooling2D layer to reduce spatial dimensions
x = GlobalAveragePooling2D()(combined_features)

# Continue with additional dense layers for further processing
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)

# Add the final output layer for classification
output_layer = Dense(n_classes, activation='softmax')(x)

# Create the final model
model = tf.keras.models.Model(inputs=[input_layer_s1, input_layer_s2], outputs=output_layer)

# Compile the model
model.compile(optimizer=Adam(lr=0.0002), loss='categorical_crossentropy', metrics=['accuracy'])
#model.compile(optimizer=RMSprop(lr=0.0002), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using both modalities as input
history = model.fit([X1,X2], Y, epochs=100)
model.save("GF-hyb.h5")

with tf.keras.utils.custom_object_scope({'GaussianSmoothing': GaussianSmoothing}):
	loaded_model = load_model('GF-hyb.h5')

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

		for i, item in enumerate(data):
			X1_image=item[0][:, : , 0]
			X2_image=item[1][:, :, 0]
			item_classes = item[2]
#			print(f"Item Classes: {item_classes}")

# Find the predicted class label with the highest probability
			predicted_class = np.argmax(item_classes) + 1  # Add 1 to get the actual class label (1-based index)
#
			classes_labels = [classes[predicted_class - 1]]  # Subtract 1 to get the correct index for classes list
#-----------
		with h5py.File('train-GF-hyb-k4.h5', 'w') as file:
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




