#i!/usr/bin/env python

#print ('test1')
import numpy as np
import matplotlib.cm as cm

import h5py 
import pickle
import matplotlib.pyplot as plt

#print ('test2')
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.layers import MaxPool2D, GlobalAveragePooling2D, BatchNormalization, Conv2D,UpSampling2D, Reshape 
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, Concatenate, Add, Softmax, Multiply
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Multiply, Concatenate, Reshape, Add, LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

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

n = s1.shape[0]
n_classes = 17


# Define input shapes
input_shape_s1 = (32, 32, 8)   
input_shape_s2 = (32, 32, 10) 

# Self-Attention Module
def self_attention_module(inputs, num_heads=8, key_dim=32):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(inputs, inputs)
    attn_output = LayerNormalization()(attn_output)
    return Add()([inputs, attn_output])


# Cross-Attention Module 
def cross_attention_module(query, key, value, name):
	attn_weights = tf.keras.layers.Attention(name=f"cross_attn_{name}")([query, key])
	attn_output = tf.keras.layers.Add(name=f"cross_attn_out_{name}")([attn_weights, value])
	return attn_output


# Define input layers
input_layer_s1 = Input(shape=input_shape_s1, name="input_s1")  
input_layer_s2 = Input(shape=input_shape_s2, name="input_s2") 

# Pixel-Level Fusion 
concatenated_raw_data = Concatenate(name="concat_raw_data")([input_layer_s1, input_layer_s2])

pix = Conv2D(32, (3, 3), padding='same', activation='relu', name="conv_pix")(concatenated_raw_data)
pix = MaxPooling2D(pool_size=(2, 2), name="pool_pix")(pix)
pix = BatchNormalization(name="bn_pix")(pix)
pix_flattened = Flatten(name="flatten_pix")(pix)

# Feature Extraction 
x_s1 = Conv2D(32, (3, 3), activation='relu', padding='same', name="conv_s1")(input_layer_s1)
x_s1 = BatchNormalization(name="bn_s1")(x_s1)
x_s1 = MaxPooling2D(pool_size=(2, 2), name="pool_s1")(x_s1)

x_s2 = Conv2D(32, (3, 3), activation='relu', padding='same', name="conv_s2")(input_layer_s2)
x_s2 = BatchNormalization(name="bn_s2")(x_s2)
x_s2 = MaxPooling2D(pool_size=(2, 2), name="pool_s2")(x_s2)

#  Reshape for Self-Attention
x_s1_reshaped = Reshape((x_s1.shape[1] * x_s1.shape[2], x_s1.shape[3]), name="reshape_s1")(x_s1)
x_s2_reshaped = Reshape((x_s2.shape[1] * x_s2.shape[2], x_s2.shape[3]), name="reshape_s2")(x_s2)

# Apply Self-Attention
attn_s1 = self_attention_module(x_s1_reshaped)
attn_s2 = self_attention_module(x_s2_reshaped)

# Extracted features from both inputs (Query, Key, Value for attention)
query_s1 = Dense(32, name="query_s1")(attn_s1)
key_s1 = Dense(32, name="key_s1")(attn_s1)
value_s1 = Dense(32, name="value_s1")(attn_s1)

query_s2 = Dense(32, name="query_s2")(attn_s2)
key_s2 = Dense(32, name="key_s2")(attn_s2)
value_s2 = Dense(32, name="value_s2")(attn_s2)

# Apply Cross-Attention
cross_s1 = cross_attention_module(query_s1, key_s2, value_s2, "s1")
cross_s2 = cross_attention_module(query_s2, key_s1, value_s1, "s2")

# Reshape back to (H, W, C) before feature fusion
cross_s1_reshaped = Reshape((x_s1.shape[1], x_s1.shape[2], x_s1.shape[3]), name="reshape_back_cross_s1")(cross_s1)
cross_s2_reshaped = Reshape((x_s2.shape[1], x_s2.shape[2], x_s2.shape[3]), name="reshape_back_cross_s2")(cross_s2)

# Feature level Fusion 
fusion_features = Concatenate(name="multiply_features")([cross_s1_reshaped, cross_s2_reshaped])
fusion_features = BatchNormalization(name="bn_fusion")(fusion_features)
flat_features = Flatten(name="flatten_fusion")(fusion_features)

# Hybrid ; pixel-level and feature-level fusion
combined_features = Concatenate(name="concat_features")([pix_flattened, flat_features])

dense1 = Dense(128, activation='relu', name="dense_1")(combined_features)
dense1 = Dropout(0.4, name="dropout_1")(dense1)
output = Dense(n_classes, activation='softmax', name="output")(dense1)
model = Model(inputs=[input_layer_s1, input_layer_s2], outputs=output, name="MS_SAR_Fusion_Model")
model.summary()

#  Compile Model
model.compile(optimizer=Adam(learning_rate=0.0002), loss='categorical_crossentropy', metrics=['accuracy'])

print(Y.shape)
#  Train Model
history = model.fit([X1, X2], Y, epochs=100)

#  Save Model
model.save("Hybatt-100.h5")

model = tf.keras.models.load_model("Hybatt-100.h5")
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
#-----------
	with h5py.File('train-hybatt-100.h5', 'w') as file:
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




