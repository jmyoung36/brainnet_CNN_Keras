#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:55:18 2019

@author: jonyoung
"""

import numpy as np
from keras import layers
from keras.layers import Input, Conv2D, Lambda, Add, Dense, Activation
from keras.models import Model, Sequential
from keras.backend import tile, permute_dimensions, variable
from keras.activations import relu
from brainNetCNN_keras import EdgeToEdge, EdgeToNode, NodeToGraph


# define the test model
def model(input_shape, size):
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)
    
    # apply convolution across rows
    X_conv_rows = Conv2D(filters = 1, kernel_size=(size, 1), strides=1, padding='valid', input_shape = input_shape, name='conv')(X_input)
    
    # get weights as they will be needed for correcting double-counting
    #weights = conv.get_weights()
    
    # tile it 
    # wrap this in lambda layer so output has type Keras layer
    tile_convolution = Lambda(lambda x: tile(x, (1, size, 1, 1)))
    X_conv_rows_tiled = tile_convolution(X_conv_rows)
    
    # tranpose for vertical convolution
    # wrap this in lambda layer so output has type Keras layer
    transpose_convolution = Lambda(lambda x: permute_dimensions(x, (0,2,1,3)))
    #X_conv_cols = transpose_convolution(X_conv_rows)
    X_conv_cols_tiled = transpose_convolution(X_conv_rows_tiled)
    
    # calculate sum of two tiled tensors
    # this represent result of edge-to-egde filter before double-counting
    E2E_uncorrected = Add()([X_conv_rows_tiled, X_conv_cols_tiled])
    
    # calculate doubled X_conv_rows
    # represents edge2node
    E2N_uncorrected = Lambda(lambda x: x * 2)(X_conv_rows)
    model = Model(inputs = X_input, outputs= [E2N_uncorrected, E2E_uncorrected])
    
    return model

#def custom_layer_model_A(input_shape, n_filters, use_bias):
#    
#    # test my Keras layer
#    X_input = Input(shape=input_shape, name='input_layer')
#    E2E = EdgeToEdge_A(n_filters, use_bias)(X_input)
#    #E2N = EdgeToNode(n_filters)(X_input)
#    
#    #model = Model(inputs = X_input, outputs= [E2E, E2N])
#    model = Model(inputs = X_input, outputs= E2E)
#    
#    return model
#
#def custom_layer_model_B(input_shape, n_filters, use_bias):
#    
#    # test my Keras layer
#    X_input = Input(shape=input_shape, name='input_layer')
#    E2E = EdgeToEdge(n_filters, use_bias)(X_input)
#    #E2N = EdgeToNode(n_filters)(X_input)
#    
#    #model = Model(inputs = X_input, outputs= [E2E, E2N])
#    model = Model(inputs = X_input, outputs= E2E)
#    
#    return model
#
#def custom_layer_model_stacked_A(input_shape, n_filters, use_bias):
#    
#    # test my Keras layer
#    X_input = Input(shape=input_shape, name='input_layer')
#    E2E1 = EdgeToEdge_A(n_filters[0], use_bias[0])(X_input)
#    E2E2 = EdgeToEdge_A(n_filters[1], use_bias[1])(E2E1)
#    model = Model(inputs = X_input, outputs= E2E2)
#    
#    return model

def custom_layer_model_stacked_B(input_shape, n_filters, use_bias):
    
    # test my Keras layer
    X_input = Input(shape=input_shape, name='input_layer')
    E2E1 = EdgeToEdge(n_filters[0], use_bias[0])(X_input)
    E2E2 = relu(E2E1)
    E2E2 = EdgeToEdge(n_filters[1], use_bias[1])(E2E1)
    E2N = EdgeToNode(n_filters[2], use_bias[2])(E2E2)
    N2G = NodeToGraph(n_filters[3], use_bias[3])(E2N)
    model = Model(inputs = X_input, outputs= N2G)
    #model = Model(inputs = X_input, outputs= E2E2)
    
    return model

def brainnetCNN_model(input_shape, n_filters, use_bias):
    
    # define network architecture
    # 2 E2E
    # followed by 1 E2N
    # followed by 1 N2G
    # followed by dense
    # then sigmoid for classification
    X_input = Input(shape=input_shape, name='input_layer')
    E2E_1 = EdgeToEdge(n_filters[0], use_bias[0])(X_input)
    #E2E1 = relu(E2E1)
    E2E_1 = Activation('relu')(E2E_1)
    E2E2 = EdgeToEdge(n_filters[1], use_bias[1])(E2E_1)
    
    #E2E2 = relu(E2E2)
    #E2N = EdgeToNode(n_filters[2], use_bias[2])(E2E2)
#    E2N = relu(E2N)
#    N2G = NodeToGraph(n_filters[3], use_bias[3])(E2N)
#    N2G = relu(N2G)
#    dense_1 = Dense(n_filters[4], activation = 'relu', use_bias = use_bias[4])(N2G)
#    dense_final = Dense(1, activation='sigmoid', use_bias = use_bias[5])(dense_1)
#    model = Model(inputs = X_input, outputs= dense_final)
    model = Model(inputs = X_input, outputs= E2E2)
    return model

#def custom_layer_sequential

# create a 'connectivity matrix' M and set of weights w
M = np.array([[1, 0.8, 0.5, 0.6], [0.8, 1, 0.4, 0.7], [0.5, 0.4, 1, 0.15], [0.6, 0.7, 0.15, 1]])
w_vec = np.array([2, 3, 6, 4])


# real data
#M_data = np.genfromtxt('/home/jonyoung/IoP_data/Data/PSYSCAN/Legacy_data/Dublin/connectivity_data.csv', delimiter=',')
#M_data = M_data[:, 1:]
#n_regions = int(np.sqrt(np.shape(M_data)[1]))yLayerModel.set_weights(([w_tensor, w_tensor]))
#M_vec = M_data[0, :]
#M = np.reshape(M_vec, (n_regions, n_regions))
# convolve the connectivity matrix with a cross shaped filter, made of w as 
# a column and a row vector intersecting at the convolution
M_conv_w = np.zeros_like(M)
h, w = np.shape(M)
w_vec = np.squeeze(np.random.rand(1, h))

M2 = np.zeros((2, h, h))
M3 = np.zeros((3, h, h))
M2[0, :, :] = M
M2[1, :, :] = np.array([[0.9, 0.15, 0.3, 0.65], [0.15, 0.75, 0.55, 0.6], [0.3, 0.55, 0.4, 0.55], [0.65, 0.6, 0.55, 0.8]])
M3[0:2, :, :] = M2
M3[2, :, :] = np.array([[0.4, 0.9, 0.55, 0.6], [0.9, 0.85, 0.2, 0.75], [0.55, 0.2, 0.7, 0.35], [0.6, 0.75, 0.35, 0.25]])

w_vec = np.array([2, 3, 6, 4])

w_vec_2_A = np.zeros((h, 2))
w_vec_2_B = np.zeros((2, h))
w_vec_2_A[:, 0] = w_vec
w_vec_2_A[:, 1] = np.array([4, 5, 3, 3])
w_vec_2_B[0, :] = w_vec
w_vec_2_B[1, :] = np.array([4, 5, 3, 3])
w_tensor = w_vec[np.newaxis, :, np.newaxis, np.newaxis]
w_tensor_2_A = w_vec_2_A[np.newaxis, :, np.newaxis]
w_tensor_2_B = w_vec_2_A[:, np.newaxis, np.newaxis, :]


M = M3[1, :, :]

for i in range(h) :
    
    for j in range(w) :
        
        M_conv_w[i, j] = np.sum(M[i, :] * w_vec) + np.sum(M[:, j] * w_vec)
        
# single convolution method: convolve horizontally then tile
M_conv_w_h = np.zeros((1, w))
for i in range(w) :
    
    M_conv_w_h[0, i] = np.sum(M[:, i] * w_vec)
    
M_conv_w_h_tiled = np.tile(M_conv_w_h, (h, 1))
        
# single convolution method: convolve vertically then tile
M_conv_w_w = np.zeros((w, 1))
for i in range(w) :
    
    M_conv_w_w[i, 0] = np.sum(M[i, :] * w_vec)
    
M_conv_w_w_tiled = np.tile(M_conv_w_w, (1, w))

# finally add the two tiled matrices together
M_conv_w_2 = M_conv_w_h_tiled + M_conv_w_w_tiled

# Weights at the crossing are counted twice. Instead, take the average
# convolve the connectivity matrix with a cross shaped filter, made of w as 
# a column and a row vector intersecting at the convolution
M_conv_w_corrected = np.zeros_like(M)
for i in range(h) :
    
    for j in range(w) :
        
        h_conv = M[i, :] * w_vec
        h_conv[j] =  h_conv[j]/2
        v_conv = M[:, j] * w_vec
        v_conv[i] =  v_conv[i]/2
        M_conv_w_corrected[i, j] = np.sum(h_conv) + np.sum(v_conv)
        
# calculate correction to make M_conw_w like M_conv_w_corrected with matrix
# at each element M_conv_i_j, corrected matrix is uncorrected matrix element 
# - (w_i + w_j) * M_i_j * 1/2
# tile the weights in each direction and add
w_grid_sum = np.tile(w_vec, (h, 1)) + np.transpose(np.tile(w_vec, (h, 1)))
correction = (w_grid_sum * M) / 2
M_conv_w_corrected_2 = M_conv_w - correction#

# inititalise 
tm = model((h, h, 1), h)
M_tensor = M[np.newaxis, :, :, np.newaxis]
M2_tensor = M2[:, :, :, np.newaxis]
M3_tensor = M3[:, :, :, np.newaxis]

w_tensor = w_vec[:, np.newaxis, np.newaxis, np.newaxis]
tm.set_weights(([w_tensor]))
X = tm.predict(M_tensor)
E2N = np.squeeze(X[0])
E2E = np.squeeze(X[1])
weights = tm.get_weights()

#myLayerModel_A = custom_layer_model_A((h, h, 1), 8, False)
#myLayerModel_A.summary()
#w_tensor_A = w_vec[np.newaxis, :, np.newaxis, np.newaxis]
##myLayerModel_A.set_weights(([w_tensor_2_A]))
#X_custom_A = myLayerModel_A.predict(M3_tensor)
#
#myLayerModel_B = custom_layer_model_B((h, h, 1), 8, False)
#myLayerModel_B.summary()
#w_tensor_B = w_vec[:, np.newaxis, np.newaxis, np.newaxis]
#myLayerModel_B.set_weights(([w_tensor_2_B]))
#X_custom_B = myLayerModel_B.predict(M3_tensor)

#myLayerModel_A = custom_layer_model_A((h, h, 1), 2, True)
#myLayerModel_A.set_weights(([w_tensor_2_A]))
#X_custom_A_biased = myLayerModel_A.predict(M3_tensor)
#
#myLayerModel_B = custom_layer_model_B((h, h, 1), 2, True)
#myLayerModel_B.set_weights(([w_tensor_2_B]))
#X_custom_B_biased = myLayerModel_B.predict(M3_tensor)

#myStackedLayerModel_A = custom_layer_model_stacked_A((h, h, 1), [4, 16], [False, False])
#myStackedLayerModel_A.summary()
#w_tensor_A = w_vec[np.newaxis, :, np.newaxis, np.newaxis]
##myLayerModel_A.set_weights(([w_tensor_2_A]))
#X_custom_stacked_A = myStackedLayerModel_A.predict(M3_tensor)

myStackedLayerModel_B = custom_layer_model_stacked_B((h, h, 1), [4, 32, 64, 128], [True, True, False, False])
myStackedLayerModel_B.summary()
w_tensor_B = w_vec[np.newaxis, :, np.newaxis, np.newaxis]
#myLayerModel_A.set_weights(([w_tensor_2_A]))
X_custom_stacked_B = myStackedLayerModel_B.predict(M3_tensor)
my_model = brainnetCNN_model((h, h, 1), [4, 32, 64, 128, 128], [True, True, False, False, False, False])
my_model.summary()
    

