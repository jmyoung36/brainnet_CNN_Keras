#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:24:13 2019

@author: jonyoung
"""

import numpy as np
from keras.layers import Layer, Add
from keras.backend import conv2d, tile, permute_dimensions

class EdgeToEdge(Layer) :
    
    def __init__(self, filters, **kwargs):
        
        self.filters = filters
        super(EdgeToEdge, self).__init__(**kwargs)
        
    def build(self, input_shape):
        
        # do we want to add bias 
        self.use_bias = False
        
        # Create a trainable weight variable for this layer.
        # kernel shape is matrix height or width as they are the same
        batch_size, n_rows, n_cols, n_channels = input_shape

        # if height and width are not the same, raise an error
        if n_rows == n_cols :
        
            kernel_shape = (n_channels, n_rows, 1, self.filters)
            self.kernel = self.add_weight(name='kernel', 
                                          shape=kernel_shape,
                                          initializer='uniform',
                                          trainable=True)
            
            if self.use_bias :
                
                #bias_shape = ()
                self.bias = self.add_weight(name='bias',
                                shape=kernel_shape,
                                initializer='zeros',
                                trainable=True)
        
            # store matrix size for convenience
            self.matrix_size = n_rows
        
            super(EdgeToEdge, self).build(input_shape)  # Be sure to call this at the end
            
        else : 
            
            raise ValueError('Adjacency matrices are not square. They must have the same number of columns and rows.')
        
    def call(self, X) :
        
        # first, apply convolution across rows
        X_conv_rows = conv2d(X, self.kernel, strides=(1, 1), padding='valid')
        
                #E2E_uncorrected = Add()([X_conv_rows_tiled, X_conv_cols_tiled])
        # then tile across rows
        X_conv_rows_tiled =  tile(X_conv_rows, (1, 1, self.matrix_size, 1))
        
        # permute dimensions of the tiled tensor to create tiled convolutiion
        # across columns
        X_conv_cols_tiled = permute_dimensions(X_conv_rows_tiled, (0,2,1,3))
        
        # calculate sum of two tiled tensors
        # this represent result of edge-to-egde filter before double-counting
        E2E_uncorrected = X_conv_rows_tiled + X_conv_cols_tiled
        
        # add bias if we are using it
        if self.use_bias :
            
            E2E_uncorrected = E2E_uncorrected + self.bias
        
        print E2E_uncorrected
        
        return E2E_uncorrected
    
    def compute_output_shape(self, input_shape):

        input_shape[:-1] + (self.filters,)
        
        
    
class EdgeToNode(Layer) :
    
    def __init__(self, filters, **kwargs):
        
        self.filters = filters
        super(EdgeToNode, self).__init__(**kwargs)
        
    def build(self, input_shape):
        
        # do we want to add bias 
        self.use_bias = False
        
        # Create a trainable weight variable for this layer.
        # kernel shape is matrix height or width as they are the same
        batch_size, n_rows, n_cols, n_channels = input_shape

        # if height and width are not the same, raise an error
        if n_rows == n_cols :
        
            kernel_shape = (n_channels, n_rows, 1, self.filters)
            self.kernel = self.add_weight(name='kernel', 
                                          shape=kernel_shape,
                                          initializer='uniform',
                                          trainable=True)
            if self.use_bias :
                
                self.bias = self.add_weight(name='bias',
                                shape=kernel_shape,
                                initializer='zeros',
                                trainable=True)
        
            # store matrix size for convenience
            self.matrix_size = n_rows
        
            super(EdgeToNode, self).build(input_shape)  # Be sure to call this at the end
            
        else : 
            
            raise ValueError('Adjacency matrices are not square. They must have the same number of columns and rows.')
        
    def call(self, X) :
        
        # first, apply convolution across rows
        X_conv_rows = conv2d(X, self.kernel, strides=(1, 1), padding='valid')
        
        # add bias if we are using it
        if self.use_bias :
            
            X_conv_rows = X_conv_rows + self.bias
        
        # square results for E2N
        E2N_uncorrected = X_conv_rows * 2
        
        # add bias if we are using it
        if self.use_bias :
            
            E2N_uncorrected = E2N_uncorrected + self.bias
        
        return E2N_uncorrected
    
    def compute_output_shape(self, input_shape):

        (input_shape[:-2] , 1) + (self.filters,)
        #input_shape[:-1] + (self.filters,)