#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:24:13 2019

@author: jonyoung
"""

from tensorflow.keras.layers import Layer
from tensorflow.keras.backend import conv2d, tile, permute_dimensions

class EdgeToEdge(Layer) :
    
    def __init__(self, filters, use_bias, **kwargs):
        
        self.filters = filters
        self.use_bias = use_bias
        super(EdgeToEdge, self).__init__(**kwargs)
        
    def build(self, input_shape):
        
        print ('input shape:') 
        print (input_shape)
        
        # Create a trainable weight variable for this layer.
        # kernel shape is matrix height or width as they are the same
        batch_size, n_rows, n_cols, n_channels = input_shape

        # if height and width are not the same, raise an error
        if n_rows == n_cols :
        
            # kernel shape: filter shape, channels in, n filters
            kernel_shape = (n_rows, 1, n_channels, self.filters)
            self.kernel = self.add_weight(name='kernel', 
                                          shape=kernel_shape,
                                          initializer='uniform',
                                          trainable=True)
            
            if self.use_bias :
                
                # use broadcasting so bias is one value per output channel
                bias_shape = (1, 1, 1, self.filters)
                self.bias = self.add_weight(name='bias',
                                shape=bias_shape,
                                initializer='zeros',
                                trainable=True)
                
            # store matrix size for convenience
            self.matrix_size = n_rows
        
        else : 
            
            raise ValueError('Adjacency matrices are not square. They must have the same number of columns and rows.')
            
        super(EdgeToEdge, self).build(input_shape)  # Be sure to call this at the end
        
    def call(self, X) :
        
        # first, apply convolution across columns
        X_conv_cols = conv2d(X, self.kernel, strides=(1, 1), padding='valid')
        
        # then tile across rows    
        X_conv_cols_tiled =  tile(X_conv_cols, (1, self.matrix_size, 1, 1))

        # permute dimensions of the tiled tensor to create tiled convolutiion
        # across rows
        X_conv_rows_tiled = permute_dimensions(X_conv_cols_tiled, (0,2,1,3))
        
        # calculate sum of two tiled tensors
        # this represent result of edge-to-edge filter
        E2E = X_conv_rows_tiled + X_conv_cols_tiled
        
        # add bias if we are using it
        if self.use_bias :
            
            E2E = E2E + self.bias     
        
        return E2E
    
    def compute_output_shape(self, input_shape):

        return (input_shape[:-1] + (self.filters,))
            
class EdgeToNode(Layer) :
    
    def __init__(self, filters, use_bias, **kwargs):
        
        self.filters = filters
        self.use_bias = use_bias
        super(EdgeToNode, self).__init__(**kwargs)
        
    def build(self, input_shape):
        
        print ('input shape:')
        print (input_shape)
        
        # Create a trainable weight variable for this layer.
        # kernel shape is matrix height or width as they are the same
        batch_size, n_rows, n_cols, n_channels = input_shape

        # if height and width are not the same, raise an error
        if n_rows == n_cols :
        
            # kernel shape, channels in, n filters
            kernel_shape = (n_rows, 1, n_channels, self.filters)
            self.kernel = self.add_weight(name='kernel', 
                                          shape=kernel_shape,
                                          initializer='uniform',
                                          trainable=True)
            
            if self.use_bias :
                
                # use broadcasting so bias is one value per output channel
                bias_shape = (1, 1, 1, self.filters)
                self.bias = self.add_weight(name='bias',
                                shape=bias_shape,
                                initializer='zeros',
                                trainable=True)
            
        else : 
            
            raise ValueError('Adjacency matrices are not square. They must have the same number of columns and rows.')
            
        super(EdgeToNode, self).build(input_shape)  # Be sure to call this at the end
        
    def call(self, X) :
        
         # first, apply convolution across columns
        X_conv_cols = conv2d(X, self.kernel, strides=(1, 1), padding='valid')
        
        # double results for E2N to account for sum of (equal) column and row
        # convolutions
        E2N = X_conv_cols * 2
        
        # add bias if we are using it
        if self.use_bias :
            
            E2N = E2N + self.bias
        
        return E2N
    
    def compute_output_shape(self, input_shape):

        return (input_shape[0] , 1, input_shape[2]) + (self.filters,)
        
class NodeToGraph(Layer) :
    
    def __init__(self, filters, use_bias, **kwargs):
        
        self.filters = filters
        self.use_bias = use_bias
        super(NodeToGraph, self).__init__(**kwargs)
        
    def build(self, input_shape):
        
        print ('input shape:')
        print (input_shape)
        
        # Create a trainable weight variable for this layer.
        # kernel shape is matrix height or width as they are the same
        batch_size, n_rows, n_cols, n_channels = input_shape

        # kernel matches shape of input EdgeToNode
        kernel_shape = (1, n_cols, n_channels, self.filters)
        self.kernel = self.add_weight(name='kernel', 
                                      shape=kernel_shape,
                                      initializer='uniform',
                                      trainable=True)
        
        if self.use_bias :
            
            bias_shape = (1, 1, 1, self.filters)
            self.bias = self.add_weight(name='bias',
                            shape=bias_shape,
                            initializer='ones',
                            trainable=True)
    
        # store matrix size for convenience
        self.matrix_size = n_rows
                     
        super(NodeToGraph, self).build(input_shape)  # Be sure to call this at the end
        
    def call(self, X) :
        
        # apply convolution
        N2G = conv2d(X, self.kernel, strides=(1, 1), padding='valid')
        
        # add bias if we are using it
        if self.use_bias :
            
            N2G = N2G + self.bias
        
        return N2G
    
    def compute_output_shape(self, input_shape):

        return (input_shape[0], 1, 1, self.filters,)
        