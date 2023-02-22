import tensorflow as tf
import random
import numpy as np
from tensorflow import keras as ks
ls = tf.keras.layers
import tensorflow_addons as tfa
import math
import copy
import einops
from tensorflow.python.ops import gen_array_ops
import logging

from typeguard import typechecked
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

# Sample Distributions
def std_normal(size, dtype=tf.float32):
    """
    Generate the standard Gaussian variable of a certain size
    """

    # return torch.normal(0, 1, size=size).cuda()
    return tf.random.normal(size, 0.0, 1.0,dtype=dtype)

# Activation Functions
@tf.function(jit_compile=True, reduce_retracing=True)
def swish(x):
    return ks.activations.swish(x)

# Normalization Functions

def l2_normalize(x, axis=None, epsilon=1e-12, name=None, dim=None):
    with ops.name_scope(name, "l2_normalize", [x]) as name:
        
        if x.dtype.is_complex:
            square_real = math_ops.square(math_ops.real(x))
            square_imag = math_ops.square(math_ops.imag(x))
            square_sum = math_ops.real(
                math_ops.reduce_sum(square_real + square_imag, axis, keepdims=True))
            x_inv_norm = math_ops.rsqrt(math_ops.maximum(square_sum, epsilon))
            norm_real = math_ops.multiply(math_ops.real(x), x_inv_norm)
            norm_imag = math_ops.multiply(math_ops.imag(x), x_inv_norm)
            return math_ops.complex(norm_real, norm_imag, name=name)
        square_sum = math_ops.reduce_sum(math_ops.square(x), axis, keepdims=True)
        x_inv_norm = math_ops.rsqrt(math_ops.maximum(square_sum, epsilon))
        return math_ops.multiply(x, x_inv_norm, name=name)


# Layer initialization
def Conv1d_with_init(in_channels, out_channels, kernel_size, weight_norm=False, conv_channels_first=True, kernel_initializer=None, dtype=None):
    #TODO: change to Conv1d_with_pytorchHe_init

    kernel_initializer = kernel_initializer if kernel_initializer else tf.keras.initializers.HeNormal()
    # This layer always recieves input:(b, c, seq)
    # conv_channels_first=False is used for compatibility with windows cpu which cannot perform convolution on (... h w c) format

    layer = ls.Conv1D(out_channels, kernel_size, padding='same', data_format='channels_first' if conv_channels_first else 'channels_last', 
        kernel_initializer=kernel_initializer,
        bias_initializer=pytorch_he_uniform_init( in_channels*kernel_size),
        dtype=dtype
        )


    if weight_norm:
        layer = WeightNormalization(layer)

    if conv_channels_first==False:
        
        layer = tf.keras.Sequential(
            [   
                ls.Permute((2,1), dtype=dtype ),
                layer,
                ls.Permute((2,1), dtype=dtype ),
            ]
        )

    return layer

def gain(nonlinearity='leaky_relu', p=None):
    
    if nonlinearity in ['sigmoid','linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']:
        return 1
    
    elif nonlinearity == 'tanh':
        return 5/3
    
    elif nonlinearity ==  'relu':
        return math.sqrt(2.0)

    elif nonlinearity == 'leaky_relu':
        p = p if p else 0.01
        return math.sqrt(2.0/ (1 + p**2 ) )
    
    else:
        raise NotImplementedError
    
    return 1
    
def pytorch_he_uniform_init(in_features, nonlinearity='leaky_relu', p=None):
    limit = gain(nonlinearity, p=p) * tf.math.sqrt( 3/in_features )
    return tf.keras.initializers.RandomUniform(minval=-limit, maxval=limit)

def pytorch_he_normal_init(in_features, nonlinearity='leaky_relu', p=None):
    std = gain(nonlinearity, p=p) * tf.math.sqrt( 1/in_features )
    return tf.keras.initializers.RandomNormal(0, std )


# Layers 
class Conv(ls.Layer):
    def __init__(self, in_channels, filters, kernel_size=3, dilation=1, conv_channels_first:bool=False, stride=1):
        super(Conv, self).__init__()
        self.conv_channels_first = conv_channels_first
        
        self.conv = ls.Conv1D(filters, kernel_size, 
                            padding='same', 
                            dilation_rate=dilation,
                                data_format='channels_first' if conv_channels_first else 'channels_last', 
                                kernel_initializer=pytorch_he_normal_init(in_channels),
                                bias_initializer=pytorch_he_uniform_init(in_channels*kernel_size ),
                                strides = stride,
                                 ) 
                                # channels first Currently not supported by CPU
        
        self.conv = WeightNormalization( self.conv )

        
        if conv_channels_first==False:
            # This layer always recieves input:(b, c, seq)
            self.conv = tf.keras.Sequential(
                [   
                    ls.Permute((2,1)),
                    self.conv,
                    ls.Permute((2,1)),
                ]
            )
                               
    def build(self, input_shape):
        if self.conv_channels_first == False:
            self.conv.layers[0].build(input_shape)
            self.conv.layers[2].build(input_shape)
            self.conv.layers[1].build(tf.TensorShape([input_shape[0],input_shape[2], input_shape[1]]))
        else:
    
            self.conv.build(input_shape)
        
        self.conv.built = True
        
        super().build(input_shape)
        
    def call(self, x):
        # x : (b, c, seq)
        out = self.conv(x)
        return out

class WeightNormalization(tf.keras.layers.Wrapper):

    def __init__(self, layer: tf.keras.layers, **kwargs):
        super().__init__(layer, **kwargs)
        
        self._track_trackable(layer, name="layer")
        
    def build(self, input_shape):
        """Build `Layer`"""
        input_shape = tf.TensorShape(input_shape)
        self.input_spec = tf.keras.layers.InputSpec(shape=[None] + input_shape[1:])

        if not self.layer.built:
            self.layer.build(input_shape)

        kernel_layer = self.layer

        if not hasattr(kernel_layer, "kernel"):
            raise ValueError(
                "`WeightNormalization` must wrap a layer that"
                " contains a `kernel` for weights"
            )

        
        kernel = kernel_layer.kernel

        # The kernel's filter or unit dimension is -1
        self.layer_depth = int(kernel.shape[-1])
        self.kernel_norm_axes = list(range(kernel.shape.rank - 1))

        self.v = kernel
 

        # Set the weight g with the norm of the/ weight vector
        v_flat = tf.reshape(self.v, [-1, self.layer_depth])
        v_norm = tf.linalg.norm(v_flat, axis=0)
        v_norm  = tf.reshape(v_norm, (self.layer_depth,))
        self.g = tf.Variable( tf.cast(v_norm,dtype=kernel.dtype), trainable=True, name='g', dtype=kernel.dtype )
        
        self.built = True        

    def call(self, inputs):
        """Call `Layer`"""

        # Replace kernel by normalized weight variable.        
        kernel = l2_normalize(self.v, axis=self.kernel_norm_axes)*tf.cast(self.g,inputs.dtype)
        
        self.layer.kernel = kernel
            
        outputs = self.layer(inputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(self.layer.compute_output_shape(input_shape).as_list())

    def _initialize_weights(self): #inputs):
        """Initialize weight g.

        The initial value of g could either from the initial value in v,
        
        """

        self.g = self._init_norm()
        # assign_tensors.append(self._initialized.assign(True))
        # return assign_tensors
        return self.g
    
    def _init_norm(self):
        """Set the weight g with the norm of the weight vector."""
        
        v_flat = tf.reshape(self.v, [-1, self.layer_depth])
        v_norm = tf.linalg.norm(v_flat, axis=0)
        v_norm  = tf.reshape(v_norm, (self.layer_depth,))
        
        self.g = self.g.assign( tf.cast(v_norm, dtype=self.g.dtype) )
        
        return self.g

    def get_config(self):
        config = {}
        base_config = super().get_config()
        return {**base_config, **config}

    def remove(self):
        kernel = tf.Variable(
            l2_normalize(self.v, axis=self.kernel_norm_axes) * self.g,
            name= "kernel",
        )

        self.layer.kernel = kernel

        return self.layer

class L2Normalize(tf.keras.layers.Layer):
    
    def call(self, x, axis=None, epsilon=1e-12, name=None, dim=None):

        with ops.name_scope(name, "l2_normalize", [x]) as name:
            
            if x.dtype.is_complex:
                square_real = math_ops.square(math_ops.real(x))
                square_imag = math_ops.square(math_ops.imag(x))
                square_sum = math_ops.real(
                    math_ops.reduce_sum(square_real + square_imag, axis, keepdims=True))
                x_inv_norm = math_ops.rsqrt(math_ops.maximum(square_sum, epsilon))
                norm_real = math_ops.multiply(math_ops.real(x), x_inv_norm)
                norm_imag = math_ops.multiply(math_ops.imag(x), x_inv_norm)
                return math_ops.complex(norm_real, norm_imag, name=name)
            square_sum = math_ops.reduce_sum(math_ops.square(x), axis, keepdims=True)
            x_inv_norm = math_ops.rsqrt(math_ops.maximum(square_sum, epsilon))
            return math_ops.multiply(x, x_inv_norm, name=name)

class ScaleLayer(tf.keras.layers.Layer):
    def __init__(self, shape=(1,) ,dtype=tf.float32):
      super(ScaleLayer, self).__init__( dtype=dtype)      
      self.scale = tf.Variable( tf.zeros( shape, dtype=dtype), dtype=dtype)

    def call(self, inputs):
       out = inputs * tf.math.exp( self.scale * 3 )
       return out