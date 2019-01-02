# Interpolated Kernels

> NOTE: This project is work in progress.
> Progress is currently limited due to upcoming exam season. 

This project is an attempt to create an efficient convolutional kernel with a large dense window size for very high resolution inputs.

It achieves this by using sparse kernels similar to [dilated kernels by Y Li et. al.](https://arxiv.org/abs/1802.10062), but offers more flexibility for the weight positions.

The remaining positions are then interpolated using an interpolation strategy which adds a relatively small amount of extra computation during the forward propagation phase which is offset by heavily reducing the number of parameters required to optimize during backpropagation. 

TK: explanitory image.

This method is desirable over dilated kernels whenever 
1. You require dense kernels. 
2. Your input distribution is regular and you want to heavily reduce the number of parameters, or higher accuracy.
3. You need more flexibility of the positions of the kernel weights. 

Currently, this implementation is especially suitable for:
- High resolution 1 channel (black and white) images
- 2d energy landscapes
- 3d energy landscapes


## Feature Comparison

| |Our solution|Dilated Kernels|Conventional Kernels
-----|-----|-----|-----
Speed|Fast -> Faster [1]|Faster|Slow
\# Parameters|Fewest|Few|Many
Window size|Large|Large|Small
Effective Density|Dense|Sparse|Dense
A priori knowledge| Flexible | Some | Limited

[1]: Depending on the regularity of the data, our solution can be just as fast if not faster than dilated kernels. 

## Installation
TK

## Usage
This [Tensorflow](tensorflow.org) implementation offers a Keras compatible layer which you can plug into your model to replace `conv2d` layers. 

Simple usage:
```Python
import tensorflow as tf

from interpolated_kernels.layers import InterpolatedConv2d
from interpolated_kernels.utils import symmetric_filters

n_filters = 5
kernel_coordinates = [
    [0,0,1], # [height, width, channel] coordinates
    [1,1,1],
    [2,2,1],
]

# we use the `symmetric_filtes` helper function
# to replicate the same known positions across 
# all filters.
filters = symmetric_filters(kernel_coordinates, n_filters)


kernel_size = 4 # the size of the interpolated kernel

# a standard callable keras layer. 
layer = InterpolatedConv2d(filters, kernel_size)

# the input to the layer 
# a rank 4 tensor with size: 
# [batch, height, width, channels]
img_tensor = tf.ones([5, 11, 11, 3])

# compute the output volume
v = layer(img_tensor)
```


TK: `op` mode.

TK: change interpolation engine.

## Acknowledgements
This project wouldn't have been possible without support from [Hammer research group](http://users-phys.au.dk/hammer/).
