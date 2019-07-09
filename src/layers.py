import tensorflow as tf
import numpy as np


def _spline_interpolate_kernel_layer(kernel_positions, kernel_values, kernel_size, channels, order=1, cont3d=False):
    """
    kernel_positions: (List[List[int]]) A list of n dimensional coordinates. (typically 3d)
    kernel_values: (List[int]) A list of kernel values
    kernel_size: (int) size of kernel (assumes square)
    order: (int) order to use during spline interpolation
    """
    if not cont3d:
        raise NotImplementedError(
            "Discrete channel interpolation is currently not implemented")

    # required for interpolation to work.
    # float32 because the spline is continous
    kernel_positions = tf.convert_to_tensor(kernel_positions, dtype=tf.float32)


    if kernel_positions.shape[2] == 3 and channels == 1:
        raise ValueError("You should omit the channels dimension from your kernel positions if you only have 1 channel.")

    # all positions in the interpolated kernel
    # TODO: Implement this in tensorflow
    n_filters = kernel_positions.shape[0]
    if not channels == 1:
        query_points_raw = tf.convert_to_tensor([[[x, y, z] for x in range(kernel_size) for y in range(
            kernel_size) for z in range(channels)] for _ in range(n_filters)], dtype=tf.float32)
    else:
        query_points_raw = tf.convert_to_tensor([[[x, y] for x in range(
            kernel_size) for y in range(kernel_size)] for _ in range(n_filters)], dtype=tf.float32)


    query_points = tf.reshape(
        query_points_raw,   [n_filters, -1, query_points_raw.shape[2]])

    if cont3d:
        # TODO: Add caching to make the forward overhead even smaller
        kernel = tf.contrib.image.interpolate_spline(
            kernel_positions, kernel_values, query_points, order=order)
    else:
        # TODO: interpolate kernel per channel
        # we can do this by segmenting the kernel_position layers
        kernel = None
        raise NotImplementedError(
            "Discrete channel interpolation is currently not implemented")

    return tf.reshape(kernel, [n_filters, kernel_size, kernel_size, channels])


class InterpolatedConv2d(tf.keras.layers.Layer):
    def __init__(self,
                 kernel_positions,
                 kernel_size,
                 strides=(1, 1, 1, 1),
                 padding="VALID",
                 order=1,
                 continous_3d=True,
                 **kwargs):
        """
        Input Shape assumes NHWC

        continous_3d: whether to interpolate continously across channels
        (the different filters are still treated as discrete units)
            - useful for modelling continous spaces such as energy landscapes
            - theoretically less useful for color channels in images.
        """
        super(InterpolatedConv2d, self).__init__(**kwargs)
        self.kernel_positions = kernel_positions
        self.kernel_size = kernel_size
        self.cont3d = continous_3d

        # passthrough arguments
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        # Inputshape assumes: NHWC
        self.channels = input_shape[3]
        self.n_filters = self.kernel_positions.shape[0]

        # 1 for channels_dim, as we only associate one value per position
        # as opposed to assuming the known positions will be the same for each filter
        self.kernel_var = self.add_variable("kernel",
                                            shape=[self.n_filters,
                                                   self.kernel_positions.shape[1], 1])

    def call(self, input):
        full_kernel = _spline_interpolate_kernel_layer(self.kernel_positions,
                                                       self.kernel_var,
                                                       kernel_size=self.kernel_size,
                                                       channels=1,#self.channels, #the method doesn't seem to like to interpolate in 3d.
                                                       cont3d=self.cont3d)

        shape = [full_kernel.shape[1], full_kernel.shape[2],
                 full_kernel.shape[3], full_kernel.shape[0]]
        kernel = tf.reshape(full_kernel, shape)

        # tile accross dimensions to make sure input dimension makes sense
        kernel = tf.tile(kernel, [1, 1, self.channels, 1])
 

        # use tf built in conv2d method for speed
        v = tf.nn.conv2d(input,
                         kernel,
                         strides=self.strides,
                         padding=self.padding)

        return v
    
    def extract_interpolated_kernel(self):
        """Returns the interpolated filters
        akin to a normal conv2d kernels
        """

        full_kernel = _spline_interpolate_kernel_layer(self.kernel_positions,
                                                       self.kernel_var,
                                                       kernel_size=self.kernel_size,
                                                       channels=self.channels,
                                                       cont3d=self.cont3d)

        shape = [full_kernel.shape[1], full_kernel.shape[2],
                 full_kernel.shape[3], full_kernel.shape[0]]
        kernel = tf.reshape(full_kernel, shape)
        return kernel


if __name__ == "__main__":
    kernel_positions =  np.array([
                                [0,0],
                                [2,2],
                                [2,0],
                                [0,2]])

    kernel_values = np.array([
        1,
        1,
        1,
        1
    ],dtype="float32")

    kernel_positions = kernel_positions.reshape([1,-1,2])
    kernel_values = kernel_values.reshape([1,4,1])

    print (kernel_positions.shape)
    print (kernel_values.shape)

    print (_spline_interpolate_kernel_layer(kernel_positions,
                                            kernel_values,
                                            kernel_size=3,
                                            channels=1,
                                            cont3d=True))