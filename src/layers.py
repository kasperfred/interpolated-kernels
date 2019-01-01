import tensorflow as tf


def _spline_interpolate_kernel_layer(kernel_positions, kernel_values, kernel_size, channels, order=1, cont3d=False):
    """
    kernel_positions: (List[List[int]]) A list of n dimensional coordinates. (typically 3d)
    kernel_values: (List[int]) A list of kernel values
    kernel_size: (int) size of kernel (assumes square)
    order: (int) order to use during spline interpolation
    """
    if not cont3d:
        raise NotImplementedError(
            "Discrete channels interpolation is currently not implemented")

    # required for interpolation to work.
    # float32 because the spline is continous
    kernel_positions = tf.convert_to_tensor(kernel_positions, dtype=tf.float32)

    batch_size = kernel_positions.shape[0]

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
            "Discrete channels interpolation is currently not implemented")

    return tf.reshape(kernel, [batch_size, kernel_size, kernel_size, channels])
    return kernel


class InterpolatedConv2d(tf.keras.layers.Layer):
    def __init__(self,
                 kernel_positions,
                 kernel_size,
                 strides=(1, 1, 1, 1),
                 padding="SAME",
                 order=1,
                 continous_3d=True):
        """
        Input Shape assumes NHWC

        continous_3d: whether to interpolate continously across channels
        (the different filters are still treated as discrete units)
            - useful for modelling continous spaces such as energy landscapes
            - theoretically less useful for color channels in images.
        """
        super(InterpolatedConv2d, self).__init__()
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
        self.kernel_var = tf.convert_to_tensor(
            np.array([[[5], [10], [20], [5]], [[5], [10], [20], [5]]]), dtype=tf.float32)

        full_kernel = _spline_interpolate_kernel_layer(self.kernel_positions,
                                                       self.kernel_var,
                                                       kernel_size=4,
                                                       channels=3,
                                                       cont3d=self.cont3d)

        shape = [full_kernel.shape[1], full_kernel.shape[2],
                 full_kernel.shape[3], full_kernel.shape[0]]
        kernel = tf.reshape(full_kernel, shape)

        # use tf built in conv2d method for speed
        v = tf.nn.conv2d(input,
                         kernel,
                         strides=self.strides,
                         padding=self.padding)

        return v
