import tensorflow as tf
import matplotlib.pyplot as plt

from src.layers import _spline_interpolate_kernel_layer
from src.utils import symmetric_filters
from mnist_example import build_baseline, build_interpolated

# constants
from mnist_example import input_shape, kernel_size, kernel_positions


def get_kernel(model: tf.keras.Model, type: str, normalize=True)->tf.Tensor:
    weights = tf.constant(model.layers[0].get_weights()[0])

    if type == "baseline":
        kernel = weights
    elif type == "interpolated":
        kcs = symmetric_filters(kernel_positions, 32)
        kernel = _spline_interpolate_kernel_layer(kcs,
                                                  weights,
                                                  kernel_size,
                                                  1,
                                                  cont3d=True)
        shape = [kernel.shape[1], kernel.shape[2],
                 kernel.shape[3], kernel.shape[0]]
        kernel = tf.reshape(kernel, shape)
    else:
        raise ValueError("Method must be either `baseline` or `interpolated`.")

    if normalize:
        return tf.reshape(kernel, [kernel.shape[3], kernel.shape[0], kernel.shape[1]])
    else:
        return kernel

def sum_filters(kernel):
    return tf.reduce_sum(kernel, axis=0)

def average_square_difference_kernels(kernel1, kernel2):
    return tf.square(sum_filters(kernel1) - sum_filters(kernel2))

if __name__ == "__main__":
    # rebuild models
    baseline = build_baseline(input_shape, kernel_size)
    interpolated = build_interpolated(
        input_shape, kernel_size, kernel_positions)

    # load weights
    baseline.load_weights("Baseline.h5")
    interpolated.load_weights("Interpolated.h5")

    # extract kernels
    baseline_kernel = get_kernel(baseline, "baseline")
    interpolated_kernel = get_kernel(interpolated, "interpolated")

    with tf.Session() as sess:
        res = sess.run(average_square_difference_kernels(baseline_kernel, interpolated_kernel))

    plt.imshow(res)
    plt.show()