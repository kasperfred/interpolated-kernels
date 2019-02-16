from src.utils import symmetric_filters
from src.layers import InterpolatedConv2d
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
from timer import Timer
import os
import pickle
import cv2

# debug
verbose = True
run_version = 0

# disable GPU due to cuda handler not being able to register
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# params
batch_size = 128
epochs = 3
im_size = 28*4

# kernel params
kernel_size = 3*4  # effective
kernel_positions = np.array([
    # h w
    [0, 0],
    [0, 11],
    [0, 8],
    [1, 5],
    [3, 2],
    [3, 9],
    [5, 0],
    [5, 4],
    [5, 11],
    [6, 7],
    [8, 2],
    [8, 9],
    [10, 5],
    [11, 0],
    [11, 8],
    [11, 11]
])

# computed:
input_shape = (im_size, im_size, 1)
run_name = f"run-i{im_size}-k{kernel_size}-ki{len(kernel_positions)}-v{run_version}"


def get_data(size=28, resize_method="cv2"):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    x_train = x_train / 255
    x_test = x_test / 255

    if not size == 28:
        if resize_method == "tf":
            x_train = tf.image.resize_images(
                x_train,
                (size, size),
                method=tf.image.ResizeMethod.BILINEAR,
            )

            x_test = tf.image.resize_images(
                x_test,
                (size, size),
                method=tf.image.ResizeMethod.BILINEAR,
            )
        elif resize_method == "cv2":
            x_train = np.array([
                cv2.resize(x, (size, size))
                for x in x_train])

            x_test = np.array([
                cv2.resize(x, (size, size))
                for x in x_test])
        else:
            raise ValueError("Invalid resize method. Use 'cv2' or 'tf'.")

    x_train = x_train.reshape(x_train.shape[0], size, size, 1)
    x_test = x_test.reshape(x_test.shape[0], size, size, 1)

    # onehot labels
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


def datagen(batch_size, x, y, method="cycle"):
    if method == "random":
        while True:
            # choose random batch sample
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            x_batch = x[idx]
            y_batch = y[idx]
            yield (x_batch, y_batch)
    elif method == "cycle":
        while True:
            n_samples = int(x_train.shape[0])

            for i in range(int(n_samples/batch_size)):
                batch_start = i * batch_size
                batch_end = (i+1) * batch_size

                x_batch = x[batch_start:batch_end]
                y_batch = y[batch_start:batch_end]

                yield (x_batch, y_batch)
    else:
        raise ValueError("Method is not supported. Use 'random' or 'cycle'.")


def build_baseline(input_shape, kernel_size, compile=True, *args, **kwargs) -> keras.Model:
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(kernel_size, kernel_size),
                     activation='linear',
                     use_bias=None,
                     input_shape=input_shape))
    model.add(tf.keras.layers.Lambda(lambda x: tf.nn.relu(x)))

    # model.add(Conv2D(64, (3, 3), activation='linear', use_bias=None))
    # model.add(tf.keras.layers.Lambda(lambda x: tf.nn.relu(x)))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    if compile:
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

    return model


def build_dilated(input_shape, kernel_size, compile=True, *args, **kwargs) -> keras.Model:
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(kernel_size, kernel_size),
                     activation='linear',
                     use_bias=None,
                     dilation_rate=(2, 2),
                     input_shape=input_shape))
    model.add(tf.keras.layers.Lambda(lambda x: tf.nn.relu(x)))

    # model.add(Conv2D(64, (3, 3), activation='linear', use_bias=None))
    # model.add(tf.keras.layers.Lambda(lambda x: tf.nn.relu(x)))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    if compile:
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

    return model


def build_interpolated(input_shape, kernel_size, kernel_positions, compile=True, *args, **kwargs) -> keras.Model:
    model = Sequential()

    kcs = symmetric_filters(kernel_positions, 32)
    model.add(InterpolatedConv2d(kcs, kernel_size, input_shape=input_shape))

    model.add(tf.keras.layers.Lambda(lambda x: tf.nn.relu(x)))

    # kcs = symmetric_filters(kernel_positions, 64)
    # model.add(InterpolatedConv2d(kcs,3))
    # model.add(tf.keras.layers.Lambda(lambda x: tf.nn.relu(x)))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    if compile:
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

    return model


def run_model(model_build_function, name=None, verbose=True):
    """Trains the model

    Assumes `x_train`,`x_test`,`y_train`,`y_test` exist in global scope.
    Assumes params exist in global scope.

    Args:
        model_build_function (callable): A function that returns a compiled model
        name (str, optional): Defaults to None. name of model
        verbose (bool, optional): Defaults to True. to spam or not to spam console

    Returns:
        dict: metrics
    """

    if not name:
        name = model_build_function.__str__()

    # run model
    model = model_build_function(input_shape, kernel_size=kernel_size,
                                 kernel_positions=kernel_positions, compile=True)

    verbose_fixed = 1 if verbose else 0

    t = Timer().start()
    hist = model.fit(x_train, y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=verbose_fixed,
                     validation_data=(x_test, y_test))
    t.stop()

    score = model.evaluate(x_test, y_test, verbose=0)

    if verbose:
        print(name)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        print('Time:', t.delta)
        print(model.summary())

    return {
        "name": name,
        "training_time": t.delta,
        "test_loss": score[0],
        "test_accuracy": score[1],
    }


def save_run(res_dict: dict, path: str=None):
    if not path:
        fname = res_dict["name"] + ".res.pkl"

        directory = os.path.join("results", run_name)

        # create directories
        if not os.path.exists(directory):
            os.makedirs(directory)

        path = os.path.join(directory, fname)

    # save data
    with open(path, 'wb') as f:
        pickle.dump(res_dict, f)

    return 0


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = get_data(im_size)
    if verbose:
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

    # run models
    res_dict_baseline = run_model(build_baseline, "Baseline", verbose)
    res_dict_interpolated = run_model(
        build_interpolated, "Interpolated", verbose)
    # res_dict_dilated = run_model(build_dilated, "Dilated", verbose)

    # save runs
    save_run(res_dict_baseline)
    save_run(res_dict_interpolated)
    # save_run(res_dict_dilated)
