import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
from timer import Timer

# params
batch_size = 128
epochs = 12
im_size = 28

# kernel params
kernel_size = 3 # effective
kernel_positions = np.array([
    # h w 
    [0,0],
    [2,0],
    [0,2],
    [2,2]
])

# computed:
input_shape = (im_size, im_size, 1)

# the data, split between train and test sets
def get_data(size=28):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], size, size, 1)
    x_test = x_test.reshape(x_test.shape[0], size, size, 1)

    if not size == 28:
        x_train = tf.image.resize_images(
            x_train,
            (size,size),
            method=tf.image.ResizeMethod.BILINEAR,
        )

        x_test = tf.image.resize_images(
            x_test,
            (size,size),
            method=tf.image.ResizeMethod.BILINEAR,
        )
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = get_data(im_size)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

def build_baseline(input_shape, kernel_size, compile=True, *args, **kwargs):
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
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    if compile:
        model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])
    
    return model


from src.layers import InterpolatedConv2d
from src.utils import symmetric_filters
def build_interpolated(input_shape, kernel_size, kernel_positions, compile=True, *args, **kwargs):
    model = Sequential()

    kcs = symmetric_filters(kernel_positions, 32)
    model.add(InterpolatedConv2d(kcs,kernel_size, input_shape=[28,28,1]))

    model.add(tf.keras.layers.Lambda(lambda x: tf.nn.relu(x)))

    # kcs = symmetric_filters(kernel_positions, 64)
    # model.add(InterpolatedConv2d(kcs,3))
    # model.add(tf.keras.layers.Lambda(lambda x: tf.nn.relu(x)))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
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
    model = build_baseline(input_shape, kernel_size=kernel_size, kernel_positions=kernel_positions, compile=True)

    t = Timer().start()
    hist = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
    t.stop()

    score = model.evaluate(x_test, y_test, verbose=0)

    if verbose:
        print (name)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        print('Time:', t.delta)
        print (model.summary())
    
    return {
        "name": name,
        "training_hist": hist,
        "training_time": t.delta,
        "test_loss": score[0],
        "test_accuracy": score[1]
    }



