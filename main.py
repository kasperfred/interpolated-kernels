import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# sess = tf.Session()

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize
x_train, x_test = x_train / 255.0, x_test / 255.0

# resize
x_train, x_test = [tf.expand_dims(x, 3) for x in [x_train, x_test]]

# onehot
y_train, y_test = map(tf.keras.utils.to_categorical, [y_train, y_test])


# def datagen(batch_size:int=256):
#     # import data
#     mnist = tf.keras.datasets.mnist
#     (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
#     # normalize
#     x_train, x_test = x_train / 255.0, x_test / 255.0

#     # resize
#     x_train, x_test = [tf.expand_dims(x, 3) for x in [x_train, x_test]]

#     # onehot
#     y_train, y_test = map(tf.keras.utils.to_categorical, [y_train, y_test])

#     # generator
#     while True:
#         # choose random sample
#         idx = np.random.randint(0, x_train.shape[0], batch_size)
#         x_batch = x_train[idx]
#         y_batch = y_train[idx]
#         yield (x_batch, y_batch)

# # upscale
# x_train_large = tf.image.resize_images(
#     x_train,
#     (28*4,28*4),
#     method=tf.image.ResizeMethod.BILINEAR,
# )
# x_test_large = tf.image.resize_images(
#     x_test,
#     (28*4,28*4),
#     method=tf.image.ResizeMethod.BILINEAR,
# )

# print (x_train_large.shape)
# plt.imshow(sess.run(tf.squeeze(x_train_large[0],2)))
# plt.show()

from src.layers import InterpolatedConv2d
from src.utils import symmetric_filters

def get_model_baseline(input_shape, compile=True):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                                    activation="linear",
                                    use_bias=None,
                                    input_shape=input_shape))
    model.add(tf.keras.layers.Lambda(lambda x: tf.nn.relu(x)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), 
                                    activation="linear",
                                    use_bias=None))
    model.add(tf.keras.layers.Lambda(lambda x: tf.nn.relu(x)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    if compile: 
        model.compile(optimizer=tf.train.AdamOptimizer(),
                    loss="categorical_crossentropy",
                    metrics=['accuracy'])
    return model

def get_model_interpolated():
    pass




baseline = get_model_baseline([28,28,1], compile=True)

baseline.fit(x_train, y_train,
          steps_per_epoch=60000,
          epochs=3,
          verbose=1,
          validation_data=(x_test, y_test))
# hist = baseline.fit_generator(dg, steps_per_epoch=200, epochs=3)

import sys;sys.exit()
img = np.ones([11,11,3])
kcs = symmetric_filters(np.array([
        #h,w,c
        [0,0,0],
        [2,1,1],
        [1,1,2],
        [3,2,0]
    ]), 2)

print (kcs.shape)

layer = InterpolatedConv2d(kcs, 4)

img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
img_tensor = tf.reshape(img_tensor, [1,11,11,3])

v = layer(img_tensor)
print ("v_shape", v.shape)

tf.keras.layers.Conv2D(32, kernel_size=(4, 4),
                        activation='relu',
                        input_shape=[11,11,3])

# model = tf.keras.models.Sequential([
#     InterpolatedConv2d(kc, 4, input_shape=[11,11,3])
# ])

# model.compile(optimizer=tf.train.AdamOptimizer(),
#         loss="categorical_crossentropy",
#         metrics=['accuracy'])

# model.fit(img_tensor, [v], steps_per_epoch=1)

# print (model.summary())