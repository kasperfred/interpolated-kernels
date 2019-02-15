# from src.layers import InterpolatedConv2d
# from src.utils import symmetric_filters

# import numpy as np
# import tensorflow as tf

# kernel_positions = np.array([
#         # h w c
#         [0,0,0],
#         [2,0,0],
#         [0,2,0],
#         [2,2,0]
#     ])

# kcs = symmetric_filters(kernel_positions, 32)
# l = InterpolatedConv2d(kcs,3, input_shape=[1,28,28,1])

# inpt = tf.convert_to_tensor(np.ones([1,28,28,1]), dtype=tf.float32)
# print (inpt.shape)
# print (l(inpt))

# img = tf.convert_to_tensor(np.ones([1,11,11,1]) , dtype=tf.float32)
# kcs = symmetric_filters(np.array([
#         #h,w,c
#         [0,0],
#         [2,1],
#         [1,1],
#         [3,2]
#     ]), 32)

# layer = InterpolatedConv2d(kcs, 3)
# v = layer(img)
# print (v)


import os
import pickle

run_name = "test"


# def save_run(res_dict: dict, path: str=None):
#     if not path:
#         fname = res_dict["name"] + ".res.pkl"

#         directory = os.path.join("results", run_name)

#         # create directories
#         if not os.path.exists(directory):
#             os.makedirs(directory)

#         path = os.path.join(directory, fname)

#     # save data
#     with open(path, 'wb') as f:
#         pickle.dump(res_dict, f)

#     return 0


# save_run({"name": "hi"})


import tensorflow as tf
import numpy as np

a = np.array([
    [1,2,3,4],
    [1,2,3,4],
    [1,2,3,4]
])

a = tf.convert_to_tensor(a)

print (a)

print (np.array(a))