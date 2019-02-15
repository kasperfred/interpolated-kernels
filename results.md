# Results

## 28x28 small view window
```
batch_size = 128
epochs = 12
im_size = 28

# kernel params
kernel_size = 3
kernel_positions = np.array([
    # h w 
    [0,0],
    [2,0],
    [0,2],
    [2,2]
])
```

### Interpolated (normalized):
Test loss: 0.04205631780798139
Test accuracy: 0.9861
Time: 101.52

Params: 128

### Baseline
Test loss: 0.038591371865413386
Test accuracy: 0.9865
Time: 58.22

Params: 288


### Analysis
Since conv layer was the first layer, we can see the constant time increase:

$$\frac{101s-58s}{288\text{param}-128\text{param}} = 42s/160\text{param} = 0.26\text{s/param/run}$$
$$=0.022 \text{s/param/epoch}$$

Param %:

$$\frac{128\text{param}}{288\text{param}} \cdot 100\% = 44.5\%$$




## 112x112 (x4) small view window 
```
# debug
verbose = True
run_version = 0

# params
batch_size = 128
epochs = 6
im_size = 28*4

# kernel params
kernel_size = 3  # effective
kernel_positions = np.array([
    # h w
    [0, 0],
    [2, 0],
    [1, 1],
    [0, 2],
    [2, 2]
])
```

### Interpolated
Test loss: 0.0748734971286729
Test accuracy: 0.9758
Time: 627.02

Params: 160

### Baseline
Test loss: 0.0648
Test accuracy:  0.9796
Time: 1458

Params: 288

### Analysis



## 112x112 (x4) large window same number interp params
```
# debug
verbose = True
run_version = 0

# params
batch_size = 128
epochs = 6
im_size = 28*4

# kernel params
kernel_size = 3*4  # effective
kernel_positions = np.array([
    # h w
    [0, 0],
    [2, 0],
    [1, 1],
    [0, 2],
    [2, 2]
])
```
### Interpolated
CUDA handler acting weird. Don't trust relative time. 

Test loss: 0.07425058183657166
Test accuracy: 0.9785
Time: 1423.06

Params: 160


## Interpolated fixed indices
CUDA handler acting weird. Don't trust relative time. 
```
kernel_positions = np.array([
    # h w
    [0, 0],
    [11, 0],
    [5, 5],
    [0, 11],
    [11, 11]
])
```
Test loss: 0.05556277578609297
Test accuracy: 0.9817
Time: 1422.66

Params: 160


### Baseline
Test loss: 0.04826091565887182
Test accuracy: 0.9847
Time: 819.23
Params: 4608

### Dilated

Test loss: 0.03538157455020464
Test accuracy: 0.9881
Time: 568.97

Params 4608



## 112x112 (x4) large window same more params params
```
# debug
verbose = True
run_version = 1

# disable GPU due to cuda handler not being able to register
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# params
batch_size = 128
epochs = 6
im_size = 28*4

# kernel params
kernel_size = 3*4  # effective
kernel_positions = np.array([
    # h w
    [0, 0],
    [0,11],
    [0,8]
    [1, 5],
    [3, 2],
    [3, 9],
    [5, 0],
    [5,4],
    [5,11],
    [6,7],
    [8,2],
    [8,9],
    [10,5],
    [11,0],
    [11,8],
    [11,11]
])
```


### Interpolated
Test loss: 0.054953625461518094
Test accuracy: 0.9836
Time: 584.77

Params: 512