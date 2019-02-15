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
(no GPU)




### Baseline
Test loss: 0.04826091565887182
Test accuracy: 0.9847
Time: 819.23
Params: 4608

### Analysis



## 112x112 (x4) large window same more params params
```
# debug
verbose = True
run_version = 0

# params
batch_size = 128
epochs = 12
im_size = 28*4

# kernel params
kernel_size = 3*4  # effective
kernel_positions = np.array([
    # h w
    [0, 0],
    [2, 0],
    [1,1],
    [0, 2],
    [2, 2]
])
```


### Interpolated


### Baseline

### Analysis