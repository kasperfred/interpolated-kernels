from typing import List
import numpy as np

def symmetric_filters(known_positions: List, n_filters: int):
    kc = [known_positions for _ in range(n_filters)]
    return np.array(kc)


def dilated_interpolated_filters(kernel_size: int, dilation_rate:int) -> List:
    """computes the kernel positions as if it was a dilated kernel
    
    Args:
        kernel_size (int): size of the kernel before 'dilation' - assumes square kernel
        dilation_rate (int): would be dilation rate; assumes square dilation
    
    Returns:
        List: kernel positions
    """
    kernel_positions = []
    for h in range(kernel_size):
        for w in range(kernel_size):
            q = [h*dilation_rate,w*dilation_rate]
            kernel_positions.append(q)
    return kernel_positions


def auto_filter_positions(kernel_size:int, spacing:int) -> List:
    kernel_positions = []

    crow = 0
    ccol = 0
    while crow < kernel_size:
        while ccol < kernel_size:
            q = [crow, ccol]
            kernel_positions.append(q)
            ccol+=spacing
        crow+=spacing
        ccol=0

    return kernel_positions


def required_kernel_size(kernel_positions:List) -> int:
    max_x = 0
    max_y = 0

    for p in kernel_positions:
        if p[0] > max_x:
            max_x = p[0]
        if p[1] > max_y:
            max_y = p[1]
    
    return max_x if max_x > max_y else max_y




if __name__ == "__main__":
    kernel_size = 5
    spacing = 2


    kernel_positions = auto_filter_positions(kernel_size,spacing)
    print (kernel_positions)

    import matplotlib.pyplot as plt

    for q in kernel_positions:
        plt.scatter(q[0],q[1])
    
    plt.show()
