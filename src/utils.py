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

    # dilation rate is like horizontal and vertical skip

    # 0,0
    # 1*dilation_rate, 0
    # kernel_size*dilation_rate, 0
    res = []
    for h in range(kernel_size):
        for w in range(kernel_size):
            q = [h*dilation_rate,w*dilation_rate]
            res.append(q)
    return res



def required_kernel_size(kernel_positions:List) -> int:
    max_x = 0
    max_y = 0

    for p in kernel_positions:
        if p[0] > max_x:
            max_x = p[0]
        if p[1] > max_y:
            max_y = p[1]
    
    return max_x if max_x > max_y else max_y

