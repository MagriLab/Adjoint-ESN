from multiprocessing.pool import ThreadPool

import numpy as np


def sum_op(arr1, arr2):
    result = np.add(arr1, arr2)
    return result


def sum_op_split(coords, arr1, arr2, result):
    # unpack array indexes
    i1, i2, i3, i4 = coords
    _ = np.add(arr1[i1:i2, i3:i4], arr2[i1:i2, i3:i4], out=result[i1:i2, i3:i4])


def sum_op_multi(arr1, arr2, n):
    result = np.empty((n, n))
    with ThreadPool(4) as pool:
        # split each dimension (divisor of matrix dimension)
        split = round(n / 2)
        # issue tasks
        for x in range(0, n, split):
            for y in range(0, n, split):
                # determine matrix coordinates
                coords = (x, x + split, y, y + split)
                # issue task
                _ = pool.apply_async(sum_op_split, args=(coords, arr1, arr2, result))
                # sum_op3(coords, dfdx_x_new, dfdx_u, result)

        # close the pool
        pool.close()
        # wait for tasks to complete
        pool.join()
    return result
