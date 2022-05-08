#import torch # torch 1.9.0+cu111
import numpy as np
from compare import *

if 1:
    output_c = np.fromfile("c", dtype=np.float32)
    # for t in range(int(len(output_c)/6)):
    #     tt = output_c[t * 6 + 4]
    #     if tt > 0.6 :
    #         print(tt)
    #         print(output_c[t * 6 ])
    #         print(output_c[t * 6 + 1])
    #         print(output_c[t * 6 + 2])
    #         print(output_c[t * 6 + 3])
    #         print(output_c[t * 6 + 4])
    #         print(output_c[t * 6 + 5])


    output_py = np.fromfile("p", dtype=np.float32)
    compare_two_tensor(output_py, output_c)
else:
    output_c = np.fromfile("trt_1", dtype=np.int8)
    output_py = np.fromfile("py_0", dtype=np.int8)
    compare_two_tensor_uint8(output_py, output_c)
    #compare_two_tensor_uint8_2(output_py, output_c)