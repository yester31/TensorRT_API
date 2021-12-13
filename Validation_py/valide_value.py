#import torch # torch 1.9.0+cu111
import numpy as np
from compare import *

if 0:
    output_c = np.fromfile("trt_3", dtype=np.float32)
    output_py = np.fromfile("py_3", dtype=np.float32)
    compare_two_tensor2(output_py, output_c)
else:
    output_c = np.fromfile("trt_1", dtype=np.int8)
    output_py = np.fromfile("py_0", dtype=np.int8)
    compare_two_tensor_uint8(output_py, output_c)