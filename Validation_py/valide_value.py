#import torch # torch 1.9.0+cu111
import numpy as np
from compare import *

#output_c = np.fromfile("trt0", dtype=np.float32) # tf32 off
output_c = np.fromfile("trt", dtype=np.float32) # tf32 on
#output_py = np.fromfile("py0", dtype=np.float32) # tf32 off
output_py = np.fromfile("py", dtype=np.float32) # tf32 on
compare_two_tensor2(output_py, output_c)

# output_c = np.fromfile("trt_1", dtype=np.int8)
# output_py = np.fromfile("py_1", dtype=np.int8)
# compare_two_tensor_uint8(output_py, output_c)