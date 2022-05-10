#import torch # torch 1.9.0+cu111
import numpy as np
from compare import *


output_c = np.fromfile("c", dtype=np.int8)
output_py = np.fromfile("p", dtype=np.int8)
compare_two_tensor_uint8(output_py, output_c)
#compare_two_tensor_uint8_2(output_py, output_c)