#import torch # torch 1.9.0+cu111
import numpy as np
import cv2
from compare import *

img = cv2.imread('../TestDate/panda0.jpg')
#img = cv2.resize(img, (224,224))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.transpose(2, 0, 1)
img = img.astype(np.float32)
img = img/255
output_py = img.flatten()

output_c = np.fromfile("C_Preproc_Result", dtype=np.float32)

compare_two_tensor2(output_py, output_c)
