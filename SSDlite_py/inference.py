import torch, torchvision, os, cv2, struct, time
import numpy as np
from torchsummary import summary

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print('gpu device count : ', torch.cuda.device_count())
print('device_name : ', torch.cuda.get_device_name(0))
print('gpu available : ', torch.cuda.is_available())

model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True) # pytorch 1.9
#model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
model.eval()
model.to('cuda:0')
img = cv2.imread('../TestDate/panda0.jpg')  # image file load
img1 = cv2.resize(img, dsize=(320, 320), interpolation=cv2.INTER_AREA)
img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32)/255
x = torch.from_numpy(img2).unsqueeze(0).to('cuda:0')

#x = [torch.rand(3, 320, 320), torch.rand(3, 500, 400)]
predictions = model(x)
print(predictions)