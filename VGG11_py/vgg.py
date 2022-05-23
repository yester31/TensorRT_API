import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import cv2
import numpy as np

def tofile(img, weight_path = "input2"):
    with open(weight_path, 'wb') as f:
        img.tofile(f)
    f.close()

def main():
    print('cuda device count: ', torch.cuda.device_count())
    net = torchvision.models.vgg11(pretrained=True)
    net = net.eval()
    net = net.to('cuda:0')
    #print(net)

    img = cv2.imread("./data/panda0.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = img.astype(np.float32)
    #tofile(img)

    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.to('cuda:0')

    out = net(img)
    max_index = out.max(dim=1)
    max_value = out.max()
    print('vgg11 max index : {} , value : {}'.format( max_index,max_value ))
    print('vgg11 out:', out.shape)
    torch.save(net, "vgg11.pth")

if __name__ == '__main__':
    main()

