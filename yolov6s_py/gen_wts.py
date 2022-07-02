import sys
import argparse
import os
import struct
import torch

from yolov6.layers.common import DetectBackend

wts_file = 'yolov6s.wts'

# Initialize
weights = 'yolov6s.pt'
device = 'cpu'
cuda = device != 'cpu' and torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

# Load model
model = DetectBackend(weights, device=device, fuse=True).model

if 1:  # LIST 형태 웨이트 파일 생성 로직
    weights = model.state_dict()
    weight_list = [(key, value) for (key, value) in weights.items()]
    for idx in range(len(weight_list)):
        key, value = weight_list[idx]
        if "num_batches_tracked" in key:
            print(idx, "--------------------")
            continue
        print(idx, key, value.shape)

if 0:
    with open(wts_file, 'w') as f:
        f.write('{}\n'.format(len(model.state_dict().keys())))
        for k, v in model.state_dict().items():
            vr = v.reshape(-1).cpu().numpy()
            f.write('{} {} '.format(k, len(vr)))
            for vv in vr:
                f.write(' ')
                f.write(struct.pack('>f' ,float(vv)).hex())
            f.write('\n')
else :
    if os.path.isfile('yolov6s.wts'):
        print('Already, yolov6s.wts file exists.')
    else:
        print('making yolov6s.wts file ...')        # vgg.wts 파일이 없다면 생성
        f = open("yolov6s.wts", 'w')
        f.write("{}\n".format(len(model.state_dict().keys())))
        for k, v in model.state_dict().items():
            print('key: ', k)
            print('value: ', v.shape)
            vr = v.reshape(-1).cpu().numpy()
            f.write("{} {}".format(k, len(vr)))
            for vv in vr:
                f.write(" ")
                f.write(struct.pack(">f", float(vv)).hex())
            f.write("\n")
        print('Completed resnet18.wts file!')