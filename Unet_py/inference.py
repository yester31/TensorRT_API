import torch
import cv2
import os
import struct
import time
from torchsummary import summary
from unet import UNet
import numpy as np
import torch.nn.functional as F
from torchvision import transforms

#torch.backends.cuda.matmul.allow_tf32 = False
#torch.backends.cudnn.allow_tf32 = False

def tofile(img, file_path = "py_data"):
    with open(file_path, 'wb') as f:
        img.tofile(f)
    f.close()

def fromfile(file_path = "../Validation_py/C_Tensor"):
    with open(file_path, 'rb') as f:
        img = np.fromfile(file_path, dtype=np.float32)
    f.close()
    return img

 # 전처리 및 추론 연산 함수
def infer(img, net, half):
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # bgr -> rgb
    img3 = img2.transpose(2, 0, 1)  # hwc -> chw
    img4 = img3.astype(np.float32)  # uint -> float32
    img4 /= 255  # 1/255
    tofile(img4, "../Validation_py/py_0")
    #exit(0)
    img5 = torch.from_numpy(img4)  # numpy -> tensor
    if half:
        img5 = img5.half()
    img6 = img5.unsqueeze(0)  # [c,h,w] -> [1,c,h,w]
    img6 = img6.to('cuda:0')
    out = net(img6)
    return out

def main():
    half = False
    size = 512
    print('cuda device count: ', torch.cuda.device_count())
    net = UNet(n_channels=3, n_classes=2)
    net = net.to('cuda:0')
    net.load_state_dict(torch.load('unet_carvana_scale0.5_epoch1.pth', map_location='cuda:0'))
    if half:
        net.half()  # to FP16
    net = net.eval()
    #summary(net, (3, size, size))
    img = cv2.imread('data/00ad56bf7ee6_03.jpg')  # image file load
    #img = cv2.imread('car0.jpg')  # image file load
    #tofile(img, "../Validation_py/py_pre")
    if 1 : # 원본 비율 크기 그대로 리사이즈 및 정사각형 입력 사이즈를 위해 여백 추가
        w = img.shape[1]
        h = img.shape[0]
        if w >= h :
            nw = size
            nh = (int)(h * (size / w))
        else:
            nw = (int)(w * (size / h))
            nh = size

        tb = (int)((size - nh) / 2)
        bb = tb + 1 if nh % 2 == 1 else tb
        lb = (int)((size - nw) / 2)
        rb = lb + 1 if nw % 2 == 1 else lb
        #img0 = cv2.resize(img, dsize=(nw, nh), interpolation=cv2.INTER_AREA) // c함수와 정합성 틀림.
        img0 = cv2.resize(img, dsize=(nw, nh), interpolation=cv2.INTER_LINEAR)
        #tofile(img0, "../Validation_py/py_0") # c 코드와 정합성 일치
        img1 = cv2.copyMakeBorder(img0, tb, bb, lb, rb, cv2.BORDER_CONSTANT, value=(128,128,128))
        #tofile(img1, "../Validation_py/py_1")  # c 코드와 정합성 일치
    else:
        img1 = cv2.resize(img, dsize=(size, size), interpolation=cv2.INTER_AREA)

    # 속도 측정에서 첫 1회 연산 제외하기 위한 계산
    out = infer(img1, net, half)
    tofile(out.cpu().data.numpy(), '../Validation_py/py')
    if 1 :
        dur_time = 0
        iteration = 100
        for i in range(iteration):
            begin = time.time()
            out = infer(img1, net, half)
            dur = time.time() - begin
            dur_time += dur
            #print('{} dur time : {}'.format(i, dur))
        print('{} iteration time : {} [sec]'.format(iteration, dur_time))

    #tofile(out.cpu().data.numpy(), '../Validation_py/output_py')
    #print('output:', out)
    full_mask = F.softmax(out, dim=1)[0].cpu().squeeze()
    tt1 = F.one_hot(full_mask.argmax(dim=0), 2).permute(2, 0, 1).numpy()
    tt2 = np.argmax(tt1, axis=0) * 255
    tt3 = tt2.astype(np.uint8)
    cv2.imshow('tt', tt3)
    cv2.waitKey(0)

    if 0:  # LIST 형태 웨이트 파일 생성 로직
        weights = net.state_dict()
        weight_list = [(key, value) for (key, value) in weights.items()]
        for idx in range(len(weight_list)):
            key, value = weight_list[idx]
            if "num_batches_tracked" in key:
                print(idx, "--------------------")
                continue
            print(idx, key, value.shape)

    if os.path.isfile('unet.wts'):
        print('Already, unet.wts file exists.')
    else:
        print('making unet.wts file ...')        # vgg.wts 파일이 없다면 생성
        f = open("unet.wts", 'w')
        f.write("{}\n".format(len(net.state_dict().keys())))
        for k,v in net.state_dict().items():
            print('key: ', k)
            print('value: ', v.shape)
            vr = v.reshape(-1).cpu().numpy()
            f.write("{} {}".format(k, len(vr)))
            for vv in vr:
                f.write(" ")
                f.write(struct.pack(">f", float(vv)).hex())
            f.write("\n")

if __name__ == '__main__':
    main()
    if 0:
        out_c = fromfile("../Validation_py/py")
        out_c = torch.from_numpy(out_c).to('cuda:0')
        out = out_c.reshape(1, 2, 512, 512)
        full_mask = F.softmax(out, dim=1)[0].cpu().squeeze()
        tt1 = F.one_hot(full_mask.argmax(dim=0), 2).permute(2, 0, 1).numpy()
        tt2 = np.argmax(tt1, axis=0) * 255
        tt3 = tt2.astype(np.uint8)
        cv2.imshow('tt', tt3)
        cv2.waitKey()
