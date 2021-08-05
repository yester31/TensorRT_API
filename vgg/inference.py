import torch, torchvision, os, cv2, struct
import numpy as np
from torchsummary import summary

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print('gpu device count : ', torch.cuda.device_count())
print('device_name : ', torch.cuda.get_device_name(0))
print('gpu available : ', torch.cuda.is_available())

def main():

    if os.path.isfile('vgg.pth'):                       # vgg.pth 파일이 있다면
        net = torch.load('vgg.pth')                     # vgg.pth 파일 로드
    else:                                               # vgg.pth 파일이 없다면
        net = torchvision.models.vgg11(pretrained=True) # torchvision에서 vgg11 pretrained weight 다운로드 수행
        torch.save(net, 'vgg.pth')                      # vgg.pth 파일 저장

    net = net.eval()                            # vgg 모델을 평가 모드로 세팅
    net = net.to('cuda:0')                      # gpu 설정
    print('model: ', net)                       # 모델 구조 출력
    summary(net, (3, 224, 224))                 # input 사이즈 기준 레이어 별 output shape 및 파라미터 사이즈 출력

    img = cv2.imread('../date/dog224.jpg')      # image file load
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # bgr -> rgb
    img = img.transpose(2, 0, 1)                # hwc -> chw
    img = img.astype(np.float32)                # uint -> float32
    #tofile(img)                                # img variable을 file로 만들기
    img = torch.from_numpy(img)                 # numpy -> tensor
    img = img.unsqueeze(0)                      # [c,h,w] -> [1,c,h,w]
    img = img.to('cuda:0')                      # host -> device

    out = net(img)
    max_index = out.max(dim=1)
    max_value = out.max()
    print('vgg max index : {} , value : {}'.format(max_index, max_value))

    if os.path.isfile('vgg.wts'):
        print('Already, vgg.wts file exists.')
    else:
        print('making vgg.wts file ...')        # vgg.wts 파일이 없다면 생성
        f = open("vgg.wts", 'w')
        f.write("{}\n".format(len(net.state_dict().keys())))
        for k, v in net.state_dict().items():
            print('key: ', k)
            print('value: ', v.shape)
            vr = v.reshape(-1).cpu().numpy()
            f.write("{} {}".format(k, len(vr)))
            for vv in vr:
                f.write(" ")
                f.write(struct.pack(">f", float(vv)).hex())
            f.write("\n")
        print('Completed vgg.wts file!')

if __name__ == '__main__':
    main()

