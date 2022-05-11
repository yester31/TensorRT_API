import math
from PIL import Image
import requests
import matplotlib.pyplot as plt
import struct
import cv2
import time
#import ipywidgets as widgets
#from IPython.display import display, clear_output

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False)

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    #T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

def gen_wts(model, filename):
    f = open(filename + '.wts', 'w')
    f.write('{}\n'.format(len(model.state_dict().keys()) + 72))
    for k, v in model.state_dict().items():
        if 'in_proj' in k:
            dim = int(v.size(0) / 3)
            q_weight = v[:dim].reshape(-1).cpu().numpy()
            k_weight = v[dim:2*dim].reshape(-1).cpu().numpy()
            v_weight = v[2*dim:].reshape(-1).cpu().numpy()
            f.write('{} {} '.format(k + '_q', len(q_weight)))
            for vv in q_weight:
                f.write(' ')
                f.write(struct.pack('>f', float(vv)).hex())
            f.write('\n')

            f.write('{} {} '.format(k + '_k', len(k_weight)))
            for vv in k_weight:
                f.write(' ')
                f.write(struct.pack('>f', float(vv)).hex())
            f.write('\n')

            f.write('{} {} '.format(k + '_v', len(v_weight)))
            for vv in v_weight:
                f.write(' ')
                f.write(struct.pack('>f', float(vv)).hex())
            f.write('\n')
        else:
            vr = v.reshape(-1).cpu().numpy()
            f.write('{} {} '.format(k, len(vr)))
            for vv in vr:
                f.write(' ')
                f.write(struct.pack('>f',float(vv)).hex())
            f.write('\n')
    f.close()



#gen_wts(model, "detr")
if 0:  # LIST 형태 웨이트 파일 생성 로직
    weights = model.state_dict()
    weight_list = [(key, value) for (key, value) in weights.items()]
    for idx in range(len(weight_list)):
        key, value = weight_list[idx]
        if "num_batches_tracked" in key:
            print(idx, "--------------------")
            continue
        print(idx, key, value.shape)



def infer(img, model, half):
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img2 = Image.fromarray(img1) # convert from openCV2 to PIL
    img3 = transform(img2).unsqueeze(0)# mean-std normalize the input image (batch-size: 1)
    if half:
        img3 = img3.half()
    # propagate through the model
    img4 = img3.to('cuda:0')
    outputs = model(img4)
    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    return probas, outputs

def main():
    half = True
    model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
    model = model.to('cuda:0')                      # gpu 설정
    if half:
        model.half()  # to FP16
    model.eval()
    gen_wts(model, 'detr')
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    im = Image.open(requests.get(url, stream=True).raw)

    img = cv2.imread('data/000000039769.jpg')  # image file load
    img0 = cv2.resize(img, dsize=(500, 500), interpolation=cv2.INTER_LINEAR)

    dur_time = 0
    iteration = 100

    # 속도 측정에서 첫 1회 연산 제외하기 위한 계산
    probas, outputs = infer(img0, model, half)

    for i in range(iteration):
        begin = time.time()
        probas, outputs = infer(img0, model, half)
        dur = time.time() - begin
        dur_time += dur
        # print('{} dur time : {}'.format(i, dur))

    print('{} iteration time : {} [sec]'.format(iteration, dur_time))

    keep = probas.max(-1).values > 0.9
    tt = probas.max(-1).values.sort(descending=True)
    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'].cpu().data[0, keep], im.size)

    plot_results(im, probas[keep], bboxes_scaled)

if __name__ == '__main__':
    main()