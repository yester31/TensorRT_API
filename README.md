# TensorRT_EX

## Enviroments
***
- Windows 10 laptop
- CPU i7-11375H
- GPU RTX-3060
- Visual studio 2017
- CUDA 11.1
- TensorRT 8.0.3.4 (unet)
- TensorRT 8.2.0.6 (detr) 
- Cudnn 8.2.1
- Opencv 3.4.5
***

## custom plugin 
- Layer that perform image preprocessing(NHWC->NCHW, BGR->RGB, [0, 255]->[0, 1] (Normalize))
- plugin_ex1.cpp (plugin sample code)
- preprocess.hpp (plugin define)
- preprocess.cu (preprocessing cuda kernel function)
- Validation_py/Validation_preproc.py (Result validation with pytorch)
***

## Simple Classification model
- vgg11 model (vgg11.cpp)
- with preprocess plugin
- Easy-to-use structure (regenerated according to the presence or absence of engine files)
- Easier and more intuitive code structure
- About 2 times faster than PyTorch(Comparison of calculation execution time of 100 iteration for one 224x224x3 image)
***

## TensorRT PTQ model
- resnet18 model (ptq_ex1.cpp)
- 100 images from COCO val2017 dataset for PTQ calibration
- Comparison of calculation execution time of 100 iteration and GPU memory usage for one 224x224x3 image 
- Pytorch  F32	-> 389 ms (1.449 GB)
- Pytorch  F16	-> 330 ms (1.421 GB)
- TensorRT F32	-> 199 ms (1.356 GB)
- TensorRT F16	-> 58 ms  (0.922 GB)
- TensorRT Int8 -> 40 ms  (0.870 GB) (PTQ)
- Match all results with PyTorch
***

## Semantic Segmentaion model
- TensorRT 8.0.3.4 (unet)
- UNet model (unet.cpp)
- Comparison of calculation execution time of 100 iteration and GPU memory usage for one 512x512x3 image
- Pytorch  F32	-> 6621 ms (3.863 GB)
- Pytorch  F16	-> 3458 ms (2.677 GB)
- TensorRT F32	-> 4722 ms (1.600 GB)
- TensorRT F16	-> 1858 ms (1.080 GB) 
- TensorRT Int8 -> 938 ms  (1.051 GB) (PTQ)
- additional preprocess (resize & letterbox padding) with openCV
- postprocess (model output to image)
- Match all results with PyTorch
***

## Object Detection model(ViT)
- TensorRT 8.2.0.6 (detr) 
- DETR model (detr_trt.cpp) 
- Comparison of calculation execution time of 100 iteration and GPU memory usage for one 500x500x3 image 
- Pytorch  F32	-> 3703 ms (1.563 GB)
- Pytorch  F16	-> 3071 ms (1.511 GB)
- TensorRT F32	-> 1640 ms (1.212 GB)
- TensorRT F16	->  607 ms (1.091 GB) 
- TensorRT Int8 ->  530 ms (1.005 GB) (PTQ)
- additional preprocess (mean std normalization function)
- postprocess (show out detection result to the image)
- Match all results with PyTorch
***

## Using C TensoRT model in Python using dll
- TRT_DLL_EX : <https://github.com/yester31/TRT_DLL_EX>
***

***

## A typical TensorRT model creation sequence using TensorRT API
0. Prepare the trained model in the training framework (generate the weight file to be used in TensorRT).
1. Implement the model using the TensorRT API to match the trained model structure.
2. Extract weights from the trained model.
3. Make sure to pass the weights appropriately to each layer of the prepared TensorRT model.
4. Build and run.
5. After the TensorRT model is built, the model stream is serialized and generated as an engine file.
6. Inference by loading only the engine file in the subsequent task(if model parameters or layers are modified, re-execute the previous (4) task).
     
***

## reference   
* tensorrtx : <https://github.com/wang-xinyu/tensorrtx>
* unet : <https://github.com/milesial/Pytorch-UNet>
* detr : <https://github.com/facebookresearch/detr>