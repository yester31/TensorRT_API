# TensorRT_EX

## Enviroments
***
- Windows 10 laptop
- CPU i7-11375H
- GPU RTX-3060
- Visual studio 2017
- CUDA 11.1
- TensorRT 8.0.3.4 (unet)
- TensorRT 8.2.0.6 (detr, yolov5s, real-esrgan) 
- Opencv 3.4.5
- make Engine directory for engine file
- make Int8_calib_table directory for ptq calibration table
***

### Custom plugin example
- Layer for input preprocess(NHWC->NCHW, BGR->RGB, [0, 255]->[0, 1] (Normalize))
- plugin_ex1.cpp (plugin sample code)
- preprocess.hpp (plugin define)
- preprocess.cu (preprocessing cuda kernel function)
- Validation_py/Validation_preproc.py (Result validation with pytorch)
***

## Classification model
### vgg11 model 
- vgg11.cpp
- with preprocess plugin

***

### resnet18 model
- resnet18.cpp
- 100 images from COCO val2017 dataset for PTQ calibration
- Match all results with PyTorch
- Comparison of calculation execution time of 100 iteration and GPU memory usage for one 224x224x3 image 

<table border="0"  width="100%">
	<tbody align="center">
		<tr>
			<td></td>
			<td><strong>Pytorch</strong></td><td><strong>TensorRT</strong></td><td><strong>TensorRT</strong></td><td><strong>TensorRT</strong></td>
		</tr>
		<tr>
			<td>Precision</td><td>FP32</td><td>FP32</td><td>FP16</td><td>Int8(PTQ)</td>
		</tr>
		<tr>
			<td>Avg Duration time [ms]</td>
			<td>4.1 ms</td>
			<td>1.7 ms </td>
			<td>0.7 ms</td>
			<td>0.6 ms</td>
		</tr>
		<tr>
			<td>FPS [frame/sec]</td>
			<td>243 fps</td>
			<td>590 fps</td>
			<td>1385 fps</td>
			<td>1577 fps</td>
		</tr>
		<tr>
			<td>Memory [GB]</td>
			<td>1.551 GB</td>
			<td>1.288 GB</td>
			<td>0.941 GB</td>
			<td>0.917 GB</td>
		</tr>
	</tbody>
</table>

***

## Semantic Segmentaion model
- UNet model (unet.cpp)
- use TensorRT 8.0.3.4 version for unet model(For version 8.2.0.6, an error about the unet model occurs)
- unet_carvana_scale0.5_epoch1.pth
- additional preprocess (resize & letterbox padding) with openCV
- postprocess (model output to image)
- Match all results with PyTorch
- Comparison of calculation execution time of 100 iteration and GPU memory usage for one 512x512x3 image

<table border="0"  width="100%">
	<tbody align="center">
		<tr>
			<td></td>
			<td><strong>Pytorch</strong></td><td><strong>Pytorch</strong></td><td><strong>TensorRT</strong></td><td><strong>TensorRT</strong></td><td><strong>TensorRT</strong></td>
		</tr>
		<tr>
			<td>Precision</td><td>FP32</td><td>FP16</td><td>FP32</td><td>FP16</td><td>Int8(PTQ)</td>
		</tr>
		<tr>
			<td>Avg Duration time [ms]</td>
			<td>66.21 ms</td>
			<td>34.58 ms</td>
			<td>40.81 ms </td>
			<td>13.52 ms</td>
			<td>8.19 ms</td>
		</tr>
		<tr>
			<td>FPS [frame/sec]</td>
			<td>15 fps</td>
			<td>29 fps</td>
			<td>25 fps</td>
			<td>77 fps</td>
			<td>125 fps</td>
		</tr>
		<tr>
			<td>Memory [GB]</td>
			<td>3.863 GB</td>
			<td>2.677 GB</td>
			<td>1.552 GB</td>
			<td>1.367 GB</td>
			<td>1.051 GB</td>
		</tr>
	</tbody>
</table>

***

## Object Detection model(ViT)
- DETR model (detr_trt.cpp) 
- additional preprocess (mean std normalization function)
- postprocess (show out detection result to the image)
- Match all results with PyTorch
- Comparison of calculation execution time of 100 iteration and GPU memory usage for one 500x500x3 image 

<table border="0"  width="100%">
	<tbody align="center">
		<tr>
			<td></td>
			<td><strong>Pytorch</strong></td><td><strong>Pytorch</strong></td><td><strong>TensorRT</strong></td><td><strong>TensorRT</strong></td><td><strong>TensorRT</strong></td>
		</tr>
		<tr>
			<td>Precision</td><td>FP32</td><td>FP16</td><td>FP32</td><td>FP16</td><td>Int8(PTQ)</td>
		</tr>
		<tr>
			<td>Avg Duration time [ms]</td>
			<td>37.03 ms</td>
			<td>30.71 ms</td>
			<td>16.40 ms </td>
			<td>6.07 ms</td>
			<td>5.30 ms</td>
		</tr>
		<tr>
			<td>FPS [frame/sec]</td>
			<td>27 fps</td>
			<td>33 fps</td>
			<td>61 fps</td>
			<td>165 fps</td>
			<td>189 fps</td>
		</tr>
		<tr>
			<td>Memory [GB]</td>
			<td>1.563 GB</td>
			<td>1.511 GB</td>
			<td>1.212 GB</td>
			<td>1.091 GB</td>
			<td>1.005 GB</td>
		</tr>
	</tbody>
</table>

***

## Object Detection model
- Yolov5s model (yolov5s.cpp) 
- Comparison of calculation execution time of 100 iteration and GPU memory usage for one 640x640x3 image resized & padded

<table border="0"  width="100%">
	<tbody align="center">
		<tr>
			<td></td>
			<td><strong>Pytorch</strong></td><td><strong>TensorRT</strong></td><td><strong>TensorRT</strong></td>
		</tr>
		<tr>
			<td>Precision</td><td>FP32</td><td>FP32</td><td>Int8(PTQ)</td>
		</tr>
		<tr>
			<td>Avg Duration time [ms]</td>
			<td>7.72 ms</td>
			<td>6.16 ms </td>
			<td>2.86 ms</td>
		</tr>
		<tr>
			<td>FPS [frame/sec]</td>
			<td>129 fps</td>
			<td>162 fps</td>
			<td>350 fps</td>
		</tr>
		<tr>
			<td>Memory [GB]</td>
			<td>1.670 GB</td>
			<td>1.359 GB</td>
			<td>0.920 GB</td>
		</tr>
	</tbody>
</table>

***

## Super-Resolution model
- Real-ESRGAN model (real-esrgan.cpp)
- RealESRGAN_x4plus.pth
- Scale up 4x (448x640x3 -> 1792x2560x3) 
- Comparison of calculation execution time of 100 iteration and GPU memory usage
- [update] RealESRGAN_x2plus model (set OUT_SCALE=2)

<table border="0"  width="100%">
	<tbody align="center">
		<tr>
			<td></td>
			<td><strong>Pytorch</strong></td><td><strong>Pytorch</strong></td><td><strong>TensorRT</strong></td><td><strong>TensorRT</strong></td>
		</tr>
		<tr>
			<td>Precision</td><td>FP32</td><td>FP16</td><td>FP32</td><td>FP16</td>
		</tr>
		<tr>
			<td>Avg Duration time [ms]</td>
			<td>4109 ms</td>
			<td>1936 ms</td>
			<td>2139 ms </td>
			<td>737 ms</td>
		</tr>
		<tr>
			<td>FPS [frame/sec]</td>
			<td>0.24 fps</td>
			<td>0.52 fps</td>
			<td>0.47 fps</td>
			<td>1.35 fps</td>
		</tr>
		<tr>
			<td>Memory [GB]</td>
			<td>5.029 GB</td>
			<td>4.407 GB</td>
			<td>3.807 GB</td>
			<td>3.311 GB</td>
		</tr>
	</tbody>
</table>

***

## Object Detection model 2
- Yolov6s model (yolov6.cpp)   
- Comparison of calculation execution time of 1000 iteration 
and GPU memory usage (with preprocess, without nms, 536 x 640 x 3)
<table border="0"  width="100%">
	<tbody align="center">
		<tr>
			<td></td>
			<td><strong>Pytorch</strong></td><td><strong>TensorRT</strong></td><td><strong>TensorRT</strong></td><td><strong>TensorRT</strong></td>
		</tr>
		<tr>
			<td>Precision</td><td>FP32</td><td>FP32</td><td>FP16</td><td>Int8(PTQ)</td>
		</tr>
		<tr>
			<td>Avg Duration time [ms]</td>
			<td>20.7 ms</td>
			<td>10.3 ms</td>
			<td>3.54 ms</td>
			<td>2.58 ms</td>
		</tr>
		<tr>
			<td>FPS [frame/sec]</td>
			<td>48.14 fps</td>
			<td>96.21 fps</td>
			<td>282.26 fps</td>
			<td>387.89 fps</td>
		</tr>
		<tr>
			<td>Memory [GB]</td>
			<td>1.582 GB</td>
			<td>1.323 GB</td>
			<td>0.956 GB</td>
			<td>0.913 GB</td>
		</tr>
	</tbody>
</table>

***

## Object Detection model 3 (in progress)
- Yolov7 model (yolov7.cpp)   

 
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
* yolov5 : <https://github.com/ultralytics/yolov5>
* real-esrgan : <https://github.com/xinntao/Real-ESRGAN>
* yolov6 : <https://github.com/meituan/YOLOv6>