# DDRNet

PyTorch implementation of paper: 

**[Deep Dual-resolution Networks for Real-time and Accurate Semantic Segmentation of Road Scenes]()**

![image](imgs/ddrnet-23.jpg)
## Requirements
- pytorch==1.7.0
- cuda==11.1

you can use [docker](https://www.docker.com/):

```shell
cd ${PROJECT}/docker/ && ./build.sh
cd ..
./docker/start.sh && ./docker/into.sh
```


##  Data preparation

You need to download the [Cityscapes](https://www.cityscapes-dataset.com/) dataset. and rename the folder `cityscapes`. 
```
└── data
  ├── cityscapes
  └── list
```

## VAL

Use the pretrained models in `${PROJECT}/pretrained_models` folder and  `eval.py` code.

```python
cd ${PROJECT}
python tools/eval.py --cfg experiments/cityscapes/ddrnet23_slim.yaml
```
## Our Metrics
All performance tests were done on `RTX 2060`

|  Model |  Params |  GFLOPS |MIoU% (on Cityscapes)   |  Pytorch inference speed(s) | TensorRT inference speed(s)   |
|---|---|---|---|---|---|
|  DDRNet23_slim | 7.57M  |  0.98G | 77.8  |  0.034 |  0.005 |
| DDRNet23  |  28.22M |  3.88G | 79.5  | 0.07  |  0.006 |
|  DDRNET-39 |  40.13M |  6.95G | 81.9  | -  |  - |
|  DDRNET-39-1.5X |  76.86M | 14.85G  |  82.4 | -  | -  |

For `real-time` applications, it is better to use DDRNet-23-slim or DDRNet-23 (DDRNet-23 is twice as wide as
DDRNet-23-slim).

To achieve a trade-off between resolution and inference speed, DDRNet-23 is the best choice.



**Note**
- with the `ALIGN_CORNERS: false` in `***.yaml` will reach higher accuracy.


## TRAIN


Use the imagenet pretrained models in `${PROJECT}/pretrained_models` folder, and then train the model.

```python
cd ${PROJECT}
python -m torch.distributed.launch --nproc_per_node=2 tools/train.py --cfg experiments/cityscapes/ddrnet23_slim.yaml
```


**Note**
- set the `ALIGN_CORNERS: true` in `***.yaml`, like the default setting in [HRNet-Semantic-Segmentation OCR](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR).
- Multi-scale with scales: 0.5,0.75,1.0,1.25,1.5,1.75. it runs too slow.
- you can change the `align_corners=True` with better performance, the default option is `False`

## Reference
[1] [HRNet-Semantic-Segmentation OCR branch](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR) 

[2] [the official repository](https://github.com/ydhongHIT/DDRNet)


## DDRNet-23- Bayer
In this version of DDRNet-23, we modified the input module of the network so, it takes as an input one channel image (noised by Bayer filter) with this shape (2048.1024.1). and trained the model on Cityscapes dataset.

![image](imgs/BAYERstuttgart_00_000000_000005_leftImg8bit.png)

All performance tests were done on `RTX 2060`

|  Model | MIoU% (on Cityscapes' test data)  | Pytorch inference speed(s) | TensorRT inference speed|
|---|---|---|---|
| DDRNet-23- Bayer | 75.7  |  0.075 |0.006   | 

## DDRNet-23-Bayer_RggB
This model is inherited from “DDRNet-23. Bayer” model, it takes as an input one channel image noised by Bayer filter with this shape (2048.1024.1), then extracts the input into 3 channels image (R, G1G2, B) with this shape (1024.512.3).

![image](imgs/untitled-drawing.png)
All performance tests were done on `RTX 2060`

|  Model | MIoU% (on Cityscapes' test data)  | Pytorch inference speed(s) | TensorRT inference speed|
|---|---|---|---|
| DDRNet-23- Bayer_RggB| 70.4  |  0.072 |-   | 